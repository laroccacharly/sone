// sone-vision: zero-shot choice decisions about an image, scored by a small vision LM on llama.cpp.
//
// Each question ends in an "ANSWER:" slot. Nothing is generated: the LM reads the image and the
// question once, and each option is scored by how much the LM wants to write that option's
// token right after "ANSWER:". One forward pass, no generation, no training.
//
// Steps 1-3 (slot, option tokens, scores) are in Scorer::score below. Step 4, the softmax over
// each question's options, is softmax_over_options in sone/questions.py.
//
// Protocol: a long-lived process driven by sone/vision.py over stdin/stdout, one JSON object
// per line. On startup it prints {"marker": ..., "load_ms": ...}; the marker is the text
// vision.py puts in the prompt where the image goes. Each request then looks like
//
//   {"image": "/tmp/x.jpg", "prompt": "...<marker>\n",
//    "questions": [{"text": "...ANSWER:", "options": ["none", "person"]}]}
//
// and gets back {"tokens": 129, "ms": 1250, "logits": [[6.79, 5.73]]}, or {"error": "..."}.
//
// usage: sone-vision -m model.gguf --mmproj mmproj.gguf [-t threads] [--image-max-tokens N]

#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

static int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static void reply(const json & message) {
    std::cout << message.dump() << std::endl;
}

// The token the LM would write for `option` right after "ANSWER:": " person", with a leading space.
static llama_token option_token(const llama_vocab * vocab, const std::string & option) {
    std::string text = " " + option;
    llama_token ids[8];
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), ids, 8, false, false);
    if (n != 1) {
        throw std::runtime_error("option '" + option + "' must be a single token, but it splits into " +
                                 std::to_string(n < 0 ? -n : n));
    }
    return ids[0];
}

struct Scorer {
    llama_model * model;
    llama_context * ctx;
    mtmd_context * vision_ctx;  // the vision encoder and projector (mtmd: llama.cpp's multimodal library)
    int n_batch;
    llama_pos n_past = 0;  // positions already in the LM's context
    size_t n_tokens = 0;   // tokens read for the current request, image included

    json score(const json & request) {
        int64_t start = now_ms();
        llama_memory_clear(llama_get_memory(ctx), true);
        n_past = 0;
        n_tokens = 0;
        read_prompt(request.at("prompt"), request.at("image"));

        const llama_vocab * vocab = llama_model_get_vocab(model);
        json logits_per_question = json::array();
        for (const auto & question : request.at("questions")) {
            // Values below are real, from the "person" question on a photo of two cats with
            // LFM2.5-VL-450M: d = 1024 (hidden size), vocab = 65,536 tokens.
            //
            // 1. The slot: the LM's last hidden vector [1024], at the ":" of "ANSWER:".
            //    It encodes "what token should come next", i.e. the LM's answer. Each question
            //    is read on its own, so the slot is its last token, the one position llama.cpp
            //    computes output for.
            read_question(question.at("text"));

            // 2. One token per option, each a row of the LM's output embedding table [65536, 1024].
            //    "none"   -> " none"   -> token 10735
            //    "person" -> " person" -> token 1811
            std::vector<llama_token> option_tokens;
            for (const std::string & option : question.at("options")) {
                option_tokens.push_back(option_token(vocab, option));
            }

            // 3. Score: dot product of each option's row with the slot.
            //    The LM's last layer already does this for every row of the table:
            //    [65536, 1024] @ [1024] -> vocab_logits: [65536], its next-token logits.
            //    We only read the options' entries:
            //    vocab_logits[10735] = 6.79 (none), vocab_logits[1811] = 5.73 (person)
            const float * vocab_logits = llama_get_logits_ith(ctx, -1);
            json option_logits = json::array();
            for (llama_token token : option_tokens) {
                option_logits.push_back(vocab_logits[token]);
            }

            // 4. Softmax over this question's options only, in sone/questions.py:
            //    [6.79, 5.73] -> [0.74, 0.26] -> "none"
            logits_per_question.push_back(option_logits);
        }
        return {{"tokens", n_tokens}, {"ms", now_ms() - start}, {"logits", logits_per_question}};
    }

    // The shared prompt, with the image encoded at the marker. No output needed yet.
    void read_prompt(const std::string & text, const std::string & image_path) {
        mtmd::bitmap image(mtmd_helper_bitmap_init_from_file(vision_ctx, image_path.c_str(), false,
                                                              mtmd_helper_init_opt_default()).bitmap);
        if (!image.ptr) {
            throw std::runtime_error("failed to load image " + image_path);
        }
        read(text, image.ptr.get(), /* add_special */ true, /* logits_last */ false);
    }

    // One question, appended after the prompt and the questions before it.
    // Its last token is the "ANSWER:" slot, so ask for the logits there.
    void read_question(const std::string & text) {
        read(text, nullptr, /* add_special */ false, /* logits_last */ true);
    }

    // Tokenize `text` (with `image` at the marker, if any) and run it through the LM after n_past.
    void read(const std::string & text, const mtmd_bitmap * image, bool add_special, bool logits_last) {
        mtmd_input_text input = {};
        input.text = text.c_str();
        input.text_len = text.size();
        input.add_special = add_special;
        input.parse_special = true;
        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        const mtmd_bitmap * images[] = {image};
        if (mtmd_tokenize(vision_ctx, chunks.ptr.get(), &input, images, image ? 1 : 0) != 0) {
            throw std::runtime_error("failed to tokenize the prompt");
        }
        n_tokens += mtmd_helper_get_n_tokens(chunks.ptr.get());
        llama_pos new_n_past = n_past;
        if (mtmd_helper_eval_chunks(vision_ctx, ctx, chunks.ptr.get(), n_past, 0, n_batch, logits_last,
                                    &new_n_past) != 0) {
            throw std::runtime_error("failed to evaluate the prompt");
        }
        n_past = new_n_past;
    }

    void free() {
        mtmd_free(vision_ctx);
        llama_free(ctx);
        llama_model_free(model);
    }
};

// Load the LM and its vision encoder, CPU only.
static Scorer load(const std::string & model_path, const std::string & mmproj_path, int threads,
                   int image_max_tokens) {
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = [](float, void *) { return true; };  // no progress dots
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) throw std::runtime_error("failed to load " + model_path);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 2048;
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) throw std::runtime_error("failed to create the LM context");

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = false;
    mparams.n_threads = threads;
    mparams.print_timings = false;
    if (image_max_tokens > 0) mparams.image_max_tokens = image_max_tokens;
    mtmd_context * vision_ctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);
    if (!vision_ctx) throw std::runtime_error("failed to load " + mmproj_path);

    return {model, ctx, vision_ctx, (int) cparams.n_batch};
}

int main(int argc, char ** argv) {
    std::string model_path, mmproj_path;
    int threads = (int) std::thread::hardware_concurrency();
    int image_max_tokens = -1;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string flag = argv[i];
        if (flag == "-m") model_path = argv[i + 1];
        else if (flag == "--mmproj") mmproj_path = argv[i + 1];
        else if (flag == "-t") threads = std::stoi(argv[i + 1]);
        else if (flag == "--image-max-tokens") image_max_tokens = std::stoi(argv[i + 1]);
    }
    if (model_path.empty() || mmproj_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf --mmproj mmproj.gguf [-t threads] [--image-max-tokens N]\n", argv[0]);
        return 1;
    }

    // stdout carries the protocol; only errors go to stderr.
    auto log_errors = [](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    };
    llama_log_set(log_errors, nullptr);
    mtmd_helper_log_set(log_errors, nullptr);
    llama_backend_init();

    int64_t start = now_ms();
    Scorer scorer;
    try {
        scorer = load(model_path, mmproj_path, threads, image_max_tokens);
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    std::string image_marker = mtmd_get_marker(scorer.vision_ctx);
    reply({{"marker", image_marker}, {"load_ms", now_ms() - start}});
    for (std::string line; std::getline(std::cin, line);) {
        try {
            reply(scorer.score(json::parse(line)));
        } catch (const std::exception & e) {
            reply({{"error", e.what()}});
        }
    }

    scorer.free();
    return 0;
}
