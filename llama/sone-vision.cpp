// Step 1 on llama.cpp: the scoring half of vision.py, for CPUs where PyTorch is too slow.
//
// A long-lived process driven by vision.py over stdin/stdout, one JSON object per line.
// On startup it prints {"marker": ..., "load_ms": ...}; the marker is the text vision.py
// puts in the prompt where the image goes. Each request then looks like
//
//   {"image": "/tmp/x.jpg", "prompt": "...<marker>\n",
//    "questions": [{"text": "...ANSWER:", "options": ["none", "person"]}]}
//
// and gets back {"tokens": 129, "ms": 1250, "logits": [[-3.1, 4.2]]}, or {"error": "..."}.
// The prompt (image included) is evaluated first, then each question as its own text chunk,
// so its "ANSWER:" slot is the chunk's last token and `logits_last` gives exactly the LM's
// logits at the slot. Only the options' tokens are read; the softmax happens in vision.py.
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

static llama_token single_token(const llama_vocab * vocab, const std::string & option) {
    // The token that would follow "ANSWER:" is " person", with a leading space.
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
    mtmd_context * mctx;
    int n_batch;

    // Evaluate `text` (with `bitmap` at the marker, if any) after n_past positions.
    void eval(const std::string & text, const mtmd_bitmap * bitmap, bool add_special, bool logits_last,
              llama_pos & n_past, size_t & n_tokens) {
        mtmd_input_text input = {};
        input.text = text.c_str();
        input.text_len = text.size();
        input.add_special = add_special;
        input.parse_special = true;
        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        const mtmd_bitmap * bitmaps[] = {bitmap};
        if (mtmd_tokenize(mctx, chunks.ptr.get(), &input, bitmaps, bitmap ? 1 : 0) != 0) {
            throw std::runtime_error("failed to tokenize the prompt");
        }
        n_tokens += mtmd_helper_get_n_tokens(chunks.ptr.get());
        llama_pos new_n_past = n_past;
        if (mtmd_helper_eval_chunks(mctx, ctx, chunks.ptr.get(), n_past, 0, n_batch, logits_last, &new_n_past) != 0) {
            throw std::runtime_error("failed to evaluate the prompt");
        }
        n_past = new_n_past;
    }

    json score(const json & request) {
        int64_t start = now_ms();
        const llama_vocab * vocab = llama_model_get_vocab(model);
        std::string path = request.at("image");
        mtmd::bitmap image(mtmd_helper_bitmap_init_from_file(mctx, path.c_str(), false, mtmd_helper_init_opt_default()).bitmap);
        if (!image.ptr) {
            throw std::runtime_error("failed to load image " + path);
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_pos n_past = 0;
        size_t n_tokens = 0;
        eval(request.at("prompt"), image.ptr.get(), true, false, n_past, n_tokens);

        json logits = json::array();
        for (const auto & question : request.at("questions")) {
            std::vector<llama_token> options;
            for (const std::string & option : question.at("options")) {
                options.push_back(single_token(vocab, option));
            }
            eval(question.at("text"), nullptr, false, true, n_past, n_tokens);
            const float * slot = llama_get_logits_ith(ctx, -1);
            json row = json::array();
            for (llama_token option : options) {
                row.push_back(slot[option]);
            }
            logits.push_back(row);
        }
        return {{"tokens", n_tokens}, {"ms", now_ms() - start}, {"logits", logits}};
    }
};

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
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = [](float, void *) { return true; };  // no progress dots
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) return 1;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 2048;
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = false;
    mparams.n_threads = threads;
    mparams.print_timings = false;
    if (image_max_tokens > 0) mparams.image_max_tokens = image_max_tokens;
    mtmd_context * mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);
    if (!mctx) return 1;

    Scorer scorer = {model, ctx, mctx, (int) cparams.n_batch};
    reply({{"marker", mtmd_get_marker(mctx)}, {"load_ms", now_ms() - start}});
    for (std::string line; std::getline(std::cin, line);) {
        try {
            reply(scorer.score(json::parse(line)));
        } catch (const std::exception & e) {
            reply({{"error", e.what()}});
        }
    }

    mtmd_free(mctx);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
