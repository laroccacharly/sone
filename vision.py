# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub", "pillow"]
# ///
"""Step 1, with an image as the state, on llama.cpp: fast enough for a Raspberry Pi 5 CPU.

Same decision model as sone.py: the image, then each question ending in an "ANSWER:" slot,
scored by the LM's logits for each option's single token, softmaxed over the options only.
One pass, no generation, no training. The forward pass runs in build/sone-vision (build it
with `uv run build.py`); this script renders the prompt, starts the binary once and sends
it one image at a time. vision-torch.py is the original PyTorch version.
"""

import argparse
import json
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

from questions import (
    ChoiceQuestion,
    OptionId,
    Prediction,
    QuestionId,
    Questions,
    choice_answer,
    render,
)

BINARY = Path(__file__).resolve().parent / "build" / "sone-vision"


@dataclass
class Model:
    repo: str  # Hugging Face repo with the GGUF files
    weights: str  # the language model
    mmproj: str  # the vision encoder and projector
    max_side: int  # images are downscaled to this before encoding; fewer pixels, fewer tokens
    image_max_tokens: int | None = None


MODELS = {
    # ~1.3 s per image on a Raspberry Pi 5; same answers when the options are reordered.
    "lfm": Model(
        "LiquidAI/LFM2.5-VL-450M-GGUF",
        "LFM2.5-VL-450M-Q8_0.gguf",
        "mmproj-LFM2.5-VL-450m-F16.gguf",
        max_side=256,
    ),
    # ~1.7 s per image on a Raspberry Pi 5, but picks whichever option is listed first.
    "qwen": Model(
        "unsloth/Qwen3.5-0.8B-GGUF",
        "Qwen3.5-0.8B-Q8_0.gguf",
        "mmproj-F16.gguf",
        max_side=640,
        image_max_tokens=64,
    ),
}


def load_image(source: str, max_side: int) -> Image.Image:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as response:
            source = BytesIO(response.read())
    image = Image.open(source).convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


class VisionDecisionModel:
    def __init__(self, model: Model = MODELS["lfm"], threads: int | None = None):
        if not BINARY.exists():
            raise FileNotFoundError(f"{BINARY} is missing; build it with `uv run build.py`")
        command = [
            str(BINARY),
            "-m", hf_hub_download(model.repo, model.weights),
            "--mmproj", hf_hub_download(model.repo, model.mmproj),
        ]
        if threads:
            command += ["-t", str(threads)]
        if model.image_max_tokens:
            command += ["--image-max-tokens", str(model.image_max_tokens)]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        ready = self.receive()
        self.marker: str = ready["marker"]  # where the image goes in the prompt
        self.load_ms: int = ready["load_ms"]

    def receive(self) -> dict:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"sone-vision exited with code {self.process.wait()}")
        message = json.loads(line)
        if "error" in message:
            raise RuntimeError(message["error"])
        return message

    def predict(self, image: Image.Image, questions: Questions) -> Prediction:
        with tempfile.NamedTemporaryFile(suffix=".png") as file:
            image.save(file.name)
            request = {
                "image": file.name,
                "prompt": f"Look at the image and answer each field.\n\nIMAGE:\n{self.marker}\n",
                # Each question is evaluated on its own, so its ANSWER: slot is its last token.
                "questions": [
                    {"text": ("\n" if k else "") + render(question_id, question),
                     "options": list(question.criteria)}
                    for k, (question_id, question) in enumerate(questions.items())
                ],
            }
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            response = self.receive()
        answers = {
            question_id: choice_answer(list(question.criteria), logits)
            for (question_id, question), logits in zip(questions.items(), response["logits"])
        }
        return Prediction(answers=answers, input_tokens=response["tokens"])

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait()


EXAMPLE_QUESTIONS: Questions = {
    QuestionId("person"): ChoiceQuestion(
        instructions="Is there a person in the image?",
        # Order matters for Qwen: listed first, "person" wins even on an image of two cats.
        criteria={
            OptionId("none"): "no person is visible",
            OptionId("person"): "at least one person is visible",
        },
    ),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", default=[
        "http://images.cocodataset.org/val2017/000000000785.jpg",  # a skier
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # two cats on a couch
    ])
    parser.add_argument("--model", choices=MODELS, default="lfm")
    parser.add_argument("--max-side", type=int, help="default: the model's max_side")
    parser.add_argument("--threads", type=int, help="default: all hardware threads")
    parser.add_argument("--repeat", type=int, default=1, help="runs per image; the first is slower")
    args = parser.parse_args()
    model = MODELS[args.model]
    decision_model = VisionDecisionModel(model, args.threads)
    print(f"{args.model}: loaded in {decision_model.load_ms} ms")
    for source in args.images:
        image = load_image(source, args.max_side or model.max_side)
        for _ in range(args.repeat):
            start = time.perf_counter()
            prediction = decision_model.predict(image, EXAMPLE_QUESTIONS)
            elapsed = time.perf_counter() - start
            print(f"{source} {image.size} {prediction.input_tokens} tokens {elapsed * 1000:.0f} ms")
            for question_id, answer in prediction.answers.items():
                print(f"  {question_id}: {answer.choice} {answer.probabilities}")
    decision_model.close()
