# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["torch", "torchvision", "transformers>=5.2", "pillow"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
# ///
"""Step 1, with an image as the state: the same decision model on a vision-language backbone.

The image goes through the VLM's vision encoder and lands in the prompt as image tokens.
The questions follow as text, each ending with an "ANSWER:" slot, and are scored exactly
as in sone.py: the slot's hidden state against the output-embedding rows of the option
names. One forward pass, no generation, no training.
"""

import argparse
import time
import urllib.request
from io import BytesIO

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from sone import (
    ChoiceQuestion,
    OptionId,
    Prediction,
    QuestionId,
    Questions,
    TokenId,
    encode_questions,
    score,
)

MODEL = "Qwen/Qwen3.5-0.8B"  # 1.75 GB in bf16
MAX_SIDE = 640  # longer images are downscaled; fewer pixels means fewer image tokens


def load_image(source: str, max_side: int = MAX_SIDE) -> Image.Image:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as response:
            source = BytesIO(response.read())
    image = Image.open(source).convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


class VisionDecisionModel:
    def __init__(self, name: str = MODEL, dtype: torch.dtype = torch.bfloat16):
        self.processor = AutoProcessor.from_pretrained(name)
        self.backbone = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype).eval()
        output_embeddings = self.backbone.get_output_embeddings()
        assert output_embeddings is not None
        self.option_table: torch.Tensor = output_embeddings.weight  # [vocab, d]

    @torch.inference_mode()
    def predict(self, image: Image.Image, questions: Questions) -> Prediction:
        # The processor expands the single image placeholder into one token per image patch.
        # Images are resized to multiples of 32 px, with at least ~130 image tokens.
        prefix = self.processor(
            text=[
                "Look at the image and answer each field.\n\n"
                f"IMAGE:\n<|vision_start|>{self.processor.image_token}<|vision_end|>\n"
            ],
            images=[image],
            return_tensors="pt",
            return_mm_token_type_ids=True,
        )
        encoded = encode_questions(
            self.processor.tokenizer,
            [TokenId(int(t)) for t in prefix["input_ids"][0]],
            questions,
        )
        # Which tokens are image patches (1) or text (0); the appended questions are text.
        token_types = prefix["mm_token_type_ids"]
        token_types = torch.cat(
            [token_types, token_types.new_zeros(1, len(encoded.ids) - token_types.shape[1])], 1
        )
        hidden = self.backbone.model(
            input_ids=torch.tensor([encoded.ids]),
            mm_token_type_ids=token_types,
            pixel_values=prefix["pixel_values"],
            image_grid_thw=prefix["image_grid_thw"],
        ).last_hidden_state[0]  # [seq, d]
        answers = score(hidden, encoded.fields, self.option_table)
        return Prediction(answers=answers, input_tokens=len(encoded.ids))


EXAMPLE_QUESTIONS: Questions = {
    QuestionId("person"): ChoiceQuestion(
        instructions="Is there a person in the image?",
        # Order matters: listed first, "person" wins even on an image of two cats.
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
    parser.add_argument("--max-side", type=int, default=MAX_SIDE)
    # CPUs without bf16 instructions (e.g. Raspberry Pi 5) take a very slow path in bf16.
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()
    model = VisionDecisionModel(dtype=getattr(torch, args.dtype))
    for source in args.images:
        image = load_image(source, args.max_side)
        start = time.perf_counter()
        prediction = model.predict(image, EXAMPLE_QUESTIONS)
        elapsed = time.perf_counter() - start
        print(f"{source} {image.size} {prediction.input_tokens} tokens {elapsed * 1000:.0f} ms")
        for question_id, answer in prediction.answers.items():
            print(f"  {question_id}: {answer.choice} {answer.probabilities}")
