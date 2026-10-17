# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["torch", "transformers>=4.51"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Step 1: a minimal System 1 decision model. Inference only, `choice` questions only.

A state and any number of choice questions go into ONE prompt and ONE forward pass.
Each question ends with an "ANSWER:" slot. The backbone's hidden state at that slot
is scored against a vector for each option name, so the number of options and
questions comes from the input, not from the weights. No text is generated.

Nothing is trained: the scores come from the LM's own output layer. The price is that
every option name must be exactly one token in the vocabulary (" billing" is,
" account_access" is not), because the slot only predicts one next token.
"""

import sys
from dataclasses import dataclass
from typing import NewType

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"

QuestionId = NewType("QuestionId", str)  # e.g. "department"
OptionId = NewType("OptionId", str)  # e.g. "billing"
TokenId = NewType("TokenId", int)  # an entry of the tokenizer's vocabulary
Position = NewType("Position", int)  # an index into the encoded token sequence


# --- Input -------------------------------------------------------------------


@dataclass
class ChoiceQuestion:
    instructions: str
    criteria: dict[OptionId, str]  # option -> description


Questions = dict[QuestionId, ChoiceQuestion]


# --- Encoding ----------------------------------------------------------------


@dataclass
class Field:
    """Where one question lives in the encoded sequence."""

    question_id: QuestionId
    answer_position: Position  # the token of this question's ANSWER: slot
    options: dict[OptionId, TokenId]  # option -> the single token of its name


@dataclass
class Encoded:
    ids: list[TokenId]
    fields: list[Field]


# --- Output ------------------------------------------------------------------


@dataclass
class ChoiceAnswer:
    choice: OptionId
    confidence: float
    probabilities: dict[OptionId, float]


@dataclass
class Prediction:
    answers: dict[QuestionId, ChoiceAnswer]
    input_tokens: int
    output_tokens: int = 0  # a decision model never generates


def encode_questions(tokenizer, ids: list[TokenId], questions: Questions) -> Encoded:
    """Append each question to an already encoded prefix and remember where its answer slot is."""

    def tokens(text: str) -> list[TokenId]:
        return [TokenId(t) for t in tokenizer.encode(text)]

    def single_token(option: OptionId) -> TokenId:
        # The token that would follow "ANSWER:" is " billing", with a leading space.
        ids = tokens(f" {option}")
        if len(ids) != 1:
            pieces = [tokenizer.decode([i]) for i in ids]
            raise ValueError(
                f"option {option!r} must be a single token, but it splits into {pieces}"
            )
        return ids[0]

    ids = list(ids)
    fields: list[Field] = []
    for question_id, question in questions.items():
        criteria = question.criteria
        lines = [f"\nFIELD {question_id}: {question.instructions}"]
        lines += [f"- {option}: {description}" for option, description in criteria.items()]
        lines.append(f"Pick one of: {', '.join(criteria)}.\nANSWER:")
        ids += tokens("\n".join(lines))
        fields.append(
            Field(
                question_id=question_id,
                answer_position=Position(len(ids) - 1),
                options={option: single_token(option) for option in criteria},
            )
        )
        ids += tokens("\n")
    return Encoded(ids=ids, fields=fields)


def encode(tokenizer, state: str, questions: Questions) -> Encoded:
    """Render state + schema into token ids and remember where each answer slot is."""
    prefix = tokenizer.encode(f"Read the state and answer each field.\n\nSTATE:\n{state}\n")
    return encode_questions(tokenizer, [TokenId(t) for t in prefix], questions)


def score(
    hidden: torch.Tensor, fields: list[Field], option_table: torch.Tensor
) -> dict[QuestionId, ChoiceAnswer]:
    """Score each field's answer slot against its options' rows of the output embedding."""
    answers: dict[QuestionId, ChoiceAnswer] = {}
    for field in fields:
        # Shapes below use real values from the billing example, question "department".
        # d = 1024 (Qwen3-0.6B hidden size), vocab = 151,936, sequence = 122 tokens.
        #
        # 1. The slot: the backbone's vector at the ":" of "ANSWER:".
        #    hidden: [122, 1024] -> slot: [1024] = [-2.63, -14.91, -1.46, ...]
        #    It encodes "what token should come next", i.e. the model's answer.
        slot = hidden[field.answer_position]

        # 2. One vector per option, looked up in the LM's output embedding table.
        #    option_table: [151936, 1024], one row per vocabulary token.
        #    "billing"   -> token 33531 (" billing")   -> row [1024]
        #    "technical" -> token 10916 (" technical") -> row [1024]
        #    "other"     -> token 1008  (" other")     -> row [1024]
        #    Stacked: options: [3, 1024], one row per option. This is the old "W",
        #    but built from the question at inference time, so it can have any number of rows.
        option_ids = list(field.options)  # ["billing", "technical", "other"]
        options = option_table[list(field.options.values())]

        # 3. Score: dot product of every option row with the slot.
        #    [3, 1024] @ [1024] -> logits: [3] = [19.79, 13.79, 17.90]
        #    Toy version with d = 3:
        #        slot    =  [1, 0, 2]
        #        options = [[2, 0, 1],   billing:   1*2 + 0*0 + 2*1 = 4
        #                   [0, 1, 0],   technical: 1*0 + 0*1 + 2*0 = 0
        #                   [1, 1, 1]]   other:     1*1 + 0*1 + 2*1 = 3
        #        logits  =  [4, 0, 3]
        #    Because the backbone's final layer is followed by exactly this table,
        #    each logit is the LM's score for that option's name being the next token.
        logits = options.float() @ slot.float()

        # 4. Softmax over this question's options only: exp, then divide by the sum.
        #    Toy: exp([4, 0, 3]) = [54.6, 1.0, 20.1], sum 75.7 -> [0.72, 0.01, 0.27]
        #    Real: [19.79, 13.79, 17.90] -> [0.87, 0.00, 0.13]
        probabilities = logits.softmax(-1)
        best = int(probabilities.argmax())
        answers[field.question_id] = ChoiceAnswer(
            choice=option_ids[best],
            confidence=round(float(probabilities[best]), 3),
            probabilities=dict(zip(option_ids, (round(float(p), 3) for p in probabilities))),
        )
    return answers


class DecisionModel:
    def __init__(self, name: str = MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.backbone = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
        # The LM's output embedding: one vector per vocabulary token.
        output_embeddings = self.backbone.get_output_embeddings()
        assert output_embeddings is not None
        self.option_table: torch.Tensor = output_embeddings.weight  # [vocab, d]

    @torch.inference_mode()
    def predict(self, state: str, questions: Questions) -> Prediction:
        encoded = encode(self.tokenizer, state, questions)
        hidden = self.backbone.model(torch.tensor([encoded.ids])).last_hidden_state[0]  # [seq, d]
        answers = score(hidden, encoded.fields, self.option_table)
        return Prediction(answers=answers, input_tokens=len(encoded.ids))


EXAMPLE_QUESTIONS: Questions = {
    QuestionId("department"): ChoiceQuestion(
        instructions="Which department should handle this?",
        criteria={
            OptionId("billing"): "invoices, payments, refunds",
            OptionId("technical"): "bugs, outages, system errors",
            OptionId("other"): "everything else",
        },
    ),
    QuestionId("sentiment"): ChoiceQuestion(
        instructions="What is the customer's tone?",
        criteria={
            OptionId("angry"): "upset or threatening",
            OptionId("neutral"): "matter of fact",
            OptionId("happy"): "pleased",
        },
    ),
}

if __name__ == "__main__":
    model = DecisionModel()
    states = sys.argv[1:] or [
        "We were billed twice for March. Refund the duplicate today or we cancel our plan.",
        "The app crashes every time I open the settings page.",
        "Thanks, the new dashboard is great!",
    ]
    for state in states:
        print(state)
        for question_id, answer in model.predict(state, EXAMPLE_QUESTIONS).answers.items():
            print(f"  {question_id}: {answer.choice} {answer.probabilities}")
