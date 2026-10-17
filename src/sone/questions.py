"""The question schema and its prompt text, shared by vision.py and the experiments (no torch)."""

import math
from dataclasses import dataclass
from typing import NewType

QuestionId = NewType("QuestionId", str)  # e.g. "department"
OptionId = NewType("OptionId", str)  # e.g. "billing"


# --- Input -------------------------------------------------------------------


@dataclass
class ChoiceQuestion:
    instructions: str
    criteria: dict[OptionId, str]  # option -> description


Questions = dict[QuestionId, ChoiceQuestion]


def render(question_id: QuestionId, question: ChoiceQuestion) -> str:
    """The question's prompt text. It ends with the "ANSWER:" slot that gets scored."""
    criteria = question.criteria
    lines = [f"\nFIELD {question_id}: {question.instructions}"]
    lines += [f"- {option}: {description}" for option, description in criteria.items()]
    lines.append(f"Pick one of: {', '.join(criteria)}.\nANSWER:")
    return "\n".join(lines)


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


def softmax_over_options(options: list[OptionId], logits: list[float]) -> ChoiceAnswer:
    """Step 4: softmax over this question's options only, e.g. [6.79, 5.73] -> [0.74, 0.26]."""
    top = max(logits)
    weights = [math.exp(logit - top) for logit in logits]
    probabilities = [weight / sum(weights) for weight in weights]
    best = probabilities.index(max(probabilities))
    return ChoiceAnswer(
        choice=options[best],
        confidence=round(probabilities[best], 3),
        probabilities=dict(zip(options, (round(p, 3) for p in probabilities))),
    )
