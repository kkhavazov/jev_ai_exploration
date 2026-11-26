"""Classify a book summary with OpenAI Decisions. Run: python example_openai.py"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROBABILITY_THRESHOLD = 0.20


def main():
    data_dir = Path(__file__).resolve().parent
    load_dotenv(data_dir / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in your .env file first.")

    categories = json.loads((data_dir / "categories.json").read_text(encoding="utf-8"))

    choices = [
        {"value": category["name"], "description": category["description"]}
        for category in categories
        if category["enabled"]
    ]
    if not 2 <= len(choices) <= 255:
        raise SystemExit("Enable between 2 and 255 categories for classification.")

    # Edit review.txt to classify another book.
    review = (data_dir / "review.txt").read_text(encoding="utf-8").strip()

    with OpenAI(api_key=api_key) as client:
        response = client.decisions.create(
            model="gpt-6-luna",
            input=review,
            questions=[
                {
                    "type": "choice",
                    "name": "book_category",
                    "instructions": (
                        "Select the single best category for the book described "
                        "in this summary, using the category descriptions."
                    ),
                    "choices": choices,
                }
            ],
        )

    if not response.answers:
        raise SystemExit("The API returned no classification.")
    answer = response.answers[0]
    if answer.type == "refusal":
        raise SystemExit("The model declined to classify this text.")
    if answer.type != "choice":
        raise SystemExit(f"Unexpected answer type: {answer.type}")

    print(f"Summary: {review.strip()}")
    # Each category has a probability; confidence applies to the whole choice.
    matches = sorted(
        (
            category for category in answer.probabilities
            if category.probability > PROBABILITY_THRESHOLD
        ),
        key=lambda category: category.probability,
        reverse=True,
    )
    print(f"Categories with probability above {PROBABILITY_THRESHOLD:.0%}:")
    for category in matches:
        print(f"- {category.value}: {category.probability:.1%}")
    if not matches:
        print("No categories exceed the threshold.")


if __name__ == "__main__":
    main()
