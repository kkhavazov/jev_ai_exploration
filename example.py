"""Classify a book summary with Jev. Run: python example.py"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Choice, TypeSafeClient

PROBABILITY_THRESHOLD = 0.20


def main():
    data_dir = Path(__file__).resolve().parent
    load_dotenv(data_dir / ".env")
    api_key = os.getenv("JEV_AI_KEY")
    if not api_key:
        raise SystemExit("Set JEV_AI_KEY in your .env file first.")

    categories = json.loads((data_dir / "categories.json").read_text(encoding="utf-8"))

    criteria = {
        category["name"]: category["description"]
        for category in categories
        if category["enabled"]
    }
    if not criteria:
        raise SystemExit("Enable at least one category for classification.")

    # Edit review.txt to classify another book.
    review = (data_dir / "review.txt").read_text(encoding="utf-8").strip()

    with TypeSafeClient(api_key=api_key, model="jev-latest") as client:
        response = client.system_one(
            state=review,
            questions={
                "book_category": Choice(
                    instructions=(
                        "Select the single best category for the book described "
                        "in this summary, using the category descriptions."
                    ),
                    criteria=criteria,
                )
            },
        )

    answer = response.choices.get("book_category")
    if answer is None:
        raise SystemExit("The API returned no book category classification.")

    print(f"Summary: {review.strip()}")
    # Jev returns a mapping of category names to probabilities.
    matches = sorted(
        (
            (name, probability)
            for name, probability in answer.probabilities.items()
            if probability > PROBABILITY_THRESHOLD
        ),
        key=lambda category: category[1],
        reverse=True,
    )
    print(f"Categories with probability above {PROBABILITY_THRESHOLD:.0%}:")
    for name, probability in matches:
        print(f"- {name}: {probability:.1%}")
    if not matches:
        print("No categories exceed the threshold.")


if __name__ == "__main__":
    main()
