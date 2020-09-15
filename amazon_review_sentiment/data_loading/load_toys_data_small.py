import json
import torch
from torchtext import data

from amazon_review_sentiment import settings

FILE_NAME = "toys_small"


def convert_file_to_proper_json(path):
    orig_json = open(path, "r")
    result_file = open(f"{path}.proper", "w")
    for line in orig_json:
        line_data = json.loads(line)
        if "reviewText" in line_data and "overall" in line_data:
            result_file.write(json.dumps(json.loads(line)) + "\n")


def load_toys_data_small():
    convert_file_to_proper_json(settings.data_path(f"{FILE_NAME}.json"))

    TEXT = data.Field(tokenize = "spacy")
    LABEL = data.LabelField(dtype = torch.float)

    toys_data_small = data.TabularDataset(
        path=settings.data_path(f"{FILE_NAME}.json.proper"), format="json",
        fields={"reviewText": ("text", TEXT), "overall": ("label", LABEL)}
    )

    return toys_data_small


if __name__ == "__main__":
    toys_data_small = load_toys_data_small()
    print(f"Data count: {len(toys_data_small.examples)}")
    print(f"Example: {vars(toys_data_small.examples[0])}")

