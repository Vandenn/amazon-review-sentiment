import json
import random
import torch
from torchtext import data

from amazon_review_sentiment import settings
from amazon_review_sentiment.data import keys

FILE_NAME = "toys_small"


class ToysDataSmall:
    def __init__(self, max_vocab_size=25000, batch_size=32):
        torch.manual_seed(settings.SEED_VALUE)
        self.max_vocab_size = max_vocab_size
        self.batch_size = batch_size
        self.load_data()

    def get_data(self, type=keys.DATA_TYPE_ALL):
        if not self.data:
            self.load_data()
        return {
            keys.DATA_TYPE_ALL: self.data,
            keys.DATA_TYPE_TRAIN: self.train_data,
            keys.DATA_TYPE_VAL: self.val_data,
            keys.DATA_TYPE_TEST: self.test_data
        }[type]

    def get_iter(self, type):
        if not self.data:
            self.load_data()
        return {
            keys.DATA_TYPE_TRAIN: self.train_iter,
            keys.DATA_TYPE_VAL: self.val_iter,
            keys.DATA_TYPE_TEST: self.test_iter
        }[type]

    def load_data(self):
        self.convert_file_to_proper_json(settings.data_path(f"{FILE_NAME}.json"))

        TEXT = data.Field(tokenize = "spacy")
        LABEL = data.LabelField(dtype = torch.float)

        self.data = data.TabularDataset(
            path=settings.data_path(f"{FILE_NAME}.json.proper"), format="json",
            fields={"reviewText": ("text", TEXT), "overall": ("label", LABEL)}
        )

        self.train_data, self.val_data, self.test_data = self.data.split(
            split_ratio=[0.4, 0.3, 0.3],
            random_state = random.seed(settings.SEED_VALUE)
        )

        TEXT.build_vocab(self.train_data, max_size=self.max_vocab_size)
        LABEL.build_vocab(self.train_data)

        self.text_vocab = TEXT.vocab
        self.label_vocab = LABEL.vocab

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (self.train_data, self.val_data, self.test_data),
            batch_size=self.batch_size,
            device=device
        )

    def convert_file_to_proper_json(self, path):
        orig_json = open(path, "r")
        result_file = open(f"{path}.proper", "w")
        counter = 0
        for line in orig_json:
            counter += 1
            if counter > 10: break
            line_data = json.loads(line)
            if "reviewText" in line_data and "overall" in line_data:
                result_file.write(json.dumps(json.loads(line)) + "\n")



if __name__ == "__main__":
    toys_data_small_object = ToysDataSmall()
    toys_data_small = toys_data_small_object.get_data()
    print(f"Data count: {len(toys_data_small.examples)}")
    print(f"Example: {vars(toys_data_small.examples[0])}")
    print(f"Text vocab: {toys_data_small_object.text_vocab.itos[:10]}")
    print(f"Label vocab: {toys_data_small_object.label_vocab.stoi}")

