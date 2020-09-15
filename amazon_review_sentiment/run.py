import logging
import torch
from torch import nn, optim

from amazon_review_sentiment import settings
from amazon_review_sentiment.data import keys
from amazon_review_sentiment.data.toys_data_small import ToysDataSmall
from amazon_review_sentiment.models.model1 import Model1

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()
    ]
)


def train_model(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss/len(iterator), epoch_acc/len(iterator)


def evaluate_model(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss/len(iterator), epoch_acc/len(iterator)



def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


if __name__ == "__main__":
    torch.manual_seed(settings.SEED_VALUE)
    torch.backends.cudnn.enabled = True

    data_object = ToysDataSmall()
    data_text_vocab = data_object.get_text_vocab()
    train_iter = data_object.get_iter(keys.DATA_TYPE_TRAIN)
    val_iter = data_object.get_iter(keys.DATA_TYPE_VAL)
    test_iter = data_object.get_iter(keys.DATA_TYPE_TEST)
    model = Model1(
        input_dim=len(data_text_vocab),
        embedding_dim=300
    )

    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(settings.TORCH_DEVICE)
    criterion = criterion.to(settings.TORCH_DEVICE)

    epochs = 5
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, train_iter, optimizer, criterion)
        val_loss, val_acc = evaluate_model(model, val_iter, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), settings.MODELS_FOLDER("model1.pt"))

        logging.info(f"Epoch: {epoch+1}")
        logging.info(f"\nTrain Loss: {train_loss:.3f} | Train Acc.: {train_acc:.3f}%")
        logging.info(f"\nValidation Loss: {val_loss:.3f} | Validation Acc.: {val_acc:.3f}%")

    model.load_state_dict(torch.load(settings.MODELS_FOLDER("model1.pt")))
    test_loss, test_acc = evaluate_model(model, test_iter, criterion)
    logging.info(f"\nTest Loss: {test_loss:.3f} | Test Acc.: {test_acc:.3f}%")

