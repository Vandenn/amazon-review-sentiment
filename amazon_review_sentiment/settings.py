import os
import pathlib
import torch

SEED_VALUE = 42
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
EMBEDDINGS_FOLDER = os.path.join(PROJECT_ROOT, "embeddings")
MODELS_FOLDER = os.path.join(PROJECT_ROOT, "models")


def data_path(path, *paths):
    return construct_path(DATA_FOLDER, path, *paths)


def embeddings_path(path, *paths):
    return construct_path(EMBEDDINGS_FOLDER, path, *paths)


def models_path(path, *paths):
    return construct_path(MODELS_FOLDER, path, *paths)


def construct_path(folder_dir, path, *paths):
    constructed_path = os.path.join(folder_dir, path, *paths)
    directory = os.path.dirname(constructed_path)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    return constructed_path