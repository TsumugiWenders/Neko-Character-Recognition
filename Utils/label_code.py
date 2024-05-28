import numpy as np


def save_label(path: str, labels: list):
    with open(path, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(label)
            f.write('\n')


def load_label(path: str) -> list:

    with open(path, 'r', encoding='utf-8') as f:
        labels = f.read().split('\n')
        labels = [label for label in labels if label]

    return labels


def decode_label(result: np.ndarray, label: list) -> str:
    assert len(result) == len(label)

    index = result.argmax()
    return label[index]

