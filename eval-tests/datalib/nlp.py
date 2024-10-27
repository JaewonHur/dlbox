import datasets

from datasets import load_dataset, load_from_disk


def load_emotion() -> datasets.arrow_dataset.Dataset:
    dataset = load_dataset(path="emotion")

    return dataset["train"]

def load_wikitext() -> datasets.arrow_dataset.Dataset:
    dataset = load_dataset(path="wikitext", name="wikitext-2-raw-v1")
    
    return dataset["train"]


def load_wikipedia() -> datasets.arrow_dataset.Dataset:
    dataset = load_dataset(path="wikipedia", name="20220301.en")

    return dataset["train"]


def load_wmt16() -> datasets.arrow_dataset.Dataset:
    dataset = load_dataset(path="wmt16", name="ro-en")

    return dataset["train"]
