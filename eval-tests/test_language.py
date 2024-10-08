#
# Copyright (c) 2022
#

import os
import pytest
import time

from typing import Any, List

from prime.proxy import Proxy, _client
from prime.utils import run_server, kill_server


@pytest.fixture
def baseline(pytestconfig):
    return pytestconfig.getoption("--baseline")


@pytest.fixture
def task(pytestconfig):
    return pytestconfig.getoption("--task")


@pytest.fixture
def model(pytestconfig):
    return pytestconfig.getoption("--model")


@pytest.fixture
def max_epochs(pytestconfig):
    return pytestconfig.getoption("--max_epochs")


################################################################################
# Init server before starting tests                                            #
################################################################################


def test_init_server(baseline: bool, task: str):

    datasets = {
        "sentiment-analysis": "emotion",
        "language-modeling": "wikitext",
        "translation": "wmt16",
    }

    if baseline:
        kill_server()

    elif "PRIMEIPADDR" in os.environ and "PRIMEPORT" in os.environ:
        kill_server()
        time.sleep(1)

        if not _client.check_server():
            raise Exception("Server not running")

    else:
        kill_server()
        run_server(dn=datasets[task], ll="ERROR")

        time.sleep(1)
        if not _client.check_server():
            raise Exception("Server not running")


################################################################################


def import_libs(baseline: bool):
    pass


def sample_init(baseline: bool, dataset: str) -> tuple:
    return


def load_dataset(task: str) -> "datasets.arrow_dataset.Dataset":
    from datasets import load_dataset

    dataset_kwargs = {
        "sentiment-analysis": dict(path="emotion"),
        "language-modeling": dict(path="wikitext", name="wikitext-2-raw-v1"),
        "translation": dict(path="wmt16", name="ro-en"),
    }

    dataset = load_dataset(**dataset_kwargs[task])
    return dataset["train"]


def transform_dataset(
    ds: "datasets.arrow_dataset.Dataset", task: str, model: str
) -> "datasets.arrow_dataset.Dataset":
    from transformers import AutoTokenizer

    tokenizer_kwargs = {
        "sentiment-analysis": dict(),
        "language-modeling": dict(),
        "translation": dict(model_max_length=512),
    }

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model, **tokenizer_kwargs
    )

    if task == "sentiment-analysis":

        def convert_to_features(examples: Any, _):
            texts = examples["text"]
            return tokenizer(texts, max_length=512, padding="max_length")

        ds = ds.map(convert_to_features, batched=True, with_indices=True)
        ds = ds.rename_column("label", "labels")
        ds.set_format(
            "torch",
            columns=["input_ids", "attention_mask", "token_type_ids", "labels"],
        )

    elif task == "language-modeling":

        def tokenizer_function(examples: Any):
            return tokenizer(examples["text"])

        ds = ds.map(tokenizer_function, batched=True, remove_columns=["text"])

        def convert_to_features(examples: Any):
            block_size = 1024

            concatenated_examples = {
                k: sum(examples[k], []) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])

            total_length = (total_length // block_size) * block_size
            result = {
                k: [
                    t[i : i + block_size]
                    for i in range(0, total_length, block_size)
                ]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        ds = ds.map(convert_to_features, batched=True)
        ds.set_format("torch")

    elif task == "translation":

        def convert_to_features(examples: Any):
            source_language = "en"
            target_language = "ro"
            max_source_length = 128
            max_target_length = 128
            padding = "max_length"

            inputs = [ex[source_language] for ex in examples["translation"]]
            targets = [ex[target_language] for ex in examples["translation"]]
            model_inputs = tokenizer(
                inputs,
                max_length=max_target_length,
                padding=padding,
                truncation=True,
            )

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=max_target_length,
                    padding=padding,
                    truncation=True,
                )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        ds = ds.map(convert_to_features, batched=True)
        ds.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )

    else:
        raise RuntimeError(f"Unknown task: {task}")

    return ds


def build_model(task: str, model: str):
    from lightning_transformers.task.nlp.text_classification import (
        TextClassificationTransformer,
    )
    from lightning_transformers.task.nlp.language_modeling import (
        LanguageModelingTransformer,
    )
    from lightning_transformers.task.nlp.translation import (
        TranslationTransformer,
    )

    if task == "sentiment-analysis":
        model = TextClassificationTransformer(
            pretrained_model_name_or_path=model, num_labels=6
        )
    elif task == "language-modeling":
        model = LanguageModelingTransformer(pretrained_model_name_or_path=model)

        model.on_fit_start = None

    elif task == "translation":
        model = TranslationTransformer(
            pretrained_model_name_or_path=model,
            n_gram=4,
            smooth=False,
            val_target_max_length=142,
            num_beams=None,
            compute_generate_metrics=None,
        )
    else:
        raise RuntimeError(f"Unknown task: {task}")

    model.validation_step = None
    model.test_step = None

    return model


def test_language(baseline: bool, task: str, model: str, max_epochs: str):

    from torch.utils.data import DataLoader
    from pytorch_lightning import Trainer

    assert task in ("sentiment-analysis", "language-modeling", "translation")
    assert model in ("bert-base-cased", "bert-large-cased", "gpt2", "t5-base")

    dataset = load_dataset(task)

    dataset = transform_dataset(dataset, task, model)
    dataloader = DataLoader(dataset, batch_size=1)

    model = build_model(task, model)

    trainer = Trainer(accelerator="auto", devices="auto", max_epochs=1)
    trainer.fit(model, train_dataloaders=dataloader)

################################################################################
# Kill server after all tests are completed                                    #
################################################################################


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def _kill_server():
        try:
            kill_server()
        except:
            pass

    request.addfinalizer(_kill_server)
