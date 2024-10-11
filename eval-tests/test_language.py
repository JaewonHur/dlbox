#
# Copyright (c) 2022
#

import os
import pytest
import inspect
import textwrap
import time

from typing import Any, List
from functools import partial
from transformers import AutoTokenizer

from prime.proxy import Proxy, _client
from prime.utils import run_server, kill_server

from lightning_transformers.task.nlp.text_classification import (
    TextClassificationTransformer,
)
from lightning_transformers.task.nlp.language_modeling import (
    LanguageModelingTransformer,
)
from lightning_transformers.task.nlp.translation import (
    TranslationTransformer,
)

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
        "language-modeling": "wikipedia",
        "translation": "wmt16",
    }
    dataset = datasets[task]

    sleep_times = {
        "emotion": 30,
        "wikipedia": 180,
        "wmt16": 30
    }
    sleep_time = sleep_times[dataset]

    if baseline:
        kill_server()

    elif "PRIMEIPADDR" in os.environ and "PRIMEPORT" in os.environ:
        kill_server()
        time.sleep(30)

        if not _client.check_server():
            raise Exception("Server not running")

    else:
        kill_server()
        run_server(dn=dataset, ll="DEBUG")

        time.sleep(sleep_time)
        if not _client.check_server():
            raise Exception("Server not running")


################################################################################

def export_def(baseline: bool):
    def decorator(func):
        if baseline:
            return func

        src = textwrap.dedent(inspect.getsource(func))
        src = '\n'.join(src.split('\n')[1:]) # Remove decorator

        ref = _client.ExportDef(f"__main__.{func.__name__}", type(func), src)
        return Proxy(f"__main__.{ref}")

    return decorator

def import_libs(baseline: bool):
    global DataLoader, Trainer
    
    if baseline:
        from torch.utils.data import DataLoader
        from pytorch_lightning import Trainer

    else:
        from prime_torch.utils.data import DataLoader
        from prime_pytorch_lightning import Trainer


def load_dataset(baseline: bool, task: str) -> "datasets.arrow_dataset.Dataset":
    from eval_tests.datalib.nlp import load_emotion, load_wikipedia, load_wmt16, load_wikipedia

    if baseline:
        if task == "sentiment-analysis":
            dataset = load_emotion()
        elif task == "language-modeling":
            dataset = load_wikipedia()
            # dataset = load_wikipedia()
        elif task == "translation":
            dataset = load_wmt16()

        else:
            raise RuntimeError(f"Unknown task: {task}")

    else:
        dataset = Proxy("_DATASET")

    return dataset


def transform_dataset(
    baseline, ds: "datasets.arrow_dataset.Dataset", task: str, model: str
) -> "datasets.arrow_dataset.Dataset":
    tokenizer_kwargs = {
        "sentiment-analysis": dict(),
        "language-modeling": dict(),
        "translation": dict(model_max_length=512),
    }

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model, **tokenizer_kwargs
    )

    if task == "sentiment-analysis":

        @export_def(baseline)
        def tokenizer_func(examples: Any, _, tok=None):
            texts = examples["text"]
            return tok(texts, max_length=512, padding="max_length")

        ds = ds.map(tokenizer_func, batched=True, with_indices=True, 
                    fn_kwargs={"tok": tokenizer}, num_proc=16)
        ds = ds.rename_column("label", "labels")
        ds.set_format(
            "torch",
            columns=["input_ids", "attention_mask", "token_type_ids", "labels"],
        )

    elif task == "language-modeling":

        # #######################
        # #### For wikitext  ####
        # #######################

        # @export_def(baseline)
        # def tokenizer_function(examples: Any, tok=None):
        #     return tok(examples["text"])

        # ds = ds.map(tokenizer_function, batched=True,
        #             fn_kwargs={"tok": tokenizer},
        #             remove_columns=["text"])

        # @export_def(baseline)
        # def convert_to_features(examples: Any):
        #     block_size = 1024

        #     concatenated_examples = {
        #         k: sum(examples[k], []) for k in examples.keys()
        #     }
        #     total_length = len(concatenated_examples[list(examples.keys())[0]])

        #     total_length = (total_length // block_size) * block_size
        #     result = {
        #         k: [
        #             t[i : i + block_size]
        #             for i in range(0, total_length, block_size)
        #         ]
        #         for k, t in concatenated_examples.items()
        #     }
        #     result["labels"] = result["input_ids"].copy()
        #     return result
        
        # ds = ds.map(convert_to_features, batched=True)
        # ds.set_format("torch")

        #######################
        #### For wikitext  ####
        #######################

        # These transformations are already done
        tokenizer.pad_token = tokenizer.eos_token

        @export_def(baseline)
        def truncate(examples: Any):
            block_size = 1024

            concatenated_examples = {
                "text": " ".join(examples["text"])
            }
            total_length = len(concatenated_examples["text"])
            total_length = (total_length // block_size) * block_size
            result = {
                "text": [
                    concatenated_examples["text"][i: i + block_size]
                    for i in range(0, total_length, block_size)
                ]
            }
            return result

        ds = ds.map(truncate, batched=True, remove_columns=["id", "url", "title"], 
                    num_proc=16)

        @export_def(baseline)
        def tokenize(examples, tok=None):
            return tok(examples["text"], max_length=1024, padding="max_length",
                       truncation=True)
        
        ds = ds.map(tokenize, batched=True, fn_kwargs={"tok": tokenizer},
                    num_proc=16)
        ds = ds.remove_columns(["text"])
        
        @export_def(baseline)
        def copy_column(examples):
            examples["labels"] = examples["input_ids"]
            return examples

        ds = ds.map(copy_column, batched=True, num_proc=16)

        ds.set_format("torch")

    elif task == "translation":

        @export_def(baseline)
        def convert_to_features(examples: Any, tok=None):
            source_language = "en"
            target_language = "ro"
            max_source_length = 128
            max_target_length = 128
            padding = "max_length"

            inputs = [ex[source_language] for ex in examples["translation"]]
            targets = [ex[target_language] for ex in examples["translation"]]
            model_inputs = tok(
                inputs,
                max_length=max_source_length,
                padding=padding,
                truncation=True,
            )

            with tok.as_target_tokenizer():
                labels = tok(
                    targets,
                    max_length=max_target_length,
                    padding=padding,
                    truncation=True,
                )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        ds = ds.map(convert_to_features, batched=True, fn_kwargs={"tok": tokenizer},
                    num_proc=16)
        ds.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )

    else:
        raise RuntimeError(f"Unknown task: {task}")

    return ds


def build_model(task: str, model: str):
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
    assert task in ("sentiment-analysis", "language-modeling", "translation")
    assert model in ("bert-base-cased", "bert-large-cased", "gpt2", "t5-base")

    import_libs(baseline)

    dataset = load_dataset(baseline, task)
    dataset = transform_dataset(baseline, dataset, task, model)

    batch_sizes = {
        "bert-base-cased": 128,
        "bert-large-cased": 32,
        "gpt2": 16,
        "t5-base": 128,
    }

    dataloader = DataLoader(dataset, batch_size=batch_sizes[model])

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
