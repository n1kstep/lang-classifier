# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=too-many-locals

import numpy as np
import typer
import yaml
import datasets
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from datasets import ClassLabel, load_dataset
from lang_classifier.constants.constants import LANGUAGES, SEED
from lang_classifier.metrics.metrics import (compute_metrics_multiclass,
                                             compute_metrics_multilabel)

datasets.disable_caching()

app = typer.Typer()


@app.command()
def train(
    config_path: str = typer.Option(
        "configs/loader_config.yaml",
        help="Path to the config with training arguments",
    ),
    do_multilabel: bool = typer.Option(
        False, help="Do multilabel instead of multiclass classification"
    ),
    data_path: str = typer.Option(
        "datasets", help="Path to directory where train.csv and val.csv datasets located"
    ),
    save_to: str = typer.Option(
        "lang_model", help="Path to directory where to save model configuration and weights"
    ),
):
    with open(config_path, "r", encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    class_labels = ClassLabel(num_classes=len(LANGUAGES), names=LANGUAGES)
    dataset = load_dataset(
        "csv",
        data_files={"train": data_path + "/train.csv", "test": data_path + "/val.csv"},
        delimiter="\t",
    )
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=len(LANGUAGES),
        problem_type="multi_label_classification"
        if do_multilabel
        else "single_label_classification",
    )
    model.config.id2label = dict(enumerate(LANGUAGES))
    model.config.label2id = {lang: ind for ind, lang in enumerate(LANGUAGES)}

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=128
        )
        tokens["label"] = class_labels.str2int(batch["label"])
        return tokens

    def tokenize_multilabel(batch):
        encoding = tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=128
        )
        labels_batch = {k: batch[k] for k in batch.keys() if k in LANGUAGES}
        labels_matrix = np.zeros((len(batch["text"]), len(LANGUAGES)))
        for idx, label in enumerate(LANGUAGES):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()
        return encoding

    tokenized_datasets = dataset.map(
        tokenize_multilabel if do_multilabel else tokenize, batched=True
    )
    train_dataset = tokenized_datasets["train"].shuffle(seed=SEED)
    eval_dataset = tokenized_datasets["test"].shuffle(seed=SEED)

    training_args = TrainingArguments(
        output_dir=save_to+config["model_name"],
        overwrite_output_dir=True,
        logging_strategy=config["logging_strategy"],
        evaluation_strategy=config["evaluation_strategy"],
        save_strategy=config["save_strategy"],
        save_total_limit=config["save_total_limit"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_multilabel
        if do_multilabel
        else compute_metrics_multiclass,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    app()
