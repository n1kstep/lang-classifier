# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=too-many-locals

import numpy as np
import pandas as pd
import torch
import typer
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer)

from datasets import ClassLabel, load_dataset
from lang_classifier.constants.constants import LANGUAGES

app = typer.Typer()


@app.command()
def predict(
    model_path: str = typer.Option(
        None, help="Path to the model config weights",
    ),
    do_multilabel: str = typer.Option(
        False, help="Do multilabel instead of multiclass classification"
    ),
    data_path: str = typer.Option(
        None, help="Path to csv file with samples for model predictions"
    ),
    save_to: str = typer.Option(
        None, help="Path to csv file where to save model predictions"
    ),
):
    class_labels = ClassLabel(num_classes=len(LANGUAGES), names=LANGUAGES)
    dataset = load_dataset(
        "csv",
        data_files={"predict": data_path},
        delimiter="\t",
    )
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

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
    predict_dataset = tokenized_datasets["predict"]

    trainer = Trainer(
        model=model,
    )
    preds = trainer.predict(predict_dataset)
    preds = preds.predictions

    if do_multilabel:
        sigmoid = torch.nn.Sigmoid()

        probs = sigmoid(torch.Tensor(preds))
        probs[probs > 0.5] = 1
        probs[probs <= 0.5] = 0
        probs = probs.int().tolist()
        preds_df = pd.DataFrame(data=probs, columns=LANGUAGES)
    else:
        preds = np.argmax(preds)

    preds_df.to_csv(save_to, sep='\t', index=False)


if __name__ == "__main__":
    app()
