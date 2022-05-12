# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=too-many-locals

import numpy as np
import pandas as pd
import torch
import typer
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer)

from datasets import load_dataset
from lang_classifier.constants.constants import LANGUAGES

app = typer.Typer()


@app.command()
def predict(
    model_path: str = typer.Option(
        None, help="Path to the model config weights",
    ),
    do_multilabel: bool = typer.Option(
        False, help="Do multilabel instead of multiclass classification"
    ),
    data_path: str = typer.Option(
        "datasets", help="Path to csv file with samples for model predictions"
    ),
    save_to: str = typer.Option(
        "predictions.csv", help="Path to csv file where to save model predictions"
    ),
):
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
        return tokens

    tokenized_datasets = dataset.map(tokenize, batched=True)
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
        outputs = np.argmax(preds, axis=1)
        outputs = outputs.tolist()
        outputs = [model.config.id2label[ind] for ind in outputs]
        preds_df = pd.DataFrame(outputs, columns=['label'])

    preds_df.to_csv(save_to, sep='\t', index=False)
    typer.secho(f"Successfully saved file with predictions to {save_to}", fg="green")


if __name__ == "__main__":
    app()
