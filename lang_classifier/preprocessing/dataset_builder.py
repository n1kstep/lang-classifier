# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring

import random

import pandas as pd
import typer
from sklearn.model_selection import train_test_split

from lang_classifier.constants.constants import LANGUAGES
from lang_classifier.utils.utils import smart_truncate

app = typer.Typer()


def build_multilabel_dataset(
    dataset: pd.DataFrame, langs: list, min_truncate: int = 32, max_seq_len: int = 512
) -> pd.DataFrame:
    texts = []
    labels = {lang: [] for lang in langs}
    for _, row in dataset.iterrows():
        max_truncate = int(0.7 * len(row["text"]))
        ml_text = smart_truncate(
            row["text"],
            max_truncate
            if max_truncate < min_truncate
            else random.randint(0, max_truncate),
        )
        for vals in labels.values():
            vals.append(0)
        labels[row["label"]][-1] = 1
        for _ in range(random.randint(0, 5)):
            another_row = dataset.sample()
            max_truncate = int(0.7 * len(another_row["text"].item()))
            another_text = smart_truncate(
                another_row["text"].item(),
                max_truncate
                if max_truncate < min_truncate
                else random.randint(0, max_truncate),
            )
            if len(ml_text + (" " + another_text)) <= max_seq_len:
                ml_text += " " + another_text
                labels[another_row["label"].item()][-1] = 1

        texts.append(ml_text)

    res_dataset = pd.DataFrame({"text": texts, **labels})
    return res_dataset


@app.command()
def build_dataset(
    samples_per_lang: int = typer.Option(
        10000, help="Number of samples for each language in dataset"
    ),
    do_multilabel: bool = typer.Option(
        False, help="Form multilabel dataset instead of multiclass dataset"
    ),
    save_to: str = typer.Option(
        "../datasets", help="Path to directory where to save loaded datasets"
    ),
):
    dfs = []
    for lang in LANGUAGES:
        cur_df = pd.read_csv(f"dataset/dataset_{lang}.csv", sep="\t").sample(
            samples_per_lang
        )
        dfs.append(cur_df)

    merged_df = pd.concat(dfs).reset_index(drop=True)
    merged_df = merged_df.sample(frac=1).reset_index(drop=True)
    merged_df = merged_df.dropna()
    merged_df = merged_df[merged_df["text"].apply(lambda x: len(x) > 0)]
    typer.secho("Merged multilingual dataset", fg="green")

    train, val = train_test_split(merged_df, test_size=0.1, stratify=merged_df["label"])
    if do_multilabel:
        train = build_multilabel_dataset(train, langs=LANGUAGES)
        val = build_multilabel_dataset(val, langs=LANGUAGES)
        typer.secho("Built train and val multilabel datasets", fg="green")

    train.to_csv(f"{save_to}/train.csv", sep="\t", index=False)
    typer.secho(f"Successfully saved train dataset to {save_to}/train.csv", fg="green")
    val.to_csv(f"{save_to}/val.csv", sep="\t", index=False)
    typer.secho(f"Successfully saved val dataset to {save_to}/val.csv", fg="green")
