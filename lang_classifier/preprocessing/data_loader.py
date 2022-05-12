# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring

from random import sample

import pandas as pd
import typer
import yaml

from datasets import load_dataset
from datasets import set_caching_enabled
from lang_classifier.utils.utils import smart_truncate

set_caching_enabled(False)

app = typer.Typer()


@app.command()
def load_data(
    config_path: str = typer.Option(
        "../configs/loader_config.yaml",
        help="Path to the config with data sources and list of languages",
    ),
    save_to: str = typer.Option(
        "../datasets", help="Path to directory where to save loaded datasets"
    ),
):
    with open(config_path, "r", encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    resources = config["resources"]
    samples_per_lang = config["samples_per_lang"]

    for lang, translate_from in {**resources["tatoeba"], **resources["open_subtitles"]}.items():
        if lang in ('ru', 'uk'):
            typer.secho(f"Loading data for language - {lang}, data resource - Tatoeba")
            dataset = load_dataset("tatoeba", lang1=translate_from, lang2=lang)
        else:
            typer.secho(f"Loading data for language - {lang}, data resource - Open Subtitles")
            dataset = load_dataset("open_subtitles", lang1=translate_from, lang2=lang)

        sampled_rows = sample(dataset["train"]["translation"], samples_per_lang)
        sampled_df = pd.DataFrame(
            [{"text": samp[lang], "label": lang} for samp in sampled_rows]
        )
        sampled_df.to_csv(f"{save_to}/dataset_{lang}.csv", sep="\t", index=False)

        typer.secho(f"Successfully saved file dataset_{lang}.csv", fg="green")

    for lang in resources["oscar"]:
        typer.secho(f"Loading data for language - {lang}, data resource - Oscar")
        dataset = load_dataset("oscar", f"unshuffled_deduplicated_{lang}")

        sampled_rows = sample(dataset["train"]["text"], samples_per_lang)
        sampled_df = pd.DataFrame(
            [{"text": smart_truncate(samp), "label": lang} for samp in sampled_rows]
        )
        sampled_df.to_csv(f"{save_to}/dataset_{lang}.csv", sep="\t", index=False)

        typer.secho(f"Successfully saved file dataset_{lang}.csv", fg="green")


if __name__ == "__main__":
    app()
