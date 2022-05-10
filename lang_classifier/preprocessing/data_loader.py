from random import sample
from datasets import load_dataset
import pandas as pd
import typer
import yaml
from lang_classifier.utils import smart_truncate


app = typer.Typer()


@app.command()
def load_data(
	config_path: str = typer.Option("../configs/loader_config.yaml", 
		help="Path to the config with data sources and list of languages"
	),
	save_to: str = typer.Option("../datasets", 
		help="Path to directory where to save loaded datasets"
	),
):
	with open(config_path, "r") as yamlfile:
		config = yaml.load(yamlfile, Loader=yaml.FullLoader)

	RESOURCES = config['resources']
	SAMPLES_PER_LANG = config['samples_per_lang']

	for key, value in RESOURCES['tatoeba'].items():
		dataset = load_dataset("tatoeba", lang1=value, lang2=key)

		sampled_rows = sample(dataset['train']['translation'], SAMPLES_PER_LANG)
		sampled_df = pd.DataFrame([{'text': samp[key], 'label': key} for samp in sampled_rows])
		sampled_df.to_csv(f"{save_to}/dataset_{key}.csv", sep='\t', index=False)

		typer.secho(f"Successfully saved file dataset_{key}.csv", fg='green')

	for key in RESOURCES['oscar']:
		dataset = load_dataset("oscar", f"unshuffled_deduplicated_{key}")

		sampled_rows = sample(dataset['train']['text'], SAMPLES_PER_LANG)
		sampled_df = pd.DataFrame([{'text': smart_truncate(samp), 'label': key} for samp in sampled_rows])
		sampled_df.to_csv(f"{save_to}/dataset_{key}.csv", sep='\t', index=False)

		typer.secho(f"Successfully saved file dataset_{key}.csv", fg='green')

	for key, value in RESOURCES['open_subtitles'].items():
		dataset = load_dataset("open_subtitles", lang1=key, lang2=value)

		sampled_rows = sample(dataset['train']['translation'], SAMPLES_PER_LANG)
		sampled_df = pd.DataFrame([{'text': samp[key], 'label': key} for samp in sampled_rows])
		sampled_df.to_csv(f"{save_to}/dataset_{key}.csv", sep='\t', index=False)

		typer.secho(f"Successfully saved file dataset_{key}.csv", fg='green')

if __name__ == "__main__":
	app()