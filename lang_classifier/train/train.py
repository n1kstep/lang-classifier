from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import ClassLabel
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
from transformers import Trainer
import typer
from lang_classifier.constants import LANGUAGES

from lang_classifier.metrics.metrics import compute_metrics_multilabel, compute_metrics

app = typer.Typer()


@app.command()
def train(
        do_multilabel: str = typer.Option(False,
                                          help="Do multilabel instead of multiclass classification"
                                          ),
        data_path: str = typer.Option(None,
                                      help="Path to directory where train.csv and val.csv datasets located"
                                      ),
        save_to: str = typer.Option(None,
                                    help="Path to directory where to save loaded datasets"
                                    ),
):
    class_labels = ClassLabel(num_classes=10, names=LANGUAGES)
    dataset = load_dataset('csv',
                           data_files={'train': data_path + "/train.csv", 'test': data_path + "/val.csv"},
                           delimiter='\t'
                           )
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=10,
        problem_type="multi_label_classification" if do_multilabel else "single_label_classification",
    )

    def tokenize(batch):
        tokens = tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)
        tokens['label'] = class_labels.str2int(batch['label'])
        return tokens

    def tokenize_multilabel(batch):
        encoding = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
        labels_batch = {k: batch[k] for k in batch.keys() if k in LANGUAGES}
        labels_matrix = np.zeros((len(batch["text"]), len(LANGUAGES)))
        for idx, label in enumerate(LANGUAGES):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()
        return encoding

    tokenized_datasets = dataset.map(tokenize_multilabel if do_multilabel else tokenize, batched=True)
    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

    batch_size = 16
    training_args = TrainingArguments(
        output_dir=save_to + "/lang-xlm-roberta-base",
        overwrite_output_dir=True,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        num_train_epochs=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_multilabel if do_multilabel else compute_metrics,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    app()
