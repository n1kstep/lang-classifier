# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring

import numpy as np
import torch
from datasets import load_metric
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score)
from transformers import EvalPrediction
from lang_classifier.constants.constants import LANGUAGES


metric_seq = load_metric("seqeval")


def metrics_multilabel(logits, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))

    preds = np.zeros(probs.shape)
    preds[np.where(probs >= threshold)] = 1

    f1_micro_average = f1_score(y_true=labels, y_pred=preds, average="micro")
    roc_auc = roc_auc_score(labels, preds, average="micro")
    accuracy = accuracy_score(labels, preds)

    report = classification_report(
        preds, labels, target_names=LANGUAGES, output_dict=True
    )
    metrics_per_class = {
        f"{lang}_{metric}": score
        for lang, lang_report in report.items()
        for metric, score in lang_report.items()
    }
    metrics_common = {
        "f1": f1_micro_average,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
    }
    return {**metrics_common, **metrics_per_class}


def compute_metrics_multiclass(output: EvalPrediction):
    logits, labels = output
    preds = np.argmax(logits, axis=-1)

    report = classification_report(
        preds, labels, target_names=LANGUAGES, output_dict=True
    )
    acc = report.pop("accuracy", None)

    scores = {
        f"{lang}_{metric}": score
        for lang, lang_report in report.items()
        for metric, score in lang_report.items()
    }
    return {"accuracy": acc, **scores}


def compute_metrics_multilabel(output: EvalPrediction):
    logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    result = metrics_multilabel(logits=logits, labels=output.label_ids)
    return result


def compute_metrics_token(output: EvalPrediction):
    logits, labels = output
    preds = np.argmax(logits, axis=2)

    preds_per_token = [
        [LANGUAGES[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    labels_per_token = [
        [LANGUAGES[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    results = metric_seq.compute(predictions=preds_per_token, references=labels_per_token)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
