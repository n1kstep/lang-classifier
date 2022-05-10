# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score)
from transformers import EvalPrediction

from lang_classifier.constants.constants import LANGUAGES


def compute_metrics_multilabel(output: EvalPrediction):
    preds = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    result = metrics_multilabel(predictions=preds, labels=output.label_ids)
    return result


def metrics_multilabel(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels

    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_pred, y_true, target_names=LANGUAGES, output_dict=True
    )
    scores = {
        f"{lang}_{metric}": score
        for lang, lang_report in report.items()
        for metric, score in lang_report.items()
    }
    metrics = {
        "f1": f1_micro_average,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
    }
    return {**metrics, **scores}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    report = classification_report(
        predictions, labels, target_names=LANGUAGES, output_dict=True
    )
    acc = report["accuracy"]
    report.pop("accuracy", None)

    scores = {
        f"{lang}_{metric}": score
        for lang, lang_report in report.items()
        for metric, score in lang_report.items()
    }
    return {"accuracy": acc, **scores}
