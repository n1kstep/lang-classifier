import numpy as np
from transformers import EvalPrediction
import torch
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from lang_classifier.constants import LANGUAGES


def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_pred, y_true, target_names=LANGUAGES, output_dict=True)
    scores = {f"{lang}_{metric}": score for lang, lang_report in report.items() for metric, score in
              lang_report.items()}
    metrics = {
        'f1': f1_micro_average,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
    }
    return {**metrics, **scores}


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def compute_metrics_multilabel(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(predictions, labels, target_names=LANGUAGES, output_dict=True)
    acc = report['accuracy']
    report.pop('accuracy', None)
    scores = {f"{lang}_{metric}": score for lang, lang_report in report.items() for metric, score in
              lang_report.items()}
    return {'accuracy': acc, **scores}