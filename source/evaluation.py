from sklearn import metrics
import numpy as np


def evaluate(label, pred):
    print("GT Test set labels are", np.round(100 * np.sum(label) / len(label)), "% Upright pianos")
    print("Predictions are", np.round(100 * np.sum(pred) / len(pred)), "% Upright pianos")

    conf = metrics.confusion_matrix(label, pred)
    acc = metrics.accuracy_score(label, pred)
    f1 = metrics.f1_score(label, pred)
    print("Confusion matrix:\n", conf)
    print("Accuracy:", np.round(acc, 2))
    print("F1 score:", np.round(f1, 2))
