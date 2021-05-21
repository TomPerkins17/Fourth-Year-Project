from sklearn import metrics
import numpy as np


def evaluate_scores(label, pred):
    print("Test set labels are", np.round(100 * np.sum(label) / len(label)), "% Upright pianos")
    print("Predictions are", np.round(100 * np.sum(pred) / len(pred)), "% Upright pianos")

    conf = metrics.confusion_matrix(label, pred)
    acc = metrics.accuracy_score(label, pred)
    f1 = metrics.f1_score(label, pred)

    return {"Confusion": conf, "Accuracy": acc, "F1": f1}
