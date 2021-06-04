from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


def evaluate_scores(label, pred):
    # print("Test set labels are", np.round(100 * np.sum(label) / len(label)), "% Upright pianos")
    # print("Predictions are", np.round(100 * np.sum(pred) / len(pred)), "% Upright pianos")
    conf = metrics.confusion_matrix(label, pred)
    acc = metrics.accuracy_score(label, pred)
    if conf.shape == (2, 2):    # These metrics are only valid if both classes are present in the gt labels
        f1 = metrics.f1_score(label, pred)
        conf_plot = metrics.ConfusionMatrixDisplay(conf, display_labels=["Grand", "Upright"])
    else:
        f1 = None
        conf_plot = None
    return {"Confusion": conf, "Accuracy": acc, "F1": f1, "Confusion_plot": conf_plot}
