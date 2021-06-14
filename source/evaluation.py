from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def evaluate_scores(label, pred):
    # print("Test set labels are", np.round(100 * np.sum(label) / len(label)), "% Upright pianos")
    # print("Predictions are", np.round(100 * np.sum(pred) / len(pred)), "% Upright pianos")
    conf = metrics.confusion_matrix(label, pred)
    acc = metrics.accuracy_score(label, pred)
    # These metrics are only valid if both classes are present in the gt labels
    if conf.shape == (2, 2) and len(np.unique(label)) == 2:
        f1 = metrics.f1_score(label, pred)
        bal_acc = metrics.balanced_accuracy_score(label, pred)
        acc_grand = conf[0, 0]/(conf[0, 0] + conf[0, 1])
        acc_upright = conf[1, 1]/(conf[1, 0] + conf[1, 1])
        min_class_acc = min(acc_grand, acc_upright)
        conf_plot = metrics.ConfusionMatrixDisplay(conf, display_labels=["Grand", "Upright"])
    else:
        f1 = None
        bal_acc = None
        acc_grand = None
        acc_upright = None
        min_class_acc = None
        conf_plot = None
    return {"Confusion": conf, "Accuracy": acc, "F1": f1, "balanced_acc": bal_acc,
            "acc_grand": acc_grand, "acc_upright": acc_upright, "min_class_acc": min_class_acc,
            "Confusion_plot": conf_plot}


def display_scores(scores, name="", plot_conf=True):
    print("Confusion matrix:\n", scores["Confusion"])
    print("Accuracy:", np.round(scores["Accuracy"], 2))
    print("F1 score:", np.round(scores["F1"], 2))
    print("Accuracy per-class:")
    print("\t Grand:", np.round(scores["acc_grand"], 2))
    print("\t Upright:", np.round(scores["acc_upright"], 2))
    print("Balanced (macro-averaged) accuracy:", np.round(scores["balanced_acc"], 2))
    print("Per-class minimum accuracy:", np.round(scores["min_class_acc"], 2))
    if plot_conf:
        conf_plot = scores["Confusion_plot"].plot(cmap=plt.cm.Blues, colorbar=False)
        title_text = ": "+name
        conf_plot.ax_.set_title("Confusion matrix"+title_text)
        plt.show()