from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


def evaluate_scores(label, pred):
    # print("Test set labels are", np.round(100 * np.sum(label) / len(label)), "% Upright pianos")
    # print("Predictions are", np.round(100 * np.sum(pred) / len(pred)), "% Upright pianos")
    conf = metrics.confusion_matrix(label, pred)
    acc = metrics.accuracy_score(label, pred)
    if conf.shape == (2, 2) and len(np.unique(label)) == 2:    # These metrics are only valid if both classes are present in the gt labels
        f1 = metrics.f1_score(label, pred)
        bal_acc = metrics.balanced_accuracy_score(label, pred)
        conf_plot = metrics.ConfusionMatrixDisplay(conf, display_labels=["Grand", "Upright"])
    else:
        f1 = None
        bal_acc = None
        conf_plot = None
    return {"Confusion": conf, "Accuracy": acc, "F1": f1, "balanced_acc": bal_acc, "Confusion_plot": conf_plot}


def display_scores(scores, name=""):
    print("Confusion matrix:\n", scores["Confusion"])
    print("Accuracy:", np.round(scores["Accuracy"], 2))
    print("Balanced accuracy:", np.round(scores["balanced_acc"], 2))
    print("F1 score:", np.round(scores["F1"], 2))
    conf_plot = scores["Confusion_plot"].plot(cmap=plt.cm.Blues, colorbar=False)
    title_text = ": "+name
    conf_plot.ax_.set_title("Confusion matrix"+title_text)
    plt.show()