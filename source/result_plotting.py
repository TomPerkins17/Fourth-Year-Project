import matplotlib.pyplot as plt
import numpy as np
# SingleNoteTimbreCNN, SingleNoteTimbreCNNSmall, MelodyTimbreCNN, MelodyTimbreCNNSmall
balanced_acc_mean = [0.714, 0.744, 0.827, 0.814]
balanced_acc_std = [0.039, 0.047, 0.094, 0.078]

min_class_acc_mean = [0.627, 0.692, 0.744, 0.742]

fig = plt.figure(figsize=(7,6))
plt.errorbar(x=[0,1,2,3], y=balanced_acc_mean, yerr=balanced_acc_std, fmt="o", capsize=10, label="Balanced acc.")
plt.plot(min_class_acc_mean, "rx", label="Min.-per-class acc.")
plt.xticks([0,1,2,3], ["SingleNoteTimbreCNN", "SingleNoteTimbreCNNSmall", "MelodyTimbreCNN", "MelodyTimbreCNNSmall"],
           rotation=45, ha="right")
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.3)
plt.legend(loc="upper left")

plt.ylabel("Accuracy scores")
plt.title("Cross-validation score statistics for each architecture, \napplied to single-note and melody-based classification")

plt.show()
fig.savefig("../Figures/CV_scorestats_results.svg")