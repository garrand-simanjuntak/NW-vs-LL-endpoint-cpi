"""
fig_style.py
Label panel (a), (b), (c) di bawah tiap sumbu, dalam koordinat figure.
"""
import numpy as np


def panel_labels_below(fig, axes, labels, dy=0.09, fontsize=10, weight="normal"):
    axes = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    for ax, lab in zip(axes, labels):
        box = ax.get_position()
        fig.text(box.x0 + box.width / 2.0, box.y0 - dy, lab,
                 ha="center", va="top",
                 fontsize=fontsize, fontweight=weight)
