from pathlib import Path

import matplotlib.pyplot as plt

FIG_WIDTH = 5.5
GOLDEN_RATIO = 1.618
SUBPLOT_HEIGHT = (FIG_WIDTH / 4) / GOLDEN_RATIO

_STYLESHEET = Path(__file__).parent / "paper.mplstyle"


def apply():
    plt.style.use(_STYLESHEET)
