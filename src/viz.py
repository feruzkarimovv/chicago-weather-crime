# tiny plotting helpers

import os
import matplotlib.pyplot as plt


def save_fig(name):
    os.makedirs("../outputs/figures", exist_ok=True)
    plt.savefig(f"../outputs/figures/{name}.png", bbox_inches="tight", dpi=120)
    plt.close()


def temp_bin(tmax_f):
    # matches the SQL bins in phase 3 + the data dictionary
    if tmax_f < 32:
        return "cold"
    elif tmax_f < 50:
        return "cool"
    elif tmax_f < 70:
        return "mild"
    elif tmax_f < 85:
        return "warm"
    else:
        return "hot"
