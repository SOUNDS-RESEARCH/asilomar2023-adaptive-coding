import numpy as np

import matplotlib.pyplot as plt

# import seaborn as sns
from seaborn import objects as so
import polars as pl

from pythonutils import utils

plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": False,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "font.size": 7,
    }
)

# Read the data from the resulting csv files (we are using polars lazy api)
q = pl.scan_csv("data/results_*.csv").with_columns(pl.col("npm").log10() * 20)

# Plot and save relevant figures
textwidth = 245
linewidth = 1.2
plot_every = 50  # downsample data a bit so plots are bettwe readable
fig, ax = plt.subplots(figsize=utils.set_size(textwidth, 1.0, (1, 1), 0.4))
plt.plot(
    q.filter(pl.col("alg") == "base")
    .groupby("series", maintain_order=True)
    .agg(pl.col("npm").median())
    .select(pl.col("npm"))
    .take_every(plot_every)
    .collect(),
    "-o",
    label=f"optimal",
    markersize=4,
    markevery=(1, 10),
    alpha=1,
    linewidth=linewidth,
)
plt.plot(
    q.filter(pl.col("alg") == "adaptive")
    .groupby("series", maintain_order=True)
    .agg(pl.col("npm").median())
    .select(pl.col("npm"))
    .take_every(plot_every)
    .collect(),
    "--x",
    label=f"adaptive",
    markersize=4,
    markevery=(5, 10),
    alpha=1,
    linewidth=linewidth,
)
plt.legend()
plt.grid()
plt.xlabel("Time [frames]")
plt.ylabel("NPM [dB]")
plt.tight_layout(pad=0.5)
plt.show()

utils.savefig(fig, "npm-over-time", format="pdf")
