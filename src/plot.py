import numpy as np
import matplotlib.pyplot as plt
import polars as pl

from pythonutils import utils


def plot():
    plt.rcParams.update(
        {
            "font.family": "serif",  # use serif/main font for text elements
            "text.usetex": False,  # don't use inline math for ticks (not on docker image)
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "font.size": 8,
        }
    )

    # Read the data from the resulting csv files (we are using polars lazy api)
    plot_every = 50  # downsample data a bit for better readability
    q = pl.scan_csv(
        "data/dynamic_tracking_*.csv",
        dtypes={
            "run_rn": pl.Int64,
            "alg": pl.Utf8,
            "series": pl.Int64,
            "npm": pl.Float64,
            "normalizer": pl.Float64,
            "symbol_mean": pl.Float64,
            "bits": pl.Int64,
        },
    ).with_columns((pl.col("npm").log10() * 20))
    q_base = (
        q.filter(pl.col("alg") == "base", pl.col("SNR") == 50)
        .groupby("series", maintain_order=True)
        .agg(npm_db_m=pl.col("npm").median())
        .take_every(plot_every)
    )
    q_symbol_mean = (
        q.filter(
            pl.col("alg") == "symbol_mean",
            pl.col("SNR") == 50,
            pl.col("codebook_entries") == 5,
        )
        .groupby("series", maintain_order=True)
        .agg(npm_db_m=pl.col("npm").median())
        .take_every(plot_every)
    )
    q_quant_var = (
        q.filter(
            pl.col("alg") == "quant_var",
            pl.col("SNR") == 50,
            pl.col("codebook_entries") == 5,
        )
        .groupby("series", maintain_order=True)
        .agg(npm_db_m=pl.col("npm").median())
        .take_every(plot_every)
    )

    # Plot and save relevant figures
    textwidth = 245
    linewidth = 1.2
    fig, ax = plt.subplots(figsize=utils.set_size(textwidth, 1.0, (1, 1), 0.4))
    data = q_base.collect()
    (line,) = plt.plot(
        data["series"],
        data["npm_db_m"],
        "-o",
        label=f"full",
        markersize=4,
        markevery=(1, 10),
        alpha=1,
        linewidth=linewidth,
    )
    # plt.fill_between(
    #     data["series"],
    #     data["npm_m"] - data["npm_std"] / np.sqrt(data["count"]),
    #     data["npm_m"] + data["npm_std"] / np.sqrt(data["count"]),
    #     color=line.get_color(),
    #     alpha=0.25,
    # )
    data = q_symbol_mean.collect()
    (line,) = plt.plot(
        data["series"],
        data["npm_db_m"],
        "--x",
        label=f"SM",
        markersize=4,
        markevery=(5, 10),
        alpha=1,
        linewidth=linewidth,
    )
    data = q_quant_var.collect()
    (line,) = plt.plot(
        data["series"],
        data["npm_db_m"],
        "-.+",
        label=f"QV",
        markersize=4,
        markevery=(5, 10),
        alpha=1,
        linewidth=linewidth,
    )
    # plt.fill_between(
    #     data["series"],
    #     data["npm_m"] - data["npm_std"] / np.sqrt(data["count"]),
    #     data["npm_m"] + data["npm_std"] / np.sqrt(data["count"]),
    #     color=line.get_color(),
    #     alpha=0.25,
    # )
    # plt.legend(title="Algorithms", fontsize=5, title_fontsize=6)
    plt.legend(fontsize=6)
    plt.grid()
    plt.xlabel("Time [frames]")
    plt.ylabel("NPM [dB]")
    plt.tight_layout(pad=0.5)
    plt.show()

    utils.savefig(fig, "npm-over-time", format="pdf")

    # Plot and save relevant figures
    # fig, ax = plt.subplots(figsize=utils.set_size(textwidth, 1.0, (1, 1), 0.4))
    # data = (
    #     q.filter(pl.col("alg") == "adaptive")
    #     .groupby("series", maintain_order=True)
    #     .agg(pl.col("delta_consensus").median())
    #     .take_every(plot_every)
    #     .collect()
    # )
    # plt.plot(
    #     data["series"],
    #     data["delta_consensus"],
    #     "k-",
    #     label=f"median",
    #     markersize=4,
    #     markevery=(5, 10),
    #     alpha=1,
    #     linewidth=linewidth,
    # )
    # data = (
    #     q.filter(pl.col("alg") == "adaptive")
    #     .groupby("series", maintain_order=True)
    #     .agg(pl.col("delta_consensus").mean())
    #     .take_every(plot_every)
    #     .collect()
    # )
    # plt.plot(
    #     data["series"],
    #     data["delta_consensus"],
    #     "k--",
    #     label=f"mean",
    #     markersize=4,
    #     markevery=(5, 10),
    #     alpha=0.5,
    #     linewidth=linewidth,
    # )
    # # plt.legend(title="Algorithms", fontsize=5, title_fontsize=6)
    # plt.yscale("log")
    # plt.legend(fontsize=6)
    # plt.grid()
    # plt.xlabel("Time [frames]")
    # plt.ylabel(r"$\Delta$")
    # plt.tight_layout(pad=0.5)
    # plt.show()

    # utils.savefig(fig, "delta-over-time", format="pdf")

    # q_adaptive = (
    #     q.filter(pl.col("alg") == "adaptive")
    #     .groupby("series", maintain_order=True)
    #     .agg(pl.col("res").median())
    #     .take_every(plot_every)
    # )

    # # Plot and save relevant figures
    # fig, ax = plt.subplots(figsize=utils.set_size(textwidth, 1.0, (1, 1), 0.4))
    # data = q_adaptive.collect()
    # plt.plot(
    #     data["series"],
    #     data["res"],
    #     "k-",
    #     label=f"adaptive",
    #     markersize=4,
    #     markevery=(5, 10),
    #     alpha=1,
    #     linewidth=linewidth,
    # )
    # # plt.legend(title="Algorithms", fontsize=5, title_fontsize=6)
    # plt.yscale("log")
    # # plt.ylim(0, 0.001)
    # # plt.legend(fontsize=6)
    # plt.grid()
    # plt.xlabel("Time [frames]")
    # plt.ylabel("Residual")
    # plt.tight_layout(pad=0.5)
    # plt.show()

    # utils.savefig(fig, "residual-over-time", format="pdf")
