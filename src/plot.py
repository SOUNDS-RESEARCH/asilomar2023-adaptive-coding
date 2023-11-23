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

    Ls = [16]
    L = 16
    add_zeross = [True, False]
    SNRs = [10, 30, 50, 70]
    codebook_entriess = [3, 5, 7, 11, 21]
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
    textwidth = 245
    rel_height = 0.4
    linewidth = 1.2
    for add_zeros in add_zeross:
        for SNR in SNRs:
            for codebook_entries in codebook_entriess:
                q_base = (
                    q.filter(
                        pl.col("alg") == "base",
                        pl.col("SNR") == SNR,
                        pl.col("add_zeros") == add_zeros,
                    )
                    .groupby("series", maintain_order=True)
                    .agg(npm_db_m=pl.col("npm").median())
                    .take_every(plot_every)
                )
                q_symbol_mean = (
                    q.filter(
                        pl.col("alg") == "symbol_mean",
                        pl.col("SNR") == SNR,
                        pl.col("codebook_entries") == codebook_entries,
                        pl.col("add_zeros") == add_zeros,
                    )
                    .groupby("series", maintain_order=True)
                    .agg(npm_db_m=pl.col("npm").median())
                    .take_every(plot_every)
                )
                q_quant_var = (
                    q.filter(
                        pl.col("alg") == "quant_var",
                        pl.col("SNR") == SNR,
                        pl.col("codebook_entries") == codebook_entries,
                        pl.col("add_zeros") == add_zeros,
                    )
                    .groupby("series", maintain_order=True)
                    .agg(npm_db_m=pl.col("npm").median())
                    .take_every(plot_every)
                )

                # Plot and save relevant figures
                fig, ax = plt.subplots(
                    figsize=utils.set_size(textwidth, 1.0, (1, 1), rel_height)
                )
                data = q_base.collect()
                (line,) = plt.plot(
                    data["series"],
                    data["npm_db_m"],
                    "-o",
                    label=f"full",
                    markersize=4,
                    markevery=(1, 20),
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
                    markevery=(5, 20),
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
                    markevery=(5, 20),
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

                utils.savefig(
                    fig,
                    f"npm-over-time-{add_zeros}-{SNR}-{codebook_entries}",
                    format="pdf",
                )
                plt.close()
    for add_zeros in add_zeross:
        for SNR in SNRs:
            for codebook_entries in codebook_entriess:
                q_base = (
                    q.filter(
                        pl.col("alg") == "base",
                        pl.col("SNR") == SNR,
                        pl.col("add_zeros") == add_zeros,
                    )
                    .groupby("series", maintain_order=True)
                    .agg(bits_m=pl.col("bits").median())
                    .take_every(plot_every)
                )
                q_symbol_mean = (
                    q.filter(
                        pl.col("alg") == "symbol_mean",
                        pl.col("SNR") == SNR,
                        pl.col("codebook_entries") == codebook_entries,
                        pl.col("add_zeros") == add_zeros,
                    )
                    .groupby("series", maintain_order=True)
                    .agg(bits_m=pl.col("bits").median())
                    .take_every(plot_every)
                )
                q_quant_var = (
                    q.filter(
                        pl.col("alg") == "quant_var",
                        pl.col("SNR") == SNR,
                        pl.col("codebook_entries") == codebook_entries,
                        pl.col("add_zeros") == add_zeros,
                    )
                    .groupby("series", maintain_order=True)
                    .agg(bits_m=pl.col("bits").median())
                    .take_every(plot_every)
                )

                # Plot and save relevant figures
                textwidth = 245
                linewidth = 1.2
                fig, ax = plt.subplots(
                    figsize=utils.set_size(textwidth, 1.0, (1, 1), rel_height)
                )
                # data = q_base.collect()
                # (line,) = plt.plot(
                #     data["series"],
                #     data["bits_m"] * L,
                #     "-o",
                #     label=f"full",
                #     markersize=4,
                #     markevery=(1, 20),
                #     alpha=1,
                #     linewidth=linewidth,
                # )
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
                    data["bits_m"],
                    "-x",
                    label=f"SM",
                    markersize=4,
                    markevery=(5, 20),
                    alpha=1,
                    linewidth=linewidth,
                )
                data = q_quant_var.collect()
                (line,) = plt.plot(
                    data["series"],
                    data["bits_m"],
                    "-+",
                    label=f"QV",
                    markersize=4,
                    markevery=(5, 20),
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
                plt.ylabel("Rate [bits]")
                plt.ylim(0, 500)
                plt.tight_layout(pad=0.5)
                plt.show()

                utils.savefig(
                    fig,
                    f"rate-over-time-{add_zeros}-{SNR}-{codebook_entries}",
                    format="pdf",
                )
                plt.close()

    q_bits_sum = q.groupby(["alg", "SNR", "codebook_entries"]).agg(
        bits_s=pl.col("bits").mean()
    )
    data = q_bits_sum.collect()
    max_bitrate = data["bits_s"].max()

    X_axis = np.arange(len(codebook_entriess))

    fig, ax = plt.subplots(figsize=utils.set_size(textwidth, 1.0, (1, 1), rel_height))
    # plt.plot(X_axis, np.ones_like(X_axis), "k--")

    plt.bar(
        X_axis - 0.2,
        data.filter(pl.col("alg") == "symbol_mean")
        .groupby("codebook_entries")
        .agg(bits_s=pl.col("bits_s").mean())
        .sort("codebook_entries")["bits_s"]
        / max_bitrate
        / L,
        0.4,
        label="SM",
    )
    plt.bar(
        X_axis + 0.2,
        data.filter(pl.col("alg") == "quant_var")
        .groupby("codebook_entries")
        .agg(bits_s=pl.col("bits_s").mean())
        .sort("codebook_entries")["bits_s"]
        / max_bitrate
        / L,
        0.4,
        label="QV",
    )
    plt.xticks(X_axis, codebook_entriess)
    plt.legend(fontsize=6)
    plt.grid()
    plt.xlabel("Number of symbols")
    plt.ylabel("R [relative]")
    plt.tight_layout(pad=0.5)
    plt.show()

    utils.savefig(fig, "bits-over-codebook_entries", format="pdf")

    frames = [5000, 10000, 15000]
    for SNR in SNRs:
        q_npm_conv = (
            q.filter(pl.col("SNR") == SNR, pl.col("series").is_in(frames))
            .groupby(["alg", "codebook_entries"])
            .agg(npm_conv=pl.col("npm").median())
        )
        data = q_npm_conv.collect()
        fig, ax = plt.subplots(figsize=utils.set_size(textwidth, 1.0, (1, 1), rel_height))

        plt.plot(
            np.arange(-2, 8),
            np.ones_like(np.arange(-2, 8))
            * data.filter(pl.col("alg") == "base")["npm_conv"].mean(),
            "k-.",
            label="full",
        )
        plt.bar(
            X_axis - 0.2,
            data.filter(pl.col("alg") == "symbol_mean")
            .groupby("codebook_entries")
            .agg(npm_conv=pl.col("npm_conv").mean())
            .sort("codebook_entries")["npm_conv"],
            0.4,
            label="SM",
        )
        plt.bar(
            X_axis + 0.2,
            data.filter(pl.col("alg") == "quant_var")
            .groupby("codebook_entries")
            .agg(npm_conv=pl.col("npm_conv").mean())
            .sort("codebook_entries")["npm_conv"],
            0.4,
            label="QV",
        )
        plt.xticks(X_axis, codebook_entriess)
        plt.legend(fontsize=6)
        plt.grid()
        plt.ylim(-90, 0)
        plt.xlim(-0.5, 4.5)
        plt.xlabel("Number of symbols")
        plt.ylabel("NPM [dB]")
        plt.tight_layout(pad=0.5)
        plt.show()

        utils.savefig(fig, f"NPM-over-codebook_entries-{SNR}", format="pdf")
