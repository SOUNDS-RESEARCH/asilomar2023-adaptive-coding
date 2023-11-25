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
    add_zeross = [False]
    SNRs = [10, 30, 50, 70]
    codebook_entriess = [3, 5, 7, 11, 21]
    q = (
        pl.scan_csv(
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
        )
        .with_columns((pl.col("npm").log10() * 20))
        .fill_nan(None)
    )
    textwidth = 245
    rel_height = 0.4
    linewidth = 1.2
    # ####################################################################################
    # for add_zeros in add_zeross:
    #     for SNR in SNRs:
    #         for codebook_entries in codebook_entriess:
    #             q_base = (
    #                 q.filter(
    #                     pl.col("alg") == "base",
    #                     pl.col("SNR") == SNR,
    #                     pl.col("add_zeros") == add_zeros,
    #                 )
    #                 .groupby("series", maintain_order=True)
    #                 .agg(npm_db_m=pl.col("npm").mean())
    #                 .take_every(plot_every)
    #             )
    #             q_symbol_mean = (
    #                 q.filter(
    #                     pl.col("alg") == "symbol_mean",
    #                     pl.col("SNR") == SNR,
    #                     pl.col("codebook_entries") == codebook_entries,
    #                     pl.col("add_zeros") == add_zeros,
    #                 )
    #                 .groupby("series", maintain_order=True)
    #                 .agg(npm_db_m=pl.col("npm").mean())
    #                 .take_every(plot_every)
    #             )
    #             q_quant_var = (
    #                 q.filter(
    #                     pl.col("alg") == "quant_var",
    #                     pl.col("SNR") == SNR,
    #                     pl.col("codebook_entries") == codebook_entries,
    #                     pl.col("add_zeros") == add_zeros,
    #                 )
    #                 .groupby("series", maintain_order=True)
    #                 .agg(npm_db_m=pl.col("npm").mean())
    #                 .take_every(plot_every)
    #             )

    #             # Plot and save relevant figures
    #             fig, ax = plt.subplots(
    #                 figsize=utils.set_size(textwidth, 1.0, (1, 1), rel_height)
    #             )
    #             data = q_base.collect()
    #             (line,) = plt.plot(
    #                 data["series"],
    #                 data["npm_db_m"],
    #                 "-o",
    #                 label=f"full",
    #                 markersize=4,
    #                 markevery=(1, 20),
    #                 alpha=1,
    #                 linewidth=linewidth,
    #             )
    #             data = q_symbol_mean.collect()
    #             (line,) = plt.plot(
    #                 data["series"],
    #                 data["npm_db_m"],
    #                 "--x",
    #                 label=f"SM",
    #                 markersize=4,
    #                 markevery=(5, 20),
    #                 alpha=1,
    #                 linewidth=linewidth,
    #             )
    #             data = q_quant_var.collect()
    #             (line,) = plt.plot(
    #                 data["series"],
    #                 data["npm_db_m"],
    #                 "-.+",
    #                 label=f"QV",
    #                 markersize=4,
    #                 markevery=(5, 20),
    #                 alpha=1,
    #                 linewidth=linewidth,
    #             )
    #             plt.legend(fontsize=6)
    #             plt.grid()
    #             plt.xlabel("Time [frames]")
    #             plt.ylabel("NPM [dB]")
    #             plt.tight_layout(pad=0.5)
    #             plt.show()

    #             utils.savefig(
    #                 fig,
    #                 f"npm-over-time-{add_zeros}-{SNR}-{codebook_entries}",
    #                 format="pdf",
    #             )
    #             plt.close()

    # ####################################################################################
    # for add_zeros in add_zeross:
    #     for SNR in SNRs:
    #         for codebook_entries in codebook_entriess:
    #             q_symbol_mean = (
    #                 q.filter(
    #                     pl.col("alg") == "symbol_mean",
    #                     pl.col("SNR") == SNR,
    #                     pl.col("codebook_entries") == codebook_entries,
    #                     pl.col("add_zeros") == add_zeros,
    #                 )
    #                 .groupby("series", maintain_order=True)
    #                 .agg(bits_m=pl.col("bits").mean())
    #                 .take_every(plot_every)
    #             )
    #             q_quant_var = (
    #                 q.filter(
    #                     pl.col("alg") == "quant_var",
    #                     pl.col("SNR") == SNR,
    #                     pl.col("codebook_entries") == codebook_entries,
    #                     pl.col("add_zeros") == add_zeros,
    #                 )
    #                 .groupby("series", maintain_order=True)
    #                 .agg(bits_m=pl.col("bits").mean())
    #                 .take_every(plot_every)
    #             )

    #             # Plot and save relevant figures
    #             textwidth = 245
    #             linewidth = 1.2
    #             fig, ax = plt.subplots(
    #                 figsize=utils.set_size(textwidth, 1.0, (1, 1), rel_height)
    #             )
    #             data = q_symbol_mean.collect()
    #             (line,) = plt.plot(
    #                 data["series"],
    #                 data["bits_m"],
    #                 "-x",
    #                 label=f"SM",
    #                 markersize=4,
    #                 markevery=(5, 20),
    #                 alpha=1,
    #                 linewidth=linewidth,
    #             )
    #             data = q_quant_var.collect()
    #             (line,) = plt.plot(
    #                 data["series"],
    #                 data["bits_m"],
    #                 "-+",
    #                 label=f"QV",
    #                 markersize=4,
    #                 markevery=(5, 20),
    #                 alpha=1,
    #                 linewidth=linewidth,
    #             )
    #             plt.legend(fontsize=6)
    #             plt.grid()
    #             plt.xlabel("Time [frames]")
    #             plt.ylabel("Rate [bits]")
    #             plt.ylim(0, 500)
    #             plt.tight_layout(pad=0.5)
    #             plt.show()

    #             utils.savefig(
    #                 fig,
    #                 f"rate-over-time-{add_zeros}-{SNR}-{codebook_entries}",
    #                 format="pdf",
    #             )
    #             plt.close()

    # ####################################################################################
    # for add_zeros in add_zeross:
    #     for SNR in SNRs:
    #         for codebook_entries in codebook_entriess:
    #             q_symbol_mean = (
    #                 q.filter(
    #                     pl.col("alg") == "symbol_mean",
    #                     pl.col("SNR") == SNR,
    #                     pl.col("codebook_entries") == codebook_entries,
    #                     pl.col("add_zeros") == add_zeros,
    #                 )
    #                 .groupby("series", maintain_order=True)
    #                 .agg(normalizer_m=pl.col("normalizer").mean())
    #                 .take_every(plot_every)
    #             )
    #             q_quant_var = (
    #                 q.filter(
    #                     pl.col("alg") == "quant_var",
    #                     pl.col("SNR") == SNR,
    #                     pl.col("codebook_entries") == codebook_entries,
    #                     pl.col("add_zeros") == add_zeros,
    #                 )
    #                 .groupby("series", maintain_order=True)
    #                 .agg(normalizer_m=pl.col("normalizer").mean())
    #                 .take_every(plot_every)
    #             )
    #             fig, ax = plt.subplots(
    #                 figsize=utils.set_size(textwidth, 1.0, (1, 1), rel_height)
    #             )
    #             data = q_symbol_mean.collect()
    #             (line,) = plt.plot(
    #                 data["series"],
    #                 data["normalizer_m"],
    #                 "-x",
    #                 label=f"SM",
    #                 markersize=4,
    #                 markevery=(5, 20),
    #                 alpha=1,
    #                 linewidth=linewidth,
    #             )
    #             data = q_quant_var.collect()
    #             (line,) = plt.plot(
    #                 data["series"],
    #                 data["normalizer_m"],
    #                 "-+",
    #                 label=f"QV",
    #                 markersize=4,
    #                 markevery=(5, 20),
    #                 alpha=1,
    #                 linewidth=linewidth,
    #             )
    #             plt.legend(fontsize=6)
    #             plt.grid()
    #             plt.xlabel("Time [frames]")
    #             plt.ylabel(r"\delta")
    #             plt.ylim(0, 20)
    #             plt.tight_layout(pad=0.5)
    #             plt.show()

    #             utils.savefig(
    #                 fig,
    #                 f"normalizer-over-time-{add_zeros}-{SNR}-{codebook_entries}",
    #                 format="pdf",
    #             )
    #             plt.close()

    # ####################################################################################
    fig, axs = plt.subplots(3, 1, figsize=utils.set_size(textwidth, 1.0, (1, 1), 1.0))
    for SNR in [10, 50]:
        for codebook_entries in [3, 7]:
            q_quant_var = (
                q.filter(
                    pl.col("alg") == "quant_var",
                    pl.col("SNR") == SNR,
                    pl.col("codebook_entries") == codebook_entries,
                    pl.col("add_zeros") == False,
                )
                .groupby("series", maintain_order=True)
                .agg(bits_m=pl.col("bits").median())
                .take_every(plot_every)
            )
            data = q_quant_var.collect()
            (line,) = axs[2].plot(
                data["series"],
                data["bits_m"],
                label=f"({SNR}dB, {codebook_entries})",
                markersize=4,
                markevery=(5, 20),
                alpha=1,
                linewidth=linewidth,
            )
    # plt.legend(fontsize=6)
    axs[2].grid()
    axs[2].set_xlabel(r"Time $k$")
    axs[2].tick_params(axis="both", labelsize=6)
    axs[2].set_ylabel(r"$R^{(k)}$ [bits]")
    axs[2].set_ylim(50, 200)
    axs[2].set_xlim(0, 15000)
    # plt.yscale("log")
    # plt.tight_layout(pad=0.8)
    # ax = plt.gca()
    # box = ax.get_position()
    # plt.show()
    # utils.savefig(
    #     fig,
    #     f"bits-over-time-comparison",
    #     format="pdf",
    # )
    # plt.close()

    # fig, ax = plt.subplots(figsize=utils.set_size(textwidth, 1.0, (1, 1), rel_height))
    for SNR in [10, 50]:
        for codebook_entries in [3, 7]:
            q_quant_var = (
                q.filter(
                    pl.col("alg") == "quant_var",
                    pl.col("SNR") == SNR,
                    pl.col("codebook_entries") == codebook_entries,
                    pl.col("add_zeros") == False,
                )
                .groupby("series", maintain_order=True)
                .agg(normalizer_m=pl.col("normalizer").median())
                .take_every(plot_every)
            )
            data = q_quant_var.collect()
            (line,) = axs[1].plot(
                data["series"],
                data["normalizer_m"],
                label=f"({SNR}dB, {codebook_entries})",
                markersize=4,
                markevery=(5, 20),
                alpha=1,
                linewidth=linewidth,
            )
    # axs[1].legend(fontsize=6, loc="center right")
    axs[1].grid()
    # axs[1].xlabel(r"Time $k$")
    axs[1].set_ylabel(r"$\delta^{(k)}$")
    axs[1].set_ylim(1e-2, 10)
    axs[1].set_xlim(0, 15000)
    axs[1].set_yscale("log")
    # axs[1].set_yticks(fontsize=6)
    axs[1].tick_params(axis="both", labelsize=6)
    axs[1].set_xticklabels([])

    # ax = plt.gca()
    # ax.set_xticklabels([])
    # ax.set_position(box)
    # plt.show()
    # utils.savefig(
    #     fig,
    #     f"normalizer-over-time-comparison",
    #     format="pdf",
    # )
    # plt.close()

    for SNR in [10, 50]:
        q_base = (
            q.filter(
                pl.col("alg") == "base",
                pl.col("SNR") == SNR,
                pl.col("add_zeros") == False,
            )
            .groupby("series", maintain_order=True)
            .agg(npm_m=pl.col("npm").median())
            .take_every(plot_every)
        )
        data = q_base.collect()
        (line,) = axs[0].plot(
            data["series"],
            data["npm_m"],
            label=f"({SNR}dB, base)",
            color="k",
            markersize=4,
            markevery=(5, 20),
            alpha=1,
            linewidth=linewidth,
        )
        for codebook_entries in [3, 7]:
            q_quant_var = (
                q.filter(
                    pl.col("alg") == "quant_var",
                    pl.col("SNR") == SNR,
                    pl.col("codebook_entries") == codebook_entries,
                    pl.col("add_zeros") == False,
                )
                .groupby("series", maintain_order=True)
                .agg(npm_m=pl.col("npm").median())
                .take_every(plot_every)
            )
            data = q_quant_var.collect()
            (line,) = axs[0].plot(
                data["series"],
                data["npm_m"],
                label=f"({SNR}dB, {codebook_entries})",
                markersize=4,
                markevery=(5, 20),
                alpha=1,
                linewidth=linewidth,
            )
    axs[0].legend(
        fontsize=6,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.55),
        ncol=3,
        fancybox=True,
        # shadow=True,
    )
    axs[0].grid()
    # axs[0].xlabel(r"Time $k$")
    axs[0].set_ylabel(r"$NPM^{(k)}$ [dB]")
    axs[0].set_ylim(-70, 10)
    axs[0].tick_params(axis="both", labelsize=6)
    axs[0].set_xticklabels([])
    axs[0].set_xlim(0, 15000)

    # ax = plt.gca()
    # ax.set_xticklabels([])
    # ax.set_position(box)
    plt.tight_layout(pad=0.5)

    utils.savefig(
        fig,
        f"npm-normalizer-bits-over-time-comparison",
        format="pdf",
    )
    plt.close()

    # ####################################################################################
    # q_bits_sum = q.groupby(["alg", "SNR", "codebook_entries"]).agg(
    #     bits_s=pl.col("bits").mean()
    # )
    # data = q_bits_sum.collect()
    # max_bitrate = data["bits_s"].max()

    # X_axis = np.arange(len(codebook_entriess))

    # fig, ax = plt.subplots(figsize=utils.set_size(textwidth, 1.0, (1, 1), rel_height))
    # # plt.plot(X_axis, np.ones_like(X_axis), "k--")

    # plt.bar(
    #     X_axis - 0.2,
    #     data.filter(pl.col("alg") == "symbol_mean")
    #     .groupby("codebook_entries")
    #     .agg(bits_s=pl.col("bits_s").mean())
    #     .sort("codebook_entries")["bits_s"]
    #     / max_bitrate
    #     / L,
    #     0.4,
    #     label="SM",
    # )
    # plt.bar(
    #     X_axis + 0.2,
    #     data.filter(pl.col("alg") == "quant_var")
    #     .groupby("codebook_entries")
    #     .agg(bits_s=pl.col("bits_s").mean())
    #     .sort("codebook_entries")["bits_s"]
    #     / max_bitrate
    #     / L,
    #     0.4,
    #     label="QV",
    # )
    # plt.xticks(X_axis, codebook_entriess)
    # plt.legend(fontsize=6)
    # plt.grid()
    # plt.xlabel("Number of symbols")
    # plt.ylabel("R [relative]")
    # plt.tight_layout(pad=0.5)
    # plt.show()

    # utils.savefig(fig, "bits-over-codebook_entries", format="pdf")

    # ####################################################################################
    # frames = [5000, 10000, 15000]
    # for SNR in SNRs:
    #     q_npm_conv = (
    #         q.filter(pl.col("SNR") == SNR, pl.col("series").is_in(frames))
    #         .groupby(["alg", "codebook_entries"])
    #         .agg(npm_conv=pl.col("npm").mean())
    #     )
    #     data = q_npm_conv.collect()
    #     fig, ax = plt.subplots(figsize=utils.set_size(textwidth, 1.0, (1, 1), rel_height))

    #     plt.plot(
    #         np.arange(-2, 8),
    #         np.ones_like(np.arange(-2, 8))
    #         * data.filter(pl.col("alg") == "base")["npm_conv"].mean(),
    #         "k-.",
    #         label="full",
    #     )
    #     plt.bar(
    #         X_axis - 0.2,
    #         data.filter(pl.col("alg") == "symbol_mean")
    #         .groupby("codebook_entries")
    #         .agg(npm_conv=pl.col("npm_conv").mean())
    #         .sort("codebook_entries")["npm_conv"],
    #         0.4,
    #         label="SM",
    #     )
    #     plt.bar(
    #         X_axis + 0.2,
    #         data.filter(pl.col("alg") == "quant_var")
    #         .groupby("codebook_entries")
    #         .agg(npm_conv=pl.col("npm_conv").mean())
    #         .sort("codebook_entries")["npm_conv"],
    #         0.4,
    #         label="QV",
    #     )
    #     plt.xticks(X_axis, codebook_entriess)
    #     plt.legend(fontsize=6)
    #     plt.grid()
    #     plt.ylim(-90, 0)
    #     plt.xlim(-0.5, 4.5)
    #     plt.xlabel("Number of symbols")
    #     plt.ylabel("NPM [dB]")
    #     plt.tight_layout(pad=0.5)
    #     plt.show()

    #     utils.savefig(fig, f"NPM-over-codebook_entries-{SNR}", format="pdf")
