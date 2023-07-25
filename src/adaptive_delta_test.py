# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import admm_fq_hist as adf
import admm_fq_base as adfb

# import admm_fq_residual as adfr
import admm_fq_prires as adfpr
import utils
import importlib
import seaborn as sns


# import cvxpy
def getPart(index, part_len) -> int:
    return int(np.floor(index / part_len))


# %%
importlib.reload(adfb)
importlib.reload(adfpr)
importlib.reload(utils)

# %%
rng = np.random.default_rng()


# %%
L = 16

# %%
nwb = adfb.Network(L)
nwb.addNode(0, 1.0)
nwb.addNode(1, 1.0)
nwb.addNode(2, 1.0)
nwb.setConnection(0, [1])
nwb.setConnection(1, [2])
nwb.setConnection(2, [0])

# %%
nwpr = adfpr.Network(L)
nwpr.addNode(0)
nwpr.addNode(1)
nwpr.addNode(2)
nwpr.setConnection(0, [1])
nwpr.setConnection(1, [2])
nwpr.setConnection(2, [0])

# %%
SNR = 20
SNR_c = np.inf
rho = 1
stepsize = 0.8
eta = 0.98
M = nwb.N
nr_samples = 100000
partitions = 2
part_len = int(nr_samples / partitions)


# %%
true_norms = [1.0, 1.0, 1.0, 1.0]

u = rng.normal(size=(nr_samples, 1))
clean_signal: np.ndarray = u / u.max()

clean_signal = clean_signal / clean_signal.std(axis=0) * 0.25

# %%
h = {}
hf = {}
noisy_signals = np.empty((0, M))
for part in range(partitions):
    h_, hf_ = utils.generateRandomIRs(L, M, true_norms, rng)
    h[part] = h_
    hf[part] = hf_
    noisy_signals_ = utils.getNoisySignal(
        clean_signal[part * part_len : (part + 1) * part_len], h_, SNR, rng
    )
    noisy_signals = np.concatenate([noisy_signals, noisy_signals_])

# %%
transmission_hist = {}
hopsize = L
npm = []
# nw.setOnTransmit(saveTransmissionHist)
nwb.reset()
nwb.setParameters(rho, stepsize, eta, 1, 0.0)
# nwb.setDeltas(1e-8, 0.1, 0.1, 2, 0.001)
for k_admm_fq in range(0, nr_samples - 2 * L, hopsize):
    nwb.step(noisy_signals[k_admm_fq : k_admm_fq + 2 * L, :])
    error = []
    for m in range(M):
        node: adfb.NodeProcessor = nwb.nodes[m]
        error.append(
            utils.NPM(
                node.getEstimate(), hf[utils.getPart(k_admm_fq, part_len)][:, m, None]
            )
        )
    # npm.append(np.mean(error))
    npm.append(error)
npm = np.asarray(npm)
nw_ = nwb

# %%
lambd = 0.25
transmission_hist = {}
hopsize = L
npm = []
# nw.setOnTransmit(saveTransmissionHist)
nwpr.reset()
nwpr.setParameters(rho, stepsize, eta, lambd, 0.0)
nwpr.setDeltas(1e-5, 0.1, 0.1, 1.005, 0.0001)
# nwpr.setDeltas(1e-8, 0.1, 0.1, 2, 0.001)
for k_admm_fq in range(0, nr_samples - 2 * L, hopsize):
    nwpr.step(noisy_signals[k_admm_fq : k_admm_fq + 2 * L, :])
    error = []
    for m in range(M):
        node: adfpr.NodeProcessor = nwpr.nodes[m]
        error.append(
            utils.NPM(
                node.getEstimate(), hf[utils.getPart(k_admm_fq, part_len)][:, m, None]
            )
        )
    # npm.append(np.mean(error))
    npm.append(error)
npm = np.asarray(npm)
nw_ = nwpr


# %%
fig = plt.figure(figsize=(5, 3))
# plt.title(rf"SNR={SNR}dB")
plt.xlabel("Frames")
plt.ylabel("NPM [dB]")
labels = [f"node{label}" for label in range(M)]
plt.plot(20 * np.log10(npm), label=labels)
# plt.ylim(-50, 0)
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()
# utils.savefig(fig, "npm")

# %%
fig = plt.figure(figsize=(5, 3))
plt.title(rf"Local")
plt.xlabel("Frames")
plt.ylabel(r"$\Delta$")
# labels = [f"node{label}" for label in range(M)]
for node in nw_.nodes.values():
    plt.plot(node.delta_local_hist, label=node.id)
    print("Avg", np.mean(node.delta_local_hist))
# plt.xlim(0, 6000)
# plt.ylim(0, 0.002)
plt.yscale("log")
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()
# utils.savefig(fig, "localdelta")

# %%
fig = plt.figure(figsize=(5, 3))
plt.title(rf"Consensus")
plt.xlabel("Frames")
plt.ylabel(r"$\Delta$")
# labels = [f"node{label}" for label in range(M)]
for node in nw_.nodes.values():
    plt.plot(node.delta_consensus_hist, label=node.id)
    print("Avg", np.mean(node.delta_consensus_hist))
# plt.xlim(0, 6000)
# plt.ylim(0, 0.002)
plt.yscale("log")
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()
# utils.savefig(fig, "consensusdelta")

# %%
fig = plt.figure(figsize=(5, 3))
plt.title(rf"ADMM dual residual")
plt.xlabel("Frames")
plt.ylabel(r"residual")
# labels = [f"node{label}" for label in range(M)]
for node in nw_.nodes.values():
    plt.plot(np.asarray(node.residuals).squeeze(), label=node.id)
    # print(np.mean(node.delta_local_state_hist))
# plt.xlim(3000, 3500)
# plt.ylim(-0.1, 0.1)
# plt.yscale("log")
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()
# utils.savefig(fig, "localstate")
# %%
fig = plt.figure(figsize=(5, 3))
plt.title(rf"norm of local var")
plt.xlabel("Frames")
plt.ylabel(r"norm")
# labels = [f"node{label}" for label in range(M)]
for node in nw_.nodes.values():
    plt.plot(np.asarray(node.xs).squeeze(), label=node.id)
    # print(np.mean(node.delta_local_state_hist))
# plt.xlim(100, 300)
# plt.ylim(-0.1, 0.1)
# plt.yscale("log")
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()
# utils.savefig(fig, "localstate")# %%
fig = plt.figure(figsize=(5, 3))
plt.title(rf"norm of mapped global var")
plt.xlabel("Frames")
plt.ylabel(r"norm")
# labels = [f"node{label}" for label in range(M)]
for node in nw_.nodes.values():
    plt.plot(np.asarray(node.zs).squeeze(), label=node.id)
    # print(np.mean(node.delta_local_state_hist))
# plt.xlim(100, 300)
# plt.ylim(-0.1, 0.1)
# plt.yscale("log")
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()
# utils.savefig(fig, "localstate")

# %%
import seaborn as sns
from seaborn import objects as so
from seaborn import axes_style
import polars as pl
import matplotlib.pyplot as plt
import utils

# %%
q = pl.scan_csv("../data/results_*.csv").with_columns(pl.col("npm").log10() * 20)

# %%
plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": False,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "font.size": 7,
    }
)
textwidth = 245
linewidth = 1.2
fig, ax = plt.subplots(figsize=utils.set_size(textwidth, 1.0, (1, 1), 0.4))
plt.plot(
    q.filter(pl.col("alg") == "base")
    .groupby("series", maintain_order=True)
    .agg(pl.col("npm").median())
    .select(pl.col("npm"))
    .collect(),
    "k-",
    label=f"optimal",
    markersize=4,
    markevery=(1, 500),
    alpha=1,
    linewidth=linewidth,
)
plt.legend()
plt.tight_layout(pad=0.5)
plt.grid()
plt.show()

utils.savefig(fig, "npm")

# p = (
#     so.Plot(
#         data=q.collect(),
#         x="series",
#         y="npm",
#         color="alg",
#         linestyle="alg",
#     )
#     .theme(
#         axes_style("whitegrid"),
#     )
#     .add(
#         so.Line(linewidth=1),
#         so.Agg("median"),
#     )
#     .add(
#         so.Band(),
#         so.Est(
#             "median",
#             ("se", 1.96),
#         ),
#     )
#     .label(
#         x="Time [frames]",
#         y="NPM [dB]",
#     )
#     .on(ax)
# )
# sns.move_legend(ax, "center left")
# p.save(
#     "test.png",
#     dpi=300,
#     bbox_inches="tight",
# )
# # plt.legend(loc=1)
# plt.show()

# %%
