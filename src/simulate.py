import sys
import numpy as np
import admm_fq_base as adfb
import admm_fq_prires as adfp

from pythonutils import utils
from pythonutils import dockersim

# The parameters of this script, the number of relizations per parameter combination,
# the seed of the random number generator, the number of processes to be used on parallel
runs = int(sys.argv[1])
seed = int(sys.argv[2])
num_processes = int(sys.argv[3])
only_plot = sys.argv[4] if len(sys.argv) > 4 else "--sim"

if only_plot == "--plot":
    import plot

    exit(0)


# This is the actual simulation function. We simulate the ADMM algorithm (imported from
# admm_fq_*) for different algorithm onfigurations and parameter combinations.
def simulate1(alg, rng=np.random.default_rng()):
    L = 16

    match alg:
        case "adaptive":
            nw = adfp.Network(L)
        case "base":
            nw = adfb.Network(L)
        case _:
            nw = adfb.Network(L)

    nw.addNode(0, 1.0)
    nw.addNode(1, 1.0)
    nw.addNode(2, 1.0)
    nw.setConnection(0, [1])
    nw.setConnection(1, [2])
    nw.setConnection(2, [0])

    SNR = 20
    rho = 1
    stepsize = 0.8
    eta = 0.98
    M = nw.N
    nr_samples = 150000
    partitions = 3
    part_len = int(nr_samples / partitions)

    true_norms = [1.0, 1.0, 1.0, 1.0]

    u = rng.normal(size=(nr_samples, 1))
    clean_signal: np.ndarray = u / u.max()

    clean_signal = clean_signal / clean_signal.std(axis=0) * 0.25

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

    hopsize = L
    nw.reset()
    if alg == "adaptive":
        nw.setParameters(rho, stepsize, eta, 0.25, 0.0)
        nw.setDeltas(1e-5, 0.1, 0.1, 1.005, 0.0001)
    else:
        nw.setParameters(rho, stepsize, eta, 1, 0.0)
    for k_admm_fq in range(0, nr_samples - 2 * L, hopsize):
        nw.step(noisy_signals[k_admm_fq : k_admm_fq + 2 * L, :])
        error = []
        # var = 0
        residual = 0
        delta_consensus = 0
        for m in range(M):
            node: adfp.NodeProcessor = nw.nodes[m]
            error.append(
                utils.NPM(
                    node.getEstimate(),
                    hf[utils.getPart(k_admm_fq, part_len)][:, m, None],
                )
            )
            if alg == "adaptive" and m == 0:
                residual = node.old_res
                delta_consensus = node.delta_consensus
                # delta_local = node.delta_local
        yield {"npm": np.mean(error), "res": residual, "delta_consensus": delta_consensus}


# This has to match the dictionary keys that simulate yields
return_value_names = ["npm", "res", "delta_consensus"]

# Define the tasks, which are basically all the parameter combinations for
# which the algorithms are supposed to be run
tasks = [{"alg": alg} for alg in ["base", "adaptive"]]

# Create and run the simulation
sim = dockersim.DockerSim(
    simulate1, tasks, return_value_names, seed, datadir="data", file_suffix="sim1"
)
sim.run(runs=runs, num_processes=num_processes)


# plot the relevant figures (in script plot.py)
import plot
