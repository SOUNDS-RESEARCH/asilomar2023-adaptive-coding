import numpy as np
import admm_fq_base as adfb
import admm_fq_huffman_symbol_mean as adfsm
import admm_fq_huffman_quant_var as adfqv
from pythonutils import utils
from pythonutils import dockersim
from pythonutils import huffman as huf


# This is the actual simulation function. We simulate the ADMM algorithm (imported from
# admm_fq_*) for different algorithm onfigurations and parameter combinations.
def _sim_dynamic_tracking(
    alg, L, SNR, codebook_entries, add_zeros, rng=np.random.default_rng()
):
    match alg:
        case "symbol_mean":
            nw = adfsm.Network(L)
        case "quant_var":
            nw = adfqv.Network(L)
        case "base":
            nw = adfb.Network(L)
        case _:
            raise Exception(f"Unknown algorithm type {alg}")

    # ####################################################################################
    # initialize network
    nw.addNode(0, 1.0)
    nw.addNode(1, 1.0)
    nw.addNode(2, 1.0)
    nw.setConnection(0, [1])
    nw.setConnection(1, [2])
    nw.setConnection(2, [0])

    # ####################################################################################
    # generate data for codebook
    nr_codebook_samples = 1000000
    mean = 0
    sd = 10e-4
    training_data = rng.normal(mean, sd, (nr_codebook_samples,))
    if add_zeros == True:
        training_data = np.concatenate(
            [training_data, np.zeros((int(nr_codebook_samples / 15),))]
        )

    sd_mult = 4
    hist_range = [-sd_mult * sd, sd_mult * sd]
    nr_bins = 301
    step = (hist_range[1] - hist_range[0]) / nr_bins
    bins = np.arange(hist_range[0], hist_range[1] + step, step)
    centers = (bins[1:] + bins[:-1]) / 2
    resample_hist_range = hist_range
    resample_nr_bins = codebook_entries
    resample_step = (resample_hist_range[1] - resample_hist_range[0]) / resample_nr_bins
    resample_bins = np.arange(
        resample_hist_range[0],
        resample_hist_range[1] + resample_step / 2,
        resample_step,
    )
    resample_centers = (resample_bins[1:] + resample_bins[:-1]) / 2
    resample_bins_ = resample_bins.copy()
    resample_bins_[0] = -np.inf
    resample_bins_[-1] = np.inf

    resample_training_data = np.array([])
    for i, cent in enumerate(resample_centers):
        pp = training_data[np.isclose(training_data, cent, atol=step / 2)]
        resample_training_data = np.concatenate(
            [resample_training_data, np.ones(pp.shape) * i]
        )
    resample_training_data = resample_training_data.astype(int)

    # ####################################################################################
    # train codebook
    data = list(resample_training_data)
    encoded, tree = huf.huffman_encode(data)
    table = huf.huffman_table(tree)

    # ####################################################################################
    # network and signal parameters
    rho = 1  # ADMM penalty parameter
    stepsize = 0.8  # local step size
    eta = 0.98  # recursive smoothing parameter (forgetting factor)
    M = nw.N  # number of channels
    nr_samples = 250000  # number of input samples
    partitions = 2  # number of IRs changes
    part_len = int(nr_samples / partitions)
    true_norms = [1.0] * M
    hopsize = L

    # ####################################################################################
    # generate signal
    u = rng.normal(size=(nr_samples, 1))
    clean_signal: np.ndarray = u / u.max()
    clean_signal = clean_signal / clean_signal.std(axis=0) * 0.25
    # generate IRs
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

    # ####################################################################################
    # network setup
    # def saveTransmissionHist(node1, node2, var_name, var):
    #     pass
    nw.reset()
    # nw.setOnTransmit(saveTransmissionHist)
    match alg:
        case "symbol_mean":
            increase_mult = 2
            decrease_mult = 1 / increase_mult
            delta_lim = 0.8
            inc_lim = 1 - delta_lim
            dec_lim = 1 + delta_lim
            smoothing = 0.5
            nw.setParameters(rho, stepsize, eta)
            nw.setCodeBook(
                tree,
                table,
                resample_bins_,
                resample_centers,
                sd**2,
                increase_mult,
                decrease_mult,
                inc_lim,
                dec_lim,
                smoothing,
            )
        case "quant_var":
            smoothing = 0.5
            nw.setParameters(rho, stepsize, eta)
            nw.setCodeBook(
                tree,
                table,
                resample_bins_,
                resample_centers,
                sd**2,
                smoothing,
            )
        case "base":
            nw.setParameters(rho, stepsize, eta, 1, 0.0)
        case _:
            raise Exception(f"Unknown algorithm type {alg}")

    for k_admm_fq in range(0, nr_samples - 2 * L, hopsize):
        nw.step(noisy_signals[k_admm_fq : k_admm_fq + 2 * L, :])
        error = []
        bits = 0
        normalizer = 0
        symbol_mean = 0
        for m in range(M):
            node: adfb.NodeProcessor = nw.nodes[m]
            error.append(
                utils.NPM(
                    node.getEstimate(),
                    hf[utils.getPart(k_admm_fq, part_len)][:, m, None],
                )
            )
            if m == 0:
                match alg:
                    case "quant_var":
                        node: adfqv.NodeProcessor
                        normalizer = node.consensus_enc_normalizer
                        bits = node.bit_buffer
                        node.bit_buffer = 0
                    case "symbol_mean":
                        node: adfsm.NodeProcessor
                        normalizer = node.consensus_enc_normalizer
                        symbol_mean = node.local_dig_mean[1]
                        bits = node.bit_buffer
                        node.bit_buffer = 0
                    case "base":
                        bits = 64 * 2 * 2 * L

        yield {
            "npm": np.mean(error),
            "normalizer": normalizer,
            "symbol_mean": symbol_mean,
            "bits": bits,
        }


def dynamic_tracking(runs, seed, num_processes):
    # This has to match the dictionary keys that simulate yields
    return_value_names = ["npm", "normalizer", "symbol_mean", "bits"]

    # Define the tasks, which are basically all the parameter combinations for
    # which the algorithms are supposed to be run
    algs = ["base", "symbol_mean", "quant_var"]
    Ls = [16]
    add_zeross = [False]
    SNRs = [10, 30, 50, 70]
    codebook_entriess = [3, 5, 7, 11, 21]

    tasks = [
        {
            "alg": alg,
            "L": L,
            "SNR": SNR,
            "codebook_entries": codebook_entries,
            "add_zeros": add_zeros,
        }
        for alg in algs
        for L in Ls
        for SNR in SNRs
        for add_zeros in add_zeross
        for codebook_entries in codebook_entriess
    ]

    # Create and run the simulation
    print("Running 'dynamic_tracking'...")
    sim = dockersim.DockerSim(
        _sim_dynamic_tracking,
        tasks,
        return_value_names,
        seed,
        datadir="data",
        file_suffix="dynamic_tracking",
    )
    sim.run(runs=runs, num_processes=num_processes)
