import numpy as np
from typing import Callable
from pythonutils import huffman as hf


class NodeProcessor:
    def __init__(self, id, filter_len, rho=1.0, mu=0.5, eta=0.98):
        # identifier for node
        self.id = id
        # definition of signal connections
        self.receive = None
        self.transmit = None
        # number of FIR filter taps
        self.L = filter_len
        # step size / penalty parameter
        self.rho = rho
        self.mu = mu
        self.eta = eta
        # number of "channels"
        self.N = None
        # signal block
        self.block = None
        self.block_q = None
        # local primal
        self.x = None
        # local dual
        self.y = None
        # variable to transmit
        self.xy = None
        self.xy_q = None
        # local dual est old
        self.z_l = None
        # CR matrix
        self.R_xp_ = None

        self.var_a = 0.0
        self.var_b = 0.98
        self.var_c = 0.98

        self.res_local_var = None
        self.last_local_var = None
        self.res_consensus_var = None
        self.last_consensus_var = None
        self.res_local_mean = None
        self.res_consensus_mean = None
        self.local_dig_mean_hist = []
        self.res_consensus_var_hist = []

        self.increase_mult = 1.5
        self.decrease_mult = 1 / self.increase_mult

        self.enc_normalizer = None
        self.dec_normalizer = None

        self.bit_buffer = 0

        self.residuals = []

    def reset(self):
        # signal block
        self.block = np.zeros(shape=(self.L * 2, self.N), dtype=np.complex128)
        self.block_q = np.zeros_like(self.block)
        # local primal
        self.x = np.zeros(shape=(self.L * self.N, 1), dtype=np.complex128)
        # local dual
        self.y = np.zeros(shape=(self.L * self.N, 1), dtype=np.complex128)
        self.xy = np.zeros_like(self.y)
        self.xy_q = np.zeros_like(self.y)
        self.xy_r = np.zeros((self.L, len(self.transmit)), dtype=np.complex128)

        self.receive_indices = {}
        self.receive_index_ranges = {}
        for i, m in enumerate(self.receive):
            self.receive_indices[m] = i
            self.receive_index_ranges[m] = range(i * self.L, (i + 1) * self.L)

        self.transmit_indices = {}
        self.transmit_index_ranges = {}
        for i, m in enumerate(self.transmit):
            self.transmit_indices[m] = i
            self.transmit_index_ranges[m] = range(i * self.L, (i + 1) * self.L)

        self.local_enc_normalizer = np.ones((len(self.receive),))
        self.local_dec_normalizer = np.ones((len(self.transmit),))
        self.local_enc_normalizer_hist = []
        self.local_dec_normalizer_hist = []
        self.consensus_enc_normalizer = 1
        self.consensus_dec_normalizer = np.ones((len(self.receive),))
        self.consensus_enc_normalizer_hist = []
        self.consensus_dec_normalizer_hist = []

        # local copy of global primal
        self.z_l = np.zeros(shape=(self.L * self.N, 1), dtype=np.complex128) / np.sqrt(
            self.L
        )
        self.z_l[self.receive_index_ranges[self.id]] = 1
        self.z_l_q = np.zeros_like(self.z_l)

        F_LxL: np.ndarray = DFT_matrix(self.L)
        F_2Lx2L: np.ndarray = DFT_matrix(2 * self.L)
        W_01_Lx2L = np.concatenate([np.zeros(shape=(self.L, self.L)), np.eye(self.L)]).T
        W_10_2LxL = np.concatenate([np.eye(self.L), np.zeros(shape=(self.L, self.L))])
        self.W_01_Lx2L_fq = F_LxL @ W_01_Lx2L @ np.linalg.inv(F_2Lx2L)
        self.W_10_2LxL_fq = F_2Lx2L @ W_10_2LxL @ np.linalg.inv(F_LxL)
        W_10_Lx2L = np.concatenate([np.eye(self.L), np.zeros(shape=(self.L, self.L))]).T
        W_01_2LxL = np.concatenate([np.zeros(shape=(self.L, self.L)), np.eye(self.L)])
        self.W_10_Lx2L_fq = F_LxL @ W_10_Lx2L @ np.linalg.inv(F_2Lx2L)
        self.W_01_2LxL_fq = F_2Lx2L @ W_01_2LxL @ np.linalg.inv(F_LxL)
        self.W_R = self.W_01_Lx2L_fq.conj().T @ self.W_01_Lx2L_fq

        self.first = True

        self.res_local_var = np.ones((len(self.receive),)) * 1e-7
        self.last_local_var = np.ones((len(self.receive),)) * 1e-7
        self.res_consensus_var = 1e-7
        self.last_consensus_var = 1e-7
        self.res_local_mean = np.zeros((len(self.receive),))
        self.res_consensus_mean = 0

        self.local_dig_mean_hist = []
        self.res_consensus_var_hist = []
        self.residuals = []
        self.bit_buffer = 0

    def setParameters(self, rho, mu, eta):
        self.rho = rho
        self.mu = mu
        self.eta = eta

    def setCodebook(
        self,
        tree,
        table,
        bins,
        centers,
        ref_var,
        increase_mult,
        decrease_mult,
        inc_lim,
        dec_lim,
        smoothing,
    ):
        self.tree = tree
        self.table = table
        self.bins = bins
        self.centers = centers
        self.mirror_bin = int((len(centers) - 1) / 2)
        self.ref_var = ref_var
        self.res_local_var = np.ones((self.N,)) * ref_var
        self.last_local_var = np.ones((self.N,)) * ref_var
        self.res_consensus_var = ref_var
        self.last_consensus_var = ref_var
        self.increase_mult = increase_mult
        self.decrease_mult = decrease_mult
        self.inc_lim = inc_lim
        self.dec_lim = dec_lim
        self.local_dig_mean = np.ones((len(self.receive),)) * self.mirror_bin / 2
        self.cons_dig_mean = self.mirror_bin / 2
        self.var_a = smoothing

    def setSignal(self, signal):
        i = self.receive_indices[self.id]
        self.block[:, i] = signal

    def encodeSignal(self):
        i = self.receive_indices[self.id]
        res = self.block[:, i, None] - self.block_q[:, i, None]
        # res_q = quantizeVariable(self.decimals, self.multiplier, res)
        res_q = res
        self.block_q[:, i, None] = self.block_q[:, i, None] + res_q
        return res, res_q

    def decodeSignal(self, from_node, res_q):
        i = self.receive_indices[from_node]
        self.block[:, i, None] = self.block[:, i, None] + res_q
        return self.block[:, i]

    # def encodeSignal(self):
    #     i = self.receive_indices[self.id]
    #     return self.block[:, i, None]

    # def decodeSignal(self, from_node, res_q):
    #     i = self.receive_indices[from_node]
    #     self.block[:, i, None] = res_q
    #     return self.block[:, i]

    def digitize(self, vec):
        x = np.concatenate([vec.real, vec.imag]).squeeze()
        # print(x.dtype)
        return list(np.searchsorted(self.bins, x, side="left") - 1)

    def dedigitize(self, digitized):
        nn = self.centers[digitized]
        pp = nn[: self.L] + 1j * nn[self.L :]
        return pp.reshape(self.L, 1)

    def hufencode(self, digitized):
        encoded = hf._encode(digitized, self.table)
        return list(encoded)

    def hufdecode(self, encoded):
        decoded = hf.huffman_decode(encoded, self.tree)
        return list(decoded)

    def encodeLocal(self, to_node):
        ind = self.receive_index_ranges[to_node]
        i = self.receive_indices[to_node]

        # compute residual
        res = self.xy[ind] - self.xy_q[ind]

        # value to symbol
        digitized = self.digitize(res / self.local_enc_normalizer[i])

        # compute symbol distribution mean
        digitized_arr = np.asarray(digitized)
        digitized_arr[digitized_arr > self.mirror_bin] = (
            len(self.centers) - 1
        ) - digitized_arr[digitized_arr > self.mirror_bin]
        self.local_dig_mean[i] = self.var_a * self.local_dig_mean[i] + (
            1 - self.var_a
        ) * np.mean(digitized_arr)

        # compute local quantized residual
        res_q = self.dedigitize(digitized) * self.local_enc_normalizer[i]

        self.xy_q[ind] = self.xy_q[ind] + res_q

        # determine scaling update for next frame
        inc_dec = 0
        if 2 * self.local_dig_mean[i] / self.mirror_bin > self.dec_lim:
            inc_dec = -1
            self.local_enc_normalizer[i] *= self.decrease_mult
        if 2 * self.local_dig_mean[i] / self.mirror_bin < self.inc_lim:
            inc_dec = 1
            self.local_enc_normalizer[i] *= self.increase_mult

        # encode
        encoded = self.hufencode(digitized)
        self.bit_buffer += len(list(hf._bits_from_bytes(encoded)))
        return encoded, inc_dec

    def decodeLocal(self, from_node, encoded, inc_dec):
        i = self.transmit_indices[from_node]
        # print("decode local")

        # decode
        digitized = self.hufdecode(encoded)

        # symbol to value
        res_q = self.dedigitize(digitized) * self.local_dec_normalizer[i]
        # print(res_q)

        # update local
        self.xy_r[:, i, None] = self.xy_r[:, i, None] + res_q

        # determine scaling update for next frame
        if inc_dec == 1:
            self.local_dec_normalizer[i] *= self.increase_mult
        if inc_dec == -1:
            self.local_dec_normalizer[i] *= self.decrease_mult

        return self.xy_r[:, i]

    def encodeConsensus(self):
        ind = self.receive_index_ranges[self.id]
        # print("encode cons")

        # compute residual
        res = self.z_l[ind] - self.z_l_q[ind]

        # value to symbol
        digitized = self.digitize(res / self.consensus_enc_normalizer)

        # compute symbol distribution mean
        digitized_arr = np.asarray(digitized)
        digitized_arr[digitized_arr > self.mirror_bin] = (
            len(self.centers) - 1
        ) - digitized_arr[digitized_arr > self.mirror_bin]
        self.cons_dig_mean = self.var_a * self.cons_dig_mean + (
            1 - self.var_a
        ) * np.mean(digitized_arr)

        # compute consensus quantized residual
        res_q = self.dedigitize(digitized) * self.consensus_enc_normalizer
        # print(res_q)
        self.z_l_q[ind] = self.z_l_q[ind] + res_q

        # determine scaling update for next frame
        inc_dec = 0
        if 2 * self.cons_dig_mean / self.mirror_bin > self.dec_lim:
            inc_dec = -1
            self.consensus_enc_normalizer *= self.decrease_mult
        if 2 * self.cons_dig_mean / self.mirror_bin < self.inc_lim:
            inc_dec = 1
            self.consensus_enc_normalizer *= self.increase_mult

        # encode
        encoded = self.hufencode(digitized)
        self.bit_buffer += len(list(hf._bits_from_bytes(encoded)))
        return encoded, inc_dec

    def decodeConsensus(self, from_node, encoded, inc_dec):
        ind = self.receive_index_ranges[from_node]
        i = self.receive_indices[from_node]

        # decode
        digitized = self.hufdecode(encoded)

        # symbol to value
        res_q = self.dedigitize(digitized) * self.consensus_dec_normalizer[i]

        # update local
        self.z_l[ind] = self.z_l[ind] + res_q

        # determine scaling update for next frame
        if inc_dec == 1:
            self.consensus_dec_normalizer[i] *= self.increase_mult
        if inc_dec == -1:
            self.consensus_dec_normalizer[i] *= self.decrease_mult

        return self.z_l[ind]

    def solveLocal(self):
        R = self.construct_Rxp()  # construct matrix R_x+
        self.R_xp_ = R if self.first else self.eta * self.R_xp_ + (1 - self.eta) * R
        y = self.R_xp_ @ self.x + self.y + self.rho * (self.x - self.z_l)
        V = 1 / (np.diag(self.R_xp_).reshape(self.N * self.L, 1) + self.rho)
        self.x = self.x - self.mu * V * y
        self.xy = self.rho * self.x + self.y

    def construct_Rxp(self):
        R_xp = np.zeros(shape=(self.L * self.N, self.L * self.N), dtype=np.complex128)
        for i in range(self.N):
            for j in range(self.N):
                R_xx = self.compute_Rxx(i, j)
                if i != j:
                    # R_xp[j, i] = -R_xx
                    R_xp[
                        j * self.L : j * self.L + self.L,
                        i * self.L : i * self.L + self.L,
                    ] = -R_xx
                else:
                    for n in range(self.N):
                        if i != n:
                            # R_xp[n, n] += R_xx
                            R_xp[
                                n * self.L : n * self.L + self.L,
                                n * self.L : n * self.L + self.L,
                            ] += R_xx
        return R_xp

    def compute_Rxx(self, i, j):
        D_xi = np.diag(self.block[:, i])
        D_xj = np.diag(self.block[:, j])
        R_xx = self.W_10_Lx2L_fq @ D_xi.conj().T @ self.W_R @ D_xj @ self.W_10_2LxL_fq
        return R_xx

    def computeConsensus(self):
        ind = self.receive_index_ranges[self.id]
        i = self.transmit_indices[self.id]
        self.xy_r[:, i, None] = self.xy[ind]
        self.z_l[ind] = self.xy_r.mean(1, keepdims=True)

    def updateDual(self):
        res = self.rho * (self.x - self.z_l)
        self.residuals.append(res)
        self.y = self.y + res

    def getEstimate(self) -> np.ndarray:
        return self.z_l[self.receive_index_ranges[self.id]]


class Network:
    def __init__(self, L, rng=np.random.default_rng()):
        # number of FIR filter taps
        self.L = L
        # number of nodes in network
        self.N = 0
        # central processor for gloabl update
        self.central_processor = None
        # node objecs
        self.nodes = {}
        # connections between nodes
        self.connections = {}
        # node index
        self.node_index = {}
        # network matrix
        self.A = None
        self.A_ = None
        # global update weights
        self.g = None
        self.global_ds = 1  # global update only done each K frames
        self.SNR_c = 1
        self.rng = rng

        self.onTransmitCallback: Callable = None

    def reset(self):
        node: NodeProcessor
        for node in self.nodes.values():
            node.reset()

    def addNode(self, id, rho=1.0, mu=0.5, eta=0.98) -> int:
        self.nodes[id] = NodeProcessor(id, self.L, rho, mu, eta)
        self.N = len(self.nodes)
        self.generateNetworkData()

    def removeNode(self, id):
        del self.nodes[id]
        self.N = len(self.nodes)
        self.generateNetworkData()

    def setConnection(self, node_id, connections):
        self.connections[node_id] = connections
        self.generateNetworkData()

    def generateNetworkData(self):
        self.A = np.zeros(shape=(self.N, self.N))
        self.A_ = np.diag(np.repeat(1, self.N))
        self.node_index = {}
        for i, node_key in enumerate(self.nodes):
            self.node_index[node_key] = i

        for i, node_key in enumerate(self.nodes):
            if node_key in self.connections:
                for connection in self.connections[node_key]:
                    j = self.node_index[connection]
                    self.A[i, j] = 1
                    self.A_[i, j] = 1

        for node_key, i in self.node_index.items():
            node: NodeProcessor = self.nodes[node_key]
            node.N = np.sum(self.A_[:, i])
            node.receive = np.where(self.A_[:, i])[0]
            node.transmit = np.where(self.A_[i, :])[0]

    def step(self, signal):
        signal_f = np.fft.fft(signal, axis=0)
        self.setSignals(signal_f)
        self.transmitSignals()
        self.localPrimalUpdate()
        self.transmitLocalVar()
        self.computeConsensus()
        self.consensusNormalization()
        self.transmitConsensus()
        self.localDualUpdate()

        node: NodeProcessor
        # for node in self.nodes.values():
        #     node.local_dig_mean_hist.append(node.local_dig_mean.copy())
        #     node.res_consensus_var_hist.append(node.res_consensus_var)
        #     node.local_enc_normalizer_hist.append(node.local_enc_normalizer.copy())
        #     node.local_dec_normalizer_hist.append(node.local_dec_normalizer.copy())
        #     node.consensus_enc_normalizer_hist.append(node.consensus_enc_normalizer)
        #     node.consensus_dec_normalizer_hist.append(
        #         node.consensus_dec_normalizer.copy()
        #     )
        if self.nodes[0].first:
            for node in self.nodes.values():
                node.first = False

    def setSignals(self, signal):
        node: NodeProcessor
        for node in self.nodes.values():
            node.setSignal(signal[:, node.id])

    def transmitSignals(self):
        node1: NodeProcessor
        for node1 in self.nodes.values():
            res_signal, res_signal_q = node1.encodeSignal()
            for node_ind2 in node1.transmit:
                node2: NodeProcessor = self.nodes[node_ind2]
                if node2 == node1:
                    continue
                # self.onTransmit(
                #     node1=node1.id,
                #     node2=node2.id,
                #     var_name="res_signal",
                #     var=res_signal,
                # )
                # self.onTransmit(
                #     node1=node1.id,
                #     node2=node2.id,
                #     var_name="res_signal_q",
                #     var=res_signal_q,
                # )
                # res_signal_q_trans = transmissionLoss(res_signal_q)
                node2.decodeSignal(node1.id, res_signal_q)

    def localPrimalUpdate(self):
        node: NodeProcessor
        for node in self.nodes.values():
            node.solveLocal()

    def transmitLocalVar(self):
        node1: NodeProcessor
        for node1 in self.nodes.values():
            for node_ind2 in node1.receive:
                node2: NodeProcessor = self.nodes[node_ind2]
                if node2 == node1:
                    continue
                encoded, inc_dec = node1.encodeLocal(node2.id)
                # self.onTransmit(
                #     node1=node1.id,
                #     node2=node2.id,
                #     var_name="res_local",
                #     delta=node1.delta_local
                #     var=res_local,
                # )
                self.onTransmit(
                    node1=node1.id,
                    node2=node2.id,
                    var_name="res_local_encoded",
                    var=encoded,
                )
                # res_local_q_trans = transmissionLoss(res_local_q)
                node2.decodeLocal(node1.id, encoded, inc_dec)

    def transmitConsensus(self):
        node1: NodeProcessor
        for node1 in self.nodes.values():
            for node_ind2 in node1.transmit:
                node2: NodeProcessor = self.nodes[node_ind2]
                if node2 == node1:
                    continue
                encoded, inc_dec = node1.encodeConsensus()
                # self.onTransmit(
                #     node1=node1.id,
                #     node2=node2.id,
                #     var_name="res_consensus",
                #     var=res_consensus,
                # )
                self.onTransmit(
                    node1=node1.id,
                    node2=node2.id,
                    var_name="res_consensus_encoded",
                    var=encoded,
                )
                # res_consensus_q_trans = transmissionLoss(res_consensus_q)
                node2.decodeConsensus(node1.id, encoded, inc_dec)

    def computeConsensus(self):
        node: NodeProcessor
        for node in self.nodes.values():
            node.computeConsensus()

    def consensusNormalization(self):
        pp = 0
        node: NodeProcessor
        for node in self.nodes.values():
            ind = node.receive_index_ranges[node.id]
            pp += np.linalg.norm(node.z_l[ind]) ** 2
        pp = np.sqrt(pp)
        for node in self.nodes.values():
            ind = node.receive_index_ranges[node.id]
            node.z_l[ind] = node.z_l[ind] / pp

    def localDualUpdate(self):
        node: NodeProcessor
        for node in self.nodes.values():
            node.updateDual()

    def setParameters(self, rho, mu, eta):
        node: NodeProcessor
        for node in self.nodes.values():
            node.setParameters(rho, mu, eta)

    def setCodeBook(
        self,
        tree,
        table,
        bins,
        centers,
        ref_var,
        increase_mult,
        decrease_mult,
        inc_lim,
        dec_lim,
        smoothing,
    ):
        node: NodeProcessor
        for node in self.nodes.values():
            node.setCodebook(
                tree,
                table,
                bins,
                centers,
                ref_var,
                increase_mult,
                decrease_mult,
                inc_lim,
                dec_lim,
                smoothing,
            )

    def onTransmit(self, node1, node2, var_name, var):
        if self.onTransmitCallback is not None:
            self.onTransmitCallback(node1, node2, var_name, var)

    def setOnTransmit(self, callback: Callable):
        self.onTransmitCallback = callback


def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(-2 * np.pi * 1j / N)
    W = np.power(omega, i * j) / np.sqrt(N)
    return W
