import numpy as np
from typing import Callable


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

        self.var_a = 0.98
        self.var_b = 0.98
        self.var_c = 0.98

        self.res_local_var = 1e-7
        self.res_consensus_var = 1e-7
        self.res_local_mean = 0
        self.res_consensus_mean = 0
        self.res_local_var_hist = []
        self.res_consensus_var_hist = []

        self.delta_min = 1e-8
        self.delta_max = 0.1
        self.delta_scaler = 2
        self.coding_scale = 1
        self.epsilon = 0.0
        self.deltas = []
        self.delta_local = 0.1
        self.delta_consensus = 0.1
        self.delta_local_ind = 0
        self.delta_consensus_ind = 0
        self.delta_local_hist = []
        self.delta_consensus_hist = []

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
        self.res_local_var = 1e-7
        self.res_consensus_var = 1e-7
        self.res_local_mean = 0
        self.res_consensus_mean = 0
        self.res_local_var_hist = []
        self.res_consensus_var_hist = []
        self.delta_local_hist = []
        self.delta_consensus_hist = []
        self.residuals = []

    def setParameters(self, rho, mu, eta, decimals, multiplier):
        self.rho = rho
        self.mu = mu
        self.eta = eta
        self.decimals = decimals
        self.multiplier = multiplier

    def setDeltas(
        self,
        deltas: list,
    ):
        self.deltas = deltas
        self.delta_ind = 0

    def setSignal(self, signal):
        i = self.receive_indices[self.id]
        self.block[:, i] = signal

    def estimateVar(self, instant: np.ndarray, var_est: float, mean_est: float = 0):
        if np.any(np.iscomplex(instant)):
            vec = np.concatenate([np.real(instant), np.imag(instant)])
        else:
            vec = instant
        mean_est = self.var_a * mean_est + (1 - self.var_a) * vec.mean()
        var_old = var_est
        var_est = self.var_b * var_est + (1 - self.var_b) / self.var_c * np.square(
            vec.mean() - mean_est
        )
        return var_est, mean_est, 2 * (var_est - var_old) / (var_est + var_old)

    def getDelta(self, var_diff: float, ref_ind: int):
        ind = ref_ind
        inc_dec = None
        if var_diff >= self.epsilon:
            # delta = ref_delta * self.delta_scaler
            ind -= 1
            inc_dec = 1
        if var_diff < -self.epsilon:
            # delta = ref_delta / self.delta_scaler
            ind += 1
            inc_dec = 0

        ind = np.clip(ind, 0, len(self.deltas) - 1)
        return ind, inc_dec

    def setRounding(self, delta):
        if delta == 0.0:
            self.decimals = np.inf
            self.multiplier = 0.0
        else:
            self.decimals = int(np.log10((1 / delta)))
            self.multiplier = 1 / (delta * 10**self.decimals)

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

    def encodeLocal(self, to_node):
        ind = self.receive_index_ranges[to_node]
        res = self.xy[ind] - self.xy_q[ind]
        self.res_local_var, self.res_local_mean, var_diff = self.estimateVar(
            res, self.res_local_var, self.res_local_mean
        )
        self.res_local_var_hist.append(self.res_local_var)
        self.delta_local_ind, inc_dec = self.getDelta(var_diff, self.delta_local_ind)
        self.delta_local = self.deltas[self.delta_local_ind]
        self.delta_local_hist.append(self.delta_local)
        self.setRounding(self.delta_local)
        res_q = quantizeVariable(self.decimals, self.multiplier, res)
        self.xy_q[ind] = self.xy_q[ind] + res_q
        return res, res_q, inc_dec

    def decodeLocal(self, from_node, res_q, inc_dec):
        i = self.transmit_indices[from_node]
        # print("res_q", res_q)
        self.xy_r[:, i, None] = self.xy_r[:, i, None] + res_q
        return self.xy_r[:, i]

    def encodeConsensus(self):
        ind = self.receive_index_ranges[self.id]
        res = self.z_l[ind] - self.z_l_q[ind]
        self.res_consensus_var, self.res_consensus_mean, var_diff = self.estimateVar(
            res, self.res_consensus_var, self.res_consensus_mean
        )
        self.res_consensus_var_hist.append(self.res_consensus_var)
        self.delta_consensus_ind, inc_dec = self.getDelta(
            var_diff, self.delta_consensus_ind
        )
        self.delta_consensus = self.deltas[self.delta_consensus_ind]
        self.delta_consensus_hist.append(self.delta_consensus)
        self.setRounding(self.delta_consensus)
        res_q = quantizeVariable(self.decimals, self.multiplier, res)
        self.z_l_q[ind] = self.z_l_q[ind] + res_q
        return res, res_q, inc_dec

    def decodeConsensus(self, from_node, res_q, inc_dec):
        ind = self.receive_index_ranges[from_node]
        self.z_l[ind] = self.z_l[ind] + res_q
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
                res_signal_q_trans = transmissionLoss(res_signal_q)
                node2.decodeSignal(node1.id, res_signal_q_trans)

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
                res_local, res_local_q, inc_dec = node1.encodeLocal(node2.id)
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
                    var_name="res_local_q",
                    delta=node1.delta_local,
                    var=res_local_q,
                )
                res_local_q_trans = transmissionLoss(res_local_q)
                node2.decodeLocal(node1.id, res_local_q_trans, inc_dec)

    def transmitConsensus(self):
        node1: NodeProcessor
        for node1 in self.nodes.values():
            for node_ind2 in node1.transmit:
                node2: NodeProcessor = self.nodes[node_ind2]
                if node2 == node1:
                    continue
                res_consensus, res_consensus_q, inc_dec = node1.encodeConsensus()
                # self.onTransmit(
                #     node1=node1.id,
                #     node2=node2.id,
                #     var_name="res_consensus",
                #     var=res_consensus,
                # )
                self.onTransmit(
                    node1=node1.id,
                    node2=node2.id,
                    var_name="res_consensus_q",
                    delta=node1.delta_consensus,
                    var=res_consensus_q,
                )
                res_consensus_q_trans = transmissionLoss(res_consensus_q)
                node2.decodeConsensus(node1.id, res_consensus_q_trans, inc_dec)

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

    def setParameters(self, rho, mu, eta, global_ds, q_step, SNR_c):
        self.SNR_c = SNR_c
        self.global_ds = global_ds
        if q_step == 0.0:
            self.decimals = np.inf
            self.multiplier = 0.0
        else:
            self.decimals = int(np.log10((1 / q_step)))
            self.multiplier = 1 / (q_step * 10**self.decimals)

        node: NodeProcessor
        for node in self.nodes.values():
            node.setParameters(rho, mu, eta, self.decimals, self.multiplier)

    def setDeltas(self, deltas: list):
        node: NodeProcessor
        for node in self.nodes.values():
            node.setDeltas(deltas)

    def onTransmit(self, node1, node2, var_name, delta, var):
        if self.onTransmitCallback is not None:
            self.onTransmitCallback(node1, node2, var_name, delta, var)

    def setOnTransmit(self, callback: Callable):
        self.onTransmitCallback = callback


def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(-2 * np.pi * 1j / N)
    W = np.power(omega, i * j) / np.sqrt(N)
    return W


def quantizeVariable(decimals, multiplier, var):
    if decimals == np.inf:
        return var
    return np.round(var * multiplier, decimals) / multiplier


def transmissionLoss(var):
    return var


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array) or np.abs(value - array[idx - 1]) < np.abs(value - array[idx])
    ):
        return array[idx - 1]
    else:
        return array[idx]


# def noisifyVariable(self, var):
#     if np.any(np.iscomplex(var)):
#         noise = self.rng.normal(
#             loc=0, scale=np.sqrt(2) / 2, size=(len(var), 2)
#         ).view(np.complex128)
#     else:
#         noise = self.rng.normal(loc=0, scale=np.sqrt(1), size=(len(var), 1))
#     var_n = np.var(var) / self.SNR_c
#     return var + noise * np.sqrt(var_n)
