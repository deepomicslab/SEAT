import numpy as np
from math import log2
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.metrics.pairwise import pairwise_kernels
from scipy import sparse
import heapq
import itertools
import networkx as nx
from itertools import chain
import pandas as pd
from queue import Queue
import kmeans1d
import scipy.linalg as la
from scipy.sparse import csgraph
import time
import copy
from itertools import compress

from . import graph_metric
import sys
sys.setrecursionlimit(10000)


class Node():

    def __init__(self, graph_stats, node_id, children, vs, parent=None,
                 is_individual=True,
                 is_leaf=True):
        self.id = node_id
        self.parent = parent
        self.children = children
        self.vs = vs
        self.dist = 1
        self.height = 0
        self.is_individual = is_individual
        self.is_leaf = is_leaf
        self.split_se = np.nan
        self.graph_stats = graph_stats

        self.g = 0.
        self.s = 0.
        self.V = 0.
        self.V_log_V = 0.
        self.d_log_d = 0.
        self.se = 0.

        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.graph_stats
        self.V = graph_metric.get_v(M, sparse_m, self.vs)
        self.log_V = log2(self.V)
        self.V_log_V = self.V * self.log_V
        self.s = graph_metric.get_s(M, sparse_m, self.vs)
        self.g = self.V - self.s
        self.d_log_d = np.sum(d_log_d[self.vs])
        if parent:
            self.se = graph_metric.get_node_se(vG, self.g, self.V, parent.V)
        else:
            self.se = 0

    def merge(self, node_id, node1, node2, is_leaf=False):
        if is_leaf:
            children = node1.children + node2.children
        else:
            children = [node1, node2]
        vs = node1.vs + node2.vs
        node = Node(self.graph_stats, node_id, children, vs, parent=self)
        if not is_leaf:
            node.dist = max(node1.dist, node2.dist) + 1
        node.is_leaf = is_leaf
        node1.parent = node
        node2.parent = node
        self.children.append(node)  # self is root
        return node

    def __repr__(self):
        return 'id:{}'.format(self.id)


class pySETree():

    def __init__(self, aff_m, knn_m, min_k=2, max_k=10,
                 objective='SE',
                 strategy='top_down',
                 split_se_cutoff=0.05,
                 verbose=False
                 ):
        self.strategy = strategy
        self.objective = objective
        self.min_k = min_k
        self.max_k = max_k
        self.split_se_cutoff = split_se_cutoff
        self.verbose = verbose

        self.vertex_num = aff_m.shape[0]
        if self.max_k > self.vertex_num:
            self.max_k = self.vertex_num - 1

        self.ks = range(self.min_k, self.max_k+1)

        if strategy == 'top_down':
            self.node_id = 2*self.vertex_num - 3
        else:
            self.node_id = -2
        self.node_list = {}

        self.aff_m = aff_m
        self.knn_m = knn_m
        self.G = nx.from_numpy_matrix(knn_m)

        self.knn_graph_stats = self.graph_stats_init(knn_m)
        self.aff_graph_stats = self.graph_stats_init(aff_m)

    def graph_stats_init(self, sym_m):
        M = sym_m
        np.fill_diagonal(M, 0)
        d = np.sum(M, 1) - M.diagonal()
        if np.any(d == 0):
            M += 1e-3
            np.fill_diagonal(M, 0)
            d = np.sum(M, 1) - M.diagonal()
        log_d = np.log2(d)
        d_log_d = np.multiply(d, log_d)

        sparce_m = sparse.csr_matrix(sym_m)
        m = sparce_m.sum() / 2
        vG = sparce_m.sum()
        log_vG = log2(vG)

        graph_stats = M, m, d, log_d, d_log_d, vG, log_vG, sparce_m   # ???
        return graph_stats

    def update_node_id(self, increment=True):
        if increment:
            self.node_id += 1
        else:
            self.node_id -= 1
        return self.node_id

    def get_current_node_id(self):
        return self.node_id

    def build_tree(self):
        M, m, d, log_d, d_log_d, vG, log_vG, sparce_m = self.knn_graph_stats
        root = Node(self.knn_graph_stats, self.update_node_id(), [], list(range(self.vertex_num)), is_leaf=False)
        self.node_list[root.id] = root
        root.V = vG

        if self.strategy == 'bottom_up':
            Z = self.bottom_up(root)
            root = self.node_list[self.node_id]
        else:
            Z = self.top_down(root)
            root = self.node_list[2*self.vertex_num - 2]
        self.root = root
        self.Z = Z[:, :4]
        return Z

    def bottom_up(self, root):
        for i in range(self.vertex_num):
            node = Node(self.knn_graph_stats, self.update_node_id(), [], [i], parent=root)
            self.node_list[node.id] = node
            root.children.append(node)
            root.vs.append(i)
        Z = self.linkage(root)
        return Z

    def top_down(self, root):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.knn_graph_stats

        N = self.vertex_num
        root.vs = np.array(range(N))
        Z = np.zeros((N - 1, 5))

        self.leaves = []
        self.beyond_leaves = []
        nodes_to_divide = Queue(maxsize=N)
        nodes_to_divide.put(root)
        while not nodes_to_divide.empty():
            node = nodes_to_divide.get()
            self.dividing_tree(node, nodes_to_divide, Z)

        height = max(Z[:, 2])
        Z[np.argwhere(Z[:, 2] == -1), 2] = height + 1
        Z[:, 2] = height + 2 - Z[:, 2]
        return Z

    def _get_dividing_delta(self, node, children):
        if len(children) < 2:
            return np.nan
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.knn_graph_stats
        left, right = children
        if self.objective == 'M':
            delta = (children[0].s/vG - np.power(children[0].V/vG, 2) + children[1].s/vG - np.power(children[1].V/vG, 2))
        else:
            if self.strategy == 'bottom_up':
                left.se = left.g/vG*log2(node.V/left.V)
                right.se = right.g/vG*log2(node.V/right.V)
            delta = (node.V_log_V - node.d_log_d)/vG \
                - (left.se + (left.V_log_V - left.d_log_d)/vG) \
                - (right.se + (right.V_log_V - right.d_log_d)/vG)
        return delta

    def dividing_tree(self, node, nodes_to_divide, Z):

        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.knn_graph_stats
        if len(node.vs) > 2:
            A = M[np.ix_(node.vs, node.vs)]
            L = csgraph.laplacian(A, normed=True)
            eig_values, eig_vectors = la.eigh(L)

            unique_eig_values = np.sort(list(set(eig_values.real)))
            if len(unique_eig_values) == 1:
                mid = int(len(node.vs)/2)
                node1_vs = range(mid)
                node2_vs = range(mid, len(node.vs))
            else:
                fiedler_pos = np.where(eig_values.real == unique_eig_values[1])[0][0]
                fiedler_vector = np.transpose(eig_vectors)[fiedler_pos]
                if len(fiedler_vector) == self.vertex_num:
                    self.root_fiedler_vector = fiedler_vector
                clusters, centroids = kmeans1d.cluster(fiedler_vector, 2)
                node1_vs = np.argwhere(np.array(clusters) == 0).T[0]
                node2_vs = np.argwhere(np.array(clusters) == 1).T[0]

                if len(fiedler_vector) != self.vertex_num:
                    node1_mean = np.mean(self.root_fiedler_vector[node1_vs])
                    node2_mean = np.mean(self.root_fiedler_vector[node2_vs])
                    if node1_mean > node2_mean:
                        tmp = node1_vs
                        node1_vs = node2_vs
                        node2_vs = tmp

        else:  # 2 vs
            Z[node.id-(self.vertex_num)] = [node.vs[0], node.vs[1], -1, 2, node.id]
            node.is_leaf = True
            self.leaves.append(node.id)
            return

        children_vs = [node1_vs, node2_vs]

        children = []
        child_id_q = Queue(maxsize=2)
        for child_vs in children_vs:
            child_vs = node.vs[child_vs]
            if len(child_vs) == 1:
                child_id = child_vs[0]
            else:
                child_id = self.update_node_id(increment=False)
                child_id_q.put(child_id)
            child = Node(self.knn_graph_stats, child_id, [], child_vs, parent=node, is_leaf=False)
            children.append(child)

        delta = self._get_dividing_delta(node, children)
        node.split_se = delta
        if delta < self.split_se_cutoff:  # not split
            node.is_leaf = True
            self.leaves.append(node.id)
            parent_id = node.id
            parent_vertex_num = len(node.vs)
            i = 0
            for v in node.vs:
                if parent_id < self.vertex_num:
                    continue
                if not child_id_q.empty():
                    child_id = child_id_q.get()
                elif i == len(node.vs)-2:
                    child_id = node.vs[-1]
                else:
                    child_id = self.update_node_id(increment=False)

                Z[parent_id-(self.vertex_num)] = [v, child_id, -1, parent_vertex_num, parent_id]
                parent_id = child_id
                parent_vertex_num -= 1
                i += 1
            return

        for child in children:
            child.height = node.height + 1
            self.node_list[child.id] = child
            node.children.append(child)

            if len(child.vs) > 1:
                nodes_to_divide.put(child)
            else:
                child.is_leaf = True

        if (node.parent and self.node_list[node.parent.id].is_leaf) or len(node.vs) in [1, 2]:
            node.is_leaf = True

        if node.is_leaf:
            for child in children:
                child.height = -1
                child.leaf = True
        else:
            self.beyond_leaves.append(node.id)

        for n in [node] + children:
            if n.is_leaf and n.parent is not None and not self.node_list[n.parent.id].is_leaf:
                self.leaves.append(n.id)

        Z[node.id-(self.vertex_num)] = [children[0].id, children[1].id, children[0].height, len(node.vs), node.id]

    def get_max_delta_from_heap(self, heap, row_ids, col_ids):
        while heap:
            max_delta, max_n1, max_n2 = heapq.heappop(heap)
            if max_n1 not in row_ids or max_n2 not in col_ids:
                continue
            max_n1 = self.node_list[max_n1]
            max_n2 = self.node_list[max_n2]
            max_delta = -max_delta
            return max_n1, max_n2, max_delta
        else:
            return None, None, None

    def linkage(self, root):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.knn_graph_stats

        N = self.vertex_num
        Z = np.zeros((N - 1, 5))
        leaves = {n: 1 for n in range(N)}
        individuals = {n: 1 for n in range(N)}

        heap = []
        heapq.heapify(heap)

        i = 0
        # for n1, n2 in zip(*np.triu(self.knn_m, 1).nonzero()):  # O(kn)
        for n1, n2, _ in self.G.edges(data=True):
            if n1 == n2:
                continue
            node1 = self.node_list[n1]
            node2 = self.node_list[n2]
            if self.objective == 'SE':
                delta = graph_metric.get_delta_merge_se(M, sparse_m, vG, d, root, node1, node2)
            else:  # network modularity
                delta = graph_metric.get_delta_nm(M, sparse_m, vG, root, node1, node2)
            heapq.heappush(heap, (-delta, n1, n2))
            i += 1
        if self.verbose:
            print('i', i, '(linkage - initial non zero pair)')

        z_i = 0
        count = 0
        for only_positive_delta in [True, False]:  # set false to merge some sub part in club
            if self.objective == 'M':
                only_positive_delta = False
            while individuals:
                if not self.G.edges(data=True):
                    break
                if self.verbose:
                    print('-----', self.objective, 'merge phase', count, only_positive_delta, 'start, individuals: ', len(individuals), ', leaves ', len(leaves))
                    print(len(self.G.edges(data=True)))
                z_i, merge_phase_i = self._merge_phase(root, individuals, leaves, Z, z_i, heap,
                                                       only_positive_delta=only_positive_delta)
                if self.verbose:
                    print('-----', self.objective, 'merge phase', count, 'end, individuals', len(individuals), ', leaves ', len(leaves))
                i += merge_phase_i
                if self.verbose:
                    print('i', i, 'merge phase')
                count += 1
                if len(individuals) > 0:
                    break
                for l in leaves:
                    self.node_list[l].is_individual = True
                    individuals[l] = 1
                    i += 1
                if self.verbose:
                    print('i', i, 'merge phase update leaves states')

        self.leaves = [l for l in leaves]
        while z_i < self.vertex_num - 1:
            if self.verbose:
                print('-----', self.objective, 'binary merge', count)
            z_i, binary_combine_i = self._binary_combine(root, leaves, Z, z_i,
                                                         only_positive_delta=False)
            i += binary_combine_i
            if self.verbose:
                print('i', i, 'binary_combine')
            count += 1

        if self.verbose:
            print('N', N, 'i', i)
        return Z

    def _merge_phase(self, root, individuals, leaves, Z, z_i, heap,
                     only_positive_delta=True, is_leaf=True):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.knn_graph_stats
        i = 0
        while individuals:
            if not self.G.edges(data=True):
                return z_i, i

            max_n1, max_n2, max_delta = self.get_max_delta_from_heap(heap, individuals, leaves)

            if max_n1 is None:
                return z_i, i
            new_node = root.merge(self.update_node_id(), max_n1, max_n2, is_leaf=is_leaf)
            self.node_list[new_node.id] = new_node

            Z[z_i] = [max_n1.id, max_n2.id, new_node.dist, len(new_node.vs), new_node.id]

            # update
            del individuals[max_n1.id]
            max_n1.is_individual = False
            if max_n2.is_individual:
                del individuals[max_n2.id]
                max_n2.is_individual = False
            if max_n1.is_leaf:
                del leaves[max_n1.id]
                max_n1.is_leaf = False
            del leaves[max_n2.id]
            max_n2.is_leaf = False

            new_node.is_leaf = True
            new_node.is_individual = False

            self.G.add_node(new_node.id)
            for x in set(chain(self.G.neighbors(max_n1.id), self.G.neighbors(max_n2.id))):
                if x == max_n1.id or x == max_n2.id:
                    continue
                i += 1
                node = self.node_list[x]
                if self.objective == 'SE':
                    delta = graph_metric.get_delta_merge_se(M, sparse_m, vG, d, root, node, new_node)
                    # delta = se.get_delta_combine_se(M, sparse_m, vG, root, node, new_node)
                else:
                    delta = graph_metric.get_delta_nm(M, sparse_m, vG, root, node, new_node)
                if only_positive_delta and delta < 0:
                    continue
                heapq.heappush(heap, (-delta, x, new_node.id))
                self.G.add_edge(x, new_node.id, weight=1)

            leaves[new_node.id] = 1
            self.G.remove_node(max_n1.id)
            self.G.remove_node(max_n2.id)

            z_i += 1
        return z_i, i

    def _binary_combine(self, root, leaves, Z, z_i, only_positive_delta=False, by='heap'):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.knn_graph_stats
        heap = []
        heapq.heapify(heap)
        i = 0
        ns = [(n1, n2) for n1, n2, _ in self.G.edges(data=True) if n1 != n2]
        if not ns:
            ns = itertools.combinations(leaves, 2)
        for n1, n2 in ns:
            node1, node2 = self.node_list[n1], self.node_list[n2]
            if self.objective == 'SE':
                delta = graph_metric.get_delta_combine_se(M, sparse_m, vG, root, node1, node2)
            else:
                delta = graph_metric.get_delta_nm(M, sparse_m, vG, root, node1, node2)
            if only_positive_delta and delta < 0:
                continue
            heapq.heappush(heap, (-delta, n1, n2))
            i += 1

        while z_i < self.vertex_num - 1:
            max_n1, max_n2, max_delta = self.get_max_delta_from_heap(heap, leaves, leaves)

            if max_n1 is None:
                return z_i, i

            new_node = root.merge(self.update_node_id(), max_n1, max_n2, is_leaf=False)
            self.node_list[new_node.id] = new_node

            Z[z_i] = [max_n1.id, max_n2.id, new_node.dist, len(new_node.vs), new_node.id]

            # update
            del leaves[max_n1.id]
            del leaves[max_n2.id]
            self.G.add_node(new_node.id)  # O(k)
            xs = set(chain(self.G.neighbors(max_n1.id), self.G.neighbors(max_n2.id)))
            # xs = []  # not solving HC problem
            if not xs:
                xs = leaves
            for x in xs:
                node = self.node_list[x]
                if self.objective == 'SE':
                    delta = graph_metric.get_delta_combine_se(M, sparse_m, vG, root, node, new_node)
                else:
                    delta = graph_metric.get_delta_nm(M, sparse_m, vG, root, node, new_node)
                if only_positive_delta and delta < 0:
                    continue
                heapq.heappush(heap, (-delta, x, new_node.id))
                i += 1
            self.G.remove_node(max_n1.id)
            self.G.remove_node(max_n2.id)
            leaves[new_node.id] = 1

            z_i += 1

        return z_i, i

    def order_tree(self):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.knn_graph_stats
        node = self.root
        A = M[np.ix_(node.vs, node.vs)]
        L = csgraph.laplacian(A, normed=True)
        eig_values, eig_vectors = la.eigh(L)

        unique_eig_values = np.sort(list(set(eig_values.real)))
        fiedler_pos = np.where(eig_values.real == unique_eig_values[1])[0][0]
        fiedler_vector = np.transpose(eig_vectors)[fiedler_pos]
        self._order_tree_aux(node, fiedler_vector)

    def _order_tree_aux(self, node, vector):
        if len(node.children) == 0:
            return
        left_v = np.median(vector[node.children[0].vs])
        right_v = np.median(vector[node.children[1].vs])
        if left_v > right_v:  # swith
            node.chidlren = node.children[::-1]
            row = self.Z[node.id - self.vertex_num]
            self.Z[node.id - self.vertex_num] = [row[1], row[0], row[2], row[3]]

        for child in node.children:
            self._order_tree_aux(child, vector)

    def contract_tree(self, Z, n_clusters):
        # update node distance
        se_scores, ks_subpopulation_node_ids, ks_clusters, optimal_k = self._contract_tree_dp(self.root)
        tmp = pd.DataFrame(np.matrix(se_scores), columns=self.ks).T
        tmp['K'] = tmp.index
        tmp.columns = ['SE Score', 'K']
        self.se_scores = tmp
        delta_se_scores = se_scores[1:] - se_scores[:-1]
        tmp = pd.DataFrame(np.matrix(delta_se_scores), columns=self.ks[1:]).T
        tmp['K'] = tmp.index
        tmp.columns = ['Delta SE Score', 'K']
        self.delta_se_scores = tmp
        self.ks_clusters = pd.DataFrame(np.matrix(ks_clusters).T, columns=['K={}'.format(k) for k in self.ks])
        Z_clusters = hierarchy.cut_tree(Z[:, :4], n_clusters=n_clusters)
        self.Z_clusters = pd.DataFrame(np.matrix(Z_clusters), columns=['K={}'.format(k) for k in n_clusters])

        self.optimal_k = optimal_k
        self.optimal_clusters = self.ks_clusters['K={}'.format(self.optimal_k)].tolist()
        self.optimal_subpopulation_node_ids = ks_subpopulation_node_ids[optimal_k-1]
        return

    def _contract_tree_dp(self, root):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.knn_graph_stats
        if self.strategy == 'bottom_up':
            node_ids = self.leaves + list(range(self.leaves[-1]+1, self.vertex_num*2 - 1))
        else:
            node_ids = self.leaves[::-1] + self.beyond_leaves[::-1]
        # print(node_ids)  # the nodes ordered from leaf to parent

        cost_m = np.zeros((len(node_ids), self.max_k+1))
        cost_m.fill(np.nan)
        cutoff_m = np.zeros((len(node_ids), self.max_k+1))
        cutoff_m.fill(-1)

        self._dp_compute_cost(cost_m, cutoff_m, node_ids)
        if self.verbose:
            print('se cost')
            print(cost_m[-1, :])
        # print(cost_m)
        # print(cost_m.shape)
        # print(cutoff_m)
        ks_clusters = []
        ks_subpopulation_node_ids = []
        for k in self.ks:
            if k == 1:
                ks_clusters.append([0]*self.vertex_num)
                ks_subpopulation_node_ids.append([self.root.id])
                continue
            subpopulation_node_ids = []
            self._trace_back(root, cost_m, cutoff_m, subpopulation_node_ids, k)
            clusters = [(v, i) for i, c in enumerate(subpopulation_node_ids) for v in self.node_list[c].vs]
            clusters = sorted(clusters)
            clusters = [c for v, c in clusters]
            if len(clusters) != self.vertex_num:  # happens in bottom up node if k larger than number of leaves
                ks_clusters.append([0]*self.vertex_num)
                ks_subpopulation_node_ids.append([])
                continue

            ks_clusters.append(clusters)
            ks_subpopulation_node_ids.append(subpopulation_node_ids)

        optimal_k = self.max_k - np.argmin(cost_m[-1, 2:][::-1])
        if self.verbose:
            print(cost_m[-1, :][optimal_k])
        return cost_m[-1, 1:], ks_subpopulation_node_ids, ks_clusters, optimal_k

    def _dp_compute_cost(self, cost_m, cutoff_m, node_ids, contract_type='2D'):

        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.knn_graph_stats
        aff_M, aff_m, aff_d, aff_log_d, aff_d_log_d, aff_vG, aff_log_vG, _ = self.aff_graph_stats

        for dp_i, node_id in enumerate(node_ids):
            node = self.node_list[node_id]
            node.dp_i = dp_i

            for k in self.ks:

                node_g = graph_metric.get_g(aff_M, sparse_m, node.vs)   # sparse_m is not used
                node_V = graph_metric.get_v(aff_M, sparse_m, node.vs)
                if node.parent and contract_type == 'high-dimensional':
                    parent_V = graph_metric.get_v(aff_M, sparse_m, node.parent.vs)
                else:  # root node or 2D hirerachy
                    parent_V = aff_vG
                # node_se = node_g/aff_vG*log2(parent_V/node_V)
                node_se = graph_metric.get_node_score(self.objective, aff_vG, node_g, node_V, parent_V, eta=1)

                if k == 1:
                    node_d_log_d = np.sum(aff_d_log_d[node.vs])
                    cost_m[node.dp_i, k] = node_se + \
                        graph_metric.get_node_vertices_score(self.objective, aff_vG, node_d_log_d, node_V)
                    # - (node_d_log_d - node_V*log2(node_V))/aff_vG
                    # print(node.id, k, cost_m[node.dp_i, k], len(node.vs))
                    continue

                if len(node.vs) < k or not node.children:
                    cost_m[node.dp_i, k] = np.inf
                    continue

                l_id = node.children[0].dp_i
                r_id = node.children[1].dp_i
                min_i = None
                min_cost = np.inf
                for i in range(1, k):
                    cost = cost_m[l_id, i] + cost_m[r_id, k-i]
                    if contract_type == 'high_dimensional':
                        cost += node_se
                    if cost < min_cost:
                        min_cost = cost
                        min_i = i
                cost_m[node.dp_i, k] = min_cost
                cutoff_m[node.dp_i, k] = min_i
                # print(node.id, k, cost_m[node.dp_i, k], len(node.vs))

    def _trace_back(self, node, cost_m, cutoff_m, clusters, k_hat):
        k_prime = cutoff_m[node.dp_i, k_hat]
        if np.isnan(k_prime) or k_prime == -1:
            return
        k_prime = int(k_prime)
        left_node = node.children[0]
        right_node = node.children[-1]

        if k_prime > 1:
            self._trace_back(left_node, cost_m, cutoff_m, clusters, k_prime)
        else:
            clusters.append(left_node.id)
        if k_prime < k_hat-1:
            self._trace_back(right_node, cost_m, cutoff_m, clusters, k_hat-k_prime)
        else:
            clusters.append(right_node.id)

    def get_tree_se(self):
        return 0
        tree_se = 0
        self.get_tree_se_aux(self.root, tree_se)
        if self.verbose:
            print(tree_se)

    def get_tree_se_aux(self, node, tree_se):
        tree_se += node.se
        if len(node.children) != 0:
            tree_se += node.se

    def to_newick(self):
        return '({});'.format(self._to_newick_aux(self.root, is_root=True))

    def _to_newick_aux(self, node, is_root=False):
        if len(node.vs) == 1:
            return 'n{}:{}'.format(node.id, 1)

        if node.is_leaf:
            if self.strategy == 'bottom_up':
                res = self._to_newick_leaf_bottom_up(node)
            else:
                res = self._to_newick_leaf_top_down(node)
        else:
            res = ','.join([self._to_newick_aux(c) for c in node.children])

        return '({})n{}:{}'.format(res, node.id, 1)

    def _to_newick_leaf_bottom_up(self, node):
        if len(node.vs) == 1:
            return 'n{}'.format(node.id, 1)
        else:
            return ','.join([self._to_newick_leaf_bottom_up(self.node_list[v]) for v in node.vs])

    def _to_newick_leaf_top_down(self, node):
        try:
            return ','.join([self._to_newick_leaf_top_down(v) for v in node.vs])
        except Exception:
            return 'n{}:{}'.format(node, 1)

    def to_split_se(self):
        split_dict_list = []
        self._to_split_se_aux(self.root, split_dict_list)
        df = pd.DataFrame(split_dict_list)
        return df

    def _to_split_se_aux(self, node, split_dict_list, subpopulation=False, club=False):
        if node.id in self.optimal_subpopulation_node_ids:
            subpopulation = True
        if node.id in self.leaves:
            club = True
        if self.strategy == 'bottom_up':
            split_se = self._get_dividing_delta(node, node.children)
        else:
            split_se = node.split_se
        split_dict_list.append({
            'node_id': node.id,
            'split_se': split_se,
            'vertex_num': len(node.vs),
            'subpopulation': subpopulation,
            'club': club,
        })
        for child in node.children:
            self._to_split_se_aux(child, split_dict_list, subpopulation, club)


class SEAT(AgglomerativeClustering):

    def __init__(self, min_k=1, max_k=10,
                 a=None,
                 affinity='precomputed',
                 sparsification='knn_neighbors',
                 knn_m=None,
                 strategy='top_down',
                 objective='SE',
                 n_neighbors=10,
                 dist_topk=5,
                 split_se_cutoff=0.05,
                 kernel_gamma=None,
                 outlier_detection=None,
                 outlier_percentile=None,
                 outlier_distance=0.5,
                 verbose=False,
                 ):
        self.min_k = min_k
        self.max_k = max_k
        self.ks = range(min_k, max_k+1)
        self.a = a
        self.affinity = affinity
        self.sparsification = sparsification
        self.strategy = strategy
        self.objective = objective
        self.n_neighbors = n_neighbors
        self.dist_topk = dist_topk
        self.split_se_cutoff = split_se_cutoff

        self.kernel_gamma = kernel_gamma

        self.knn_m = knn_m

        self.outlier_detection = outlier_detection
        self.outlier_percentile = outlier_percentile
        self.outlier_distance = outlier_distance
        self.verbose = verbose

    def construct_affinity(self, X):
        # https://scikit-learn.org/stable/modules/metrics.html

        if self.affinity == 'precomputed':
            aff_m = X

        elif self.affinity == 'cosine_similarity':
            aff_m = pairwise_kernels(X, metric='cosine')

        elif self.affinity == 'linear_kernel':
            aff_m = pairwise_kernels(X, metric='linear')

        elif self.affinity == 'gaussian_kernel':
            if self.kernel_gamma:
                sigma = self.kernel_gamma
            else:
                sigma = X.std()
            if self.verbose:
                print('sigma', sigma)
            gamma = 1/(sigma*sigma)
            aff_m = pairwise_kernels(X, metric='rbf', gamma=gamma)

        elif self.affinity == 'gaussian_kernel_topk':
            if self.kernel_gamma:
                sigma = self.kernel_gamma
            else:
                sigma = X.std()
            if self.dist_topk > X.shape[1]:
                dist_topk = X.shape[1]
            else:
                dist_topk = self.dist_topk
            n = X.shape[0]
            aff_m = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    dist = (X[i] - X[j])**2
                    dist.sort()
                    dist = dist[-dist_topk:]
                    v = np.exp(-np.sum(dist)/sigma/sigma)
                    aff_m[i][j] = v
                    aff_m[j][i] = v

        elif self.affinity == 'laplacian_kernel':
            if self.kernel_gamma:
                gamma = self.kernel_gamma
            else:
                gamma = 0.1
            aff_m = pairwise_kernels(X, metric='laplacian', gamma=gamma)

        elif self.affinity == 'knn_neighbors_from_X':
            aff_m = kneighbors_graph(X, self.n_neighbors).toarray()
            aff_m = (aff_m + aff_m.T)/2
            aff_m[np.nonzero(aff_m)] = 1

        if (aff_m < 0).any():
            aff_m = aff_m + np.abs(np.min(aff_m))

        self.aff_m = aff_m

    def graph_sparsification(self, X):
        knn_m = None
        if self.sparsification == 'affinity':
            knn_m = copy.deepcopy(self.aff_m)
        elif self.sparsification == 'precomputed':
            knn_m = self.knn_m
        if self.sparsification == 'knn_neighbors':
            k = self.n_neighbors
            knn_m = np.zeros((X.shape[0], X.shape[0]))
            for i in range(X.shape[0]):
                ids = np.argpartition(self.aff_m[i], -k)[-k:]
                top_set = set(self.aff_m[i, ids])
                if len(top_set) == 1:
                    b = self.aff_m[i] == top_set.pop()
                    ids = []
                    offset = 1
                    left = True
                    while len(ids) < k:
                        if left:
                            idx = i + offset
                        else:
                            idx = i - offset
                        if idx < 0 or idx > len(b)-1:
                            offset += 1
                            left = not left
                            continue
                        if b[idx]:
                            ids.append(idx)
                        offset += 1
                        left = not left
                knn_m[i, ids] = 1

            knn_m = (knn_m + knn_m.T)/2
            knn_m[np.nonzero(knn_m)] = 1

        if self.sparsification == 'knn_neighbors_from_X':
            knn_m = kneighbors_graph(X, self.n_neighbors).toarray()
            knn_m = (knn_m + knn_m.T)/2
            knn_m[np.nonzero(knn_m)] = 1

        self.knn_m = knn_m

    def detect_outlier(self, X):
        
        if self.outlier_detection == 'knn_neighbors':
            knn = NearestNeighbors(n_neighbors=self.n_neighbors)
            knn.fit(X)
            distances, indexes = knn.kneighbors(X)
            distances_mean = distances.mean(axis=1)
            if self.outlier_percentile:
                cutoff = np.quantile(distances_mean, self.outlier_percentile)
            else:
                cutoff = self.outlier_distance
            self.outlier_index = np.where(distances_mean > cutoff)[0]
            self.inlier_index = np.where(distances_mean <= cutoff)[0]
            return X[self.inlier_index, :]
        else:
            self.outlier_index = []
            return X

    def insert_outliers(self, labels):
        #print(len(self.outlier_index), len(self.inlier_index))
        if len(self.outlier_index) == 0:
            return labels
        
        n = len(self.outlier_index) + len(self.inlier_index)
        new_labels = []
        inliner_i = 0
        for i in range(n):
            if i in self.outlier_index:
                new_labels.append(-1)
            else:
                new_labels.append(labels[inliner_i])
                inliner_i += 1
        return new_labels

    def insert_outliers_df(self, label_df):
        #print(len(self.outlier_index), len(self.inlier_index))
        if len(self.outlier_index) == 0:
            return label_df
        label_m = label_df.values.tolist()
       
        n = len(self.outlier_index) + len(self.inlier_index)
        new_label_m = []
        inliner_i = 0
        for i in range(n):
            if i in self.outlier_index:
                new_label_m.append([-1]*label_df.shape[1])
            else:
                new_label_m.append(label_m[inliner_i])
                inliner_i += 1
        new_label_df = pd.DataFrame(new_label_m, columns = label_df.columns)
        return new_label_df
        
    def fit(self, X, y=None):

        X = self._validate_data(X, ensure_min_samples=2, estimator=self)

        if self.min_k is not None and self.min_k <= 0:
            raise ValueError("min_k should be an integer greater than 0."
                             " %s was provided." % str(self.min_k))

        if self.max_k is not None and self.max_k <= 2:
            raise ValueError("max_k should be an integer greater than 2."
                             " %s was provided." % str(self.max_k))

        if self.affinity not in ['precomputed', 'gaussian_kernel', 'gaussian_kernel_topk', 'linear_kernel', 'cosine_similarity', 'knn_neighbors_from_X', 'laplacian_kernel']:
            raise ValueError("affinity should be precomputed, gaussian_kernel, linear_kernel, cosine_similarity, knn_neighbors_from_X, "
                             "laplacian_kernel. "
                             "%s was provided." % str(self.affinity))

        if self.sparsification not in ['affinity', 'precomputed', 'knn_neighbors', 'knn_neighbors_from_X']:
            raise ValueError("sparsification should be affinity, precomputed, knn_neighbors, knn_neighbors_from_X."
                             " %s was provided." % str(self.sparsification))

        if self.outlier_detection not in [None, 'knn_neighbors']:
            raise ValueError("outlier_detection should be None, precomputed."
                             " %s was provided." % str(self.outlier_detection))            
            
        if self.strategy not in ['bottom_up', 'top_down']:
            raise ValueError("affinity should be bottom_up, top_down."
                             " %s was provided." % str(self.strategy))
            
        if self.verbose:
            print('fit', self.strategy)
            
        X = self.detect_outlier(X)
        self.construct_affinity(X)
        self.graph_sparsification(X)
        

        # build the tree

        setree_class = pySETree
        # setree_class = seat_wrapper.SETree

        se_tree = setree_class(self.aff_m, self.knn_m,
                               self.min_k, self.max_k,
                               self.objective,
                               self.strategy,
                               self.split_se_cutoff,
                               verbose=self.verbose)
        self.se_tree = se_tree
        t1 = time.time()
        Z = se_tree.build_tree()
        t2 = time.time()
        if self.verbose:
            print('build tree time', t2 - t1)
        self.aff_m = se_tree.aff_m

        # se_tree.order_tree()
        self.tree_se = se_tree.get_tree_se()

        se_tree.contract_tree(Z, self.ks)
        self.vertex_num = se_tree.vertex_num
        self.ks = list(se_tree.ks)
        self.se_scores = se_tree.se_scores
        self.delta_se_scores = se_tree.delta_se_scores
        self.optimal_k = se_tree.optimal_k
        
        self.labels_ = self.insert_outliers(se_tree.optimal_clusters)
        self.Z_ = Z[:, :4]
        self.leaves_list = hierarchy.leaves_list(self.Z_)
        self.order = self.insert_outliers(self._order())
        self.ks_clusters = self.insert_outliers_df(se_tree.ks_clusters)
        self.Z_clusters = self.insert_outliers_df(se_tree.Z_clusters)
        self.clubs = self.insert_outliers(self._get_clubs())
        self.club_k = len(se_tree.leaves)

        self.newick = se_tree.to_newick()

        self.split_se = se_tree.to_split_se()

        return self

    def _order(self):
        # hierarchy.leaves_list(self.Z_)
        order = [(l, i) for i, l in enumerate(self.leaves_list)]
        order.sort()
        return [i for l, i in order]

    def _get_clubs(self):
        leaves = sorted([(self.order[self.se_tree.node_list[l].vs[0]], l) for l in self.se_tree.leaves])
        order = [(v, i) for i, l in enumerate(leaves) for v in self.se_tree.node_list[l[1]].vs]
        order.sort()
        return [i for n, i in order]

    def oval_embedding(self, a=3, b=2, k=0.2):
        # http://www.mathematische-basteleien.de/eggcurves.htm
        angle = np.array([self.order])*(2*np.pi/len(self.order))
        xcor = a*np.cos(angle)
        ycor = b*np.sin(angle)*1/np.sqrt(np.exp(k*np.cos(angle)))
        plane_coordinate = np.concatenate((xcor, ycor), axis=0).T
        return plane_coordinate
