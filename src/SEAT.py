import numpy as np
from math import log2
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
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


from . import se


class Node():
    def __init__(self, graph_stats, node_id, parent=None, leaf=False,
                 is_singleton=True,
                 is_leaf=True):
        self.id = node_id
        self.parent = parent
        self.leaf = leaf
        self.is_singleton = is_singleton
        self.is_leaf = is_leaf
        self.children = []
        self.left = 0
        self.right = 0
        self.g = 0.
        self.g_log_V = 0.
        self.V = 0.
        self.log_V = 0.
        self.V_log_V = 0.
        self.s = 0.
        self.se = 0.
        self.vs = []
        self.height = 0
        self.dist = 1
        self.graph_stats = graph_stats

    def reset(self, parent=None):
        self.init(parent)

    def init(self, parent=None, setVertices=True):
        if setVertices:
            self.setVertices()
        self.setV()
        self.setS()
        self.setG()
        if parent:
            self.setSE(parent)
        else:
            # root
            self.se = 0
        for c in self.children:
            if (isinstance(c, Node)):
                c.setSE(self)

    def setVertices(self):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.graph_stats
        vs = []
        if self.leaf:
            vs += self.children
        else:
            for c in self.children:
                if isinstance(c, Node):
                    vs += c.setVertices()
                elif isinstance(c, int):
                    vs.append(c)
                else:
                    raise TypeError('child can only be int or Node')
        self.vs = vs
        self.left = self.vs[0]
        self.right = self.vs[-1]
        return vs

    def setS(self):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.graph_stats
        self.s = se.get_s(M, sparse_m, self.vs)

    def setG(self):
        self.g = self.V - self.s
        self.g_log_V = self.g * self.log_V

    def setV(self):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.graph_stats
        self.V = se.get_v(M, sparse_m, self.vs)
        self.log_V = log2(self.V)
        self.V_log_V = self.V * self.log_V

    def setSE(self, parent):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.graph_stats
        self.se = se.get_se(vG, self.g, self.V, parent.V)

    def increase_height(self, increment=1):
        self.height += increment
        for c in self.children:
            if isinstance(c, Node):
                c.increase_height(increment)

    def merge(self, node_id, node1, node2, is_leaf=False):
        if (node1.parent != node2.parent):
            raise ValueError("parents are not the same")
        node = Node(self.graph_stats, node_id, parent=self.id)
        node.leaf = is_leaf
        node.height = node1.height
        node1.parent = node.id
        node1.increase_height(1)
        node2.parent = node.id
        node2.increase_height(1)
        if is_leaf:
            node.children = node1.children + node2.children
        else:
            node.children.append(node1)
            node.children.append(node2)
            node.dist = max(node1.dist, node2.dist) + 1
        node.reset(self)
        self.children.append(node)
        if not self.delChild(node1):
            raise("fail to delete child", node1)
        if not self.delChild(node2):
            raise("fail to delete child", node1)
        return node

    def delChild(self, node):
        idx = -1
        for i, c in enumerate(self.children):
            if c.id == node.id:
                idx = i
                break
        if idx >= 0:
            del self.children[idx]
            return True
        return False


class pySETree():

    def __init__(self, aff_m, min_k=2, max_k=10,
                 max_g_ratio=0,
                 objective='structure_entropy',
                 strategy='top_down'):
        self.max_g_ratio = max_g_ratio
        self.strategy = strategy
        self.objective = objective
        self.min_k = min_k
        self.max_k = max_k

        self.vertex_num = aff_m.shape[0]
        if self.max_k > self.vertex_num:
            self.max_k = self.vertex_num - 1

        self.ks = range(self.min_k, self.max_k+1)

        if strategy == 'top_down':
            self.node_id = 2*self.vertex_num - 3
        else:
            self.node_id = -2
        self.node_list = {}

        self.affinity_m = aff_m
        self.graph_stats = self.graph_stats_init(aff_m)

    def graph_stats_init(self, aff_m):
        M = aff_m
        np.fill_diagonal(M, 0)
        d = None
        if np.any(d == 0):
            M += 1e-3
            np.fill_diagonal(M, 0)
            d = None
        log_d = None
        d_log_d = None

        sparce_m = sparse.csr_matrix(aff_m)
        m = sparce_m.sum() / 2
        vG = sparce_m.sum()
        log_vG = log2(vG)

        graph_stats = M, m, d, log_d, d_log_d, vG, log_vG, sparce_m
        return graph_stats

    def get_node_id(self, increment=True):
        if increment:
            self.node_id += 1
        else:
            self.node_id -= 1
        return self.node_id

    def build_tree(self):
        M, m, d, log_d, d_log_d, vG, log_vG, sparce_m = self.graph_stats
        root = Node(self.graph_stats, self.get_node_id())
        self.node_list[root.id] = root
        root.V = 2.*m

        print(self.strategy)
        if self.strategy == 'bottom_up':
            Z = self.bottom_up(root)
        else:
            Z = self.top_down(root)

        return Z

    def bottom_up(self, root):
        for i in range(self.vertex_num):
            node = Node(self.graph_stats, self.get_node_id(), parent=root.id, leaf=True)
            self.node_list[node.id] = node
            node.height = 1
            node.children = [i]
            node.left = node.right = i
            node.init(root)
            root.children.append(node)
        root.reset()
        Z = self.linkage(root, tree_type='multinary', by='heap')
        return Z

    def top_down(self, root):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.graph_stats

        N = self.vertex_num
        root.vs = np.array(range(N))
        Z = np.zeros((N - 1, 5))

        self.leafs = []
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
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.graph_stats
        if self.objective == 'modularity':
            delta = (children[0].s/vG - np.power(children[0].V/vG, 2) + children[1].s/vG - np.power(children[1].V/vG, 2))
        else:
            if self.max_g_ratio:
                delta = 0
                for child in children:
                    g = max(child.g - self.vertex_num*(self.vertex_num-len(child.vs))*self.max_g_ratio, 0)
                    delta += se.get_se(vG, g, child.V, node.V)
            else:
                delta = children[0].se + children[1].se
        return delta

    def dividing_tree(self, node, nodes_to_divide, Z):

        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.graph_stats

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
                clusters, centroids = kmeans1d.cluster(fiedler_vector, 2)
                node1_vs = np.argwhere(np.array(clusters) == 0).T[0]
                node2_vs = np.argwhere(np.array(clusters) == 1).T[0]
        else:
            node1_vs = [0]
            node2_vs = [1]

        children_vs = [node1_vs, node2_vs]

        children = []
        for child_vs in children_vs:
            child_vs = node.vs[child_vs]
            if len(child_vs) == 1:
                child_id = child_vs[0]
            else:
                child_id = self.get_node_id(increment=False)
            child = Node(self.graph_stats, child_id, parent=node.id)
            child.vs = child_vs
            child.init(parent=node, setVertices=False)
            self.node_list[child.id] = child
            node.children.append(child)
            children.append(child)
            child.height = node.height + 1

            if len(child.vs) > 1:
                nodes_to_divide.put(child)
            else:
                child.leaf = True

        delta = self._get_dividing_delta(node, children)
        if delta > 0 and self.vertex_num != len(node.vs):  # not split
            node.leaf = True
        if (node.parent and self.node_list[node.parent].leaf) or len(node.vs) == 2:
            node.leaf = True

        if node.leaf:
            for child in children:
                child.height = -1
                child.leaf = True

        for n in [node] + children:
            if n.leaf and not self.node_list[n.parent].leaf:
                self.leafs.append(n)

        Z[node.id-(self.vertex_num)] = [children[0].id, children[1].id, children[0].height, len(node.vs), node.id]

    def get_max_delta_from_table(self, m, row_ids, col_ids):
        m = m[np.ix_(row_ids, col_ids)]
        max_i = np.argmax(m)
        row_i = int(max_i/m.shape[1])
        col_i = max_i % m.shape[1]
        max_delta = m[row_i, col_i]
        max_n1 = self.node_list[row_ids[row_i]]
        max_n2 = self.node_list[col_ids[col_i]]
        return max_n1, max_n2, max_delta

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

    def linkage(self, root, tree_type='multinary', by='heap'):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.graph_stats
        G = nx.from_numpy_matrix(M)

        N = self.vertex_num
        Z = np.zeros((N - 1, 5))
        delta_m = np.zeros((2*N - 1, 2*N - 1))
        delta_m.fill(-10000)
        leafs = {n: 1 for n in range(N)}
        singletons = {n: 1 for n in range(N)}

        heap = []
        heapq.heapify(heap)

        i = 0
        for n1, n2 in zip(*np.triu(M, 1).nonzero()):  # O(kn)
            node1 = self.node_list[n1]
            node2 = self.node_list[n2]
            if self.objective == 'structure_entropy':
                delta = se.get_delta_se_plus(M, sparse_m, vG, d, root, node1, node2)
            else:  # network modularity
                delta = se.get_delta_nm(M, sparse_m, vG, root, node1, node2)
            delta_m[n1, n2] = delta
            heapq.heappush(heap, (-delta, n1, n2))
            i += 1

        z_i = 0

        count = 0
        for only_positive_delta in [True, False]:
            if self.objective == 'modularity':
                only_positive_delta = False
            while singletons:
                if tree_type == 'binary' and count > 0:
                    break
                print('merge phase', self.objective, only_positive_delta, count)
                z_i, merge_phase_i = self._merge_phase(G, root, singletons, leafs, Z, z_i, delta_m, heap, count+1,
                                                       only_positive_delta=only_positive_delta, by=by)
                i += merge_phase_i
                print('i', i, 'merge phase')
                count += 1
                if singletons:
                    break
                singletons = {}
                for l in leafs:
                    self.node_list[l].is_singleton = True
                    singletons[l] = 1
                    i += 1
                print('i', i, 'merge phase update leafs states')
                if not G.edges(data=True):
                    break

        self.leafs = [self.node_list[l] for l in leafs]
        while z_i < self.vertex_num - 1:
            only_positive_delta = False
            print('binary merge', self.objective, only_positive_delta, count)
            z_i, binary_merge_i = self._binary_merge(G, root, leafs, Z, z_i, delta_m,
                                                     only_positive_delta=only_positive_delta,
                                                     by=by)
            i += binary_merge_i
            count += 1

        return Z

    def _merge_phase(self, G, root, singletons, leafs, Z, z_i, delta_m, heap, dist,
                     only_positive_delta=True, is_leaf=True, by='heap'):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.graph_stats
        i = 0
        while singletons:
            if not G.edges(data=True):
                return z_i, i

            if by == 'table':
                max_n1, max_n2, max_delta = self.get_max_delta_from_table(delta_m, singletons, leafs)
            else:
                max_n1, max_n2, max_delta = self.get_max_delta_from_heap(heap, singletons, leafs)

            if max_n1 is None:
                return z_i, i

            new_node = root.merge(self.get_node_id(), max_n1, max_n2, is_leaf=is_leaf)
            self.node_list[new_node.id] = new_node

            Z[z_i] = [max_n1.id, max_n2.id, dist, len(max_n1.vs) + len(max_n2.vs), new_node.id]

            # update
            del singletons[max_n1.id]
            max_n1.is_singleton = False
            if max_n2.is_singleton:
                del singletons[max_n2.id]
                max_n2.is_singleton = False
            if max_n1.is_leaf:
                del leafs[max_n1.id]
                max_n1.is_leaf = False
            del leafs[max_n2.id]
            max_n2.is_leaf = False

            new_node.is_leaf = True
            new_node.is_singleton = False

            G.add_node(new_node.id)
            for x in set(chain(G.neighbors(max_n1.id), G.neighbors(max_n2.id))):
                i += 1
                node = self.node_list[x]
                if self.objective == 'structure_entropy':
                    delta = se.get_delta_se_plus(M, sparse_m, vG, d, root, node, new_node)
                else:
                    delta = se.get_delta_nm(M, sparse_m, vG, root, node, new_node)
                if only_positive_delta and delta < 0:
                    continue
                delta_m[x, new_node.id] = delta
                heapq.heappush(heap, (-delta, x, new_node.id))
                G.add_edge(x, new_node.id, weight=1)

            leafs[new_node.id] = 1
            G.remove_node(max_n1.id)
            G.remove_node(max_n2.id)

            z_i += 1

        return z_i, i

    def _binary_merge(self, G, root, leafs, Z, z_i, delta_m, only_positive_delta=True, by='heap'):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.graph_stats
        heap = []
        heapq.heapify(heap)
        i = 0
        ns = [(n1, n2) for n1, n2, _ in G.edges(data=True)]
        if not ns:
            ns = itertools.combinations(leafs, 2)
        for n1, n2 in ns:
            node1, node2 = self.node_list[n1], self.node_list[n2]
            if self.objective == 'structure_entropy':
                delta = se.get_delta_se(M, sparse_m, vG, root, node1, node2)
            else:
                delta = se.get_delta_nm(M, sparse_m, vG, root, node1, node2)
            delta_m[n1, n2] = delta
            if only_positive_delta and delta < 0:
                continue
            heapq.heappush(heap, (-delta, n1, n2))
            i += 1

        while z_i < self.vertex_num - 1:
            if by == 'table':
                max_n1, max_n2, max_delta = self.get_max_delta_from_table(delta_m, leafs, leafs)
            else:
                max_n1, max_n2, max_delta = self.get_max_delta_from_heap(heap, leafs, leafs)

            if max_n1 is None:
                return z_i, i

            new_node = root.merge(self.get_node_id(), max_n1, max_n2, is_leaf=False)
            self.node_list[new_node.id] = new_node

            Z[z_i] = [max_n1.id, max_n2.id, new_node.dist, len(max_n1.vs) + len(max_n2.vs), new_node.id]

            # update
            del leafs[max_n1.id]
            del leafs[max_n2.id]
            G.add_node(new_node.id)  # O(k)
            xs = set(chain(G.neighbors(max_n1.id), G.neighbors(max_n2.id)))
            if not xs:
                xs = leafs
            for x in xs:
                node = self.node_list[x]
                if self.objective == 'structure_entropy':
                    delta = se.get_delta_se(M, sparse_m, vG, root, node, new_node)
                else:
                    delta = se.get_delta_nm(M, sparse_m, vG, root, node, new_node)
                if only_positive_delta and delta < 0:
                    continue
                delta_m[x, new_node.id] = delta
                heapq.heappush(heap, (-delta, x, new_node.id))
                i += 1
            G.remove_node(max_n1.id)
            G.remove_node(max_n2.id)
            leafs[new_node.id] = 1

            z_i += 1

        return z_i, i

    def cut_tree(self, Z, n_clusters):
        # update node distance
        if self.strategy == 'bottom_up':
            root = self.node_list[self.node_id]
        else:
            root = self.node_list[2*self.vertex_num - 2]
        self.root = root
        se_scores, ks_clusters, optimal_k = self._cut_tree_dp(root)
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
        return

    def _cut_tree_dp(self, root):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.graph_stats
        nodes = range(self.vertex_num*2 - 1)
        cost_m = np.zeros((len(nodes), 5))
        cutoff_m = np.zeros((len(nodes), 5))
        cutoff_m.fill(-1)
        np.set_printoptions(suppress=True)

        self._dp_compute_cost(cost_m, cutoff_m, nodes)
        ks_clusters = []
        for k in self.ks:
            if k == 1:
                ks_clusters.append([0]*self.vertex_num)
                continue
            clusters = []
            self._trace_back(root, cost_m, cutoff_m, nodes, clusters, k)
            clusters = [(v, i) for i, c in enumerate(clusters) for v in self.node_list[c].vs]
            clusters = sorted(clusters)
            clusters = [c for v, c in clusters]
            if len(clusters) != self.vertex_num:  # happens in bottom up node if k larger than number of leafs
                ks_clusters.append([0]*self.vertex_num)
                continue

            ks_clusters.append(clusters)

        optimal_k = self.max_k - np.argmin(cost_m[-1, 2:][::-1])
        return cost_m[-1, 1:], ks_clusters, optimal_k

    def _trace_back(self, node, cost_m, cutoff_m, nodes, clusters, k_hat):
        if type(node) == int:
            return
        if len(node.vs) == 1:
            return
        i = node.id

        if k_hat == 1:
            return

        k_prime = int(cutoff_m[i, k_hat])
        left_node = node.children[0]
        right_node = node.children[-1]

        if k_prime > 1:
            self._trace_back(left_node, cost_m, cutoff_m, nodes, clusters, k_prime)
        else:
            if type(left_node) == int:
                clusters.append(left_node)
            else:
                clusters.append(left_node.id)
        if k_prime < k_hat-1:
            self._trace_back(right_node, cost_m, cutoff_m, nodes, clusters, k_hat-k_prime)
        else:
            if type(right_node) == int:
                clusters.append(right_node)
            else:
                clusters.append(right_node.id)

    def _dp_compute_cost(self, cost_m, cutoff_m, nodes):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.graph_stats
        for n in nodes:
            node = self.node_list[n]
            if type(node) == int:
                cost_m[node, :] = 1000
                continue

            for k in self.ks:
                if (self.strategy == 'bottom_up' and k != 1 and node.leaf) or len(node.vs) < k:
                    cost_m[node.id, k] = 1000
                    continue

                if self.objective == 'structure_entropy':
                    if self.max_g_ratio:
                        g = max(node.g - self.vertex_num*(self.vertex_num-len(node.vs))*self.max_g_ratio, 0)
                        parent_V = vG if not node.parent else self.node_list[node.parent].V
                        node_cost = se.get_se(vG, g, node.V, parent_V)
                    else:
                        node_cost = node.se
                else:
                    node_cost = -(node.s/vG - np.power(node.V/vG, 2))
                if k == 1:
                    cost_m[node.id, k] = node_cost
                    continue

                if type(node.children[0]) == int:
                    l_id = node.children[0]
                else:
                    l_id = node.children[0].id
                if type(node.children[1]) == int:
                    r_id = node.children[1]
                else:
                    r_id = node.children[1].id
                min_i = None
                min_cost = 100000
                for i in range(1, k):
                    cost = cost_m[l_id, i] + cost_m[r_id, k-i] + node_cost
                    if cost < min_cost:
                        min_cost = cost
                        min_i = i

                cost_m[node.id, k] = min_cost
                cutoff_m[node.id, k] = min_i

    def _cut_tree_dp_recursive(self, parent, node, k, vG, cost_m, cutoff_m, nodes):
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m = self.graph_stats
        if type(node) == int:
            return 1000
        if len(node.vs) < k:
            return 1000
        if k == 1:
            if self.objective == 'structure_entropy':
                cost = node.se
            else:
                cost = -(node.s/vG - np.power(node.V/vG, 2))
            return cost

        min_cost = 100000000
        min_i = None
        for i in range(1, k):
            l_cost = self._cut_tree_dp_recursive(node, node.children[0], i, vG, cost_m, cutoff_m, nodes)
            r_cost = self._cut_tree_dp_recursive(node, node.children[1], k-i, vG, cost_m, cutoff_m, nodes)
            cost = l_cost + r_cost + node.se
            if cost < min_cost:
                min_cost = cost
                min_i = i
        cost_m[node.id, k] = min_cost
        cutoff_m[node.id, k] = min_i
        return min_cost

    def _update_dist_to_level(self, parent, Z):
        for c in parent.children:
            c.reset(parent)
            if c.leaf:
                c.reset(parent)
                continue
            c.dist = parent.dist - 1
            if c.id > self.vertex_num:
                Z[c.id-self.vertex_num, 2] = c.dist
            self._update_dist_to_level(c, Z)

    def to_newick(self):
        return '({});'.format(self._to_newick_aux(self.root, is_root=True))

    def _to_newick_aux(self, node, is_root=False):
        if type(node) == int or type(node) == np.int32:
            return 'n{}:{}'.format(node, 1)
        if len(node.vs) == 1:
            return 'n{}:{}'.format(node.id, 1)

        if node.leaf:
            if self.strategy == 'bottom_up':
                res = self._to_newick_leaf_bottom_up(node)
            else:
                res = self._to_newick_leaf_top_down(node)
        else:
            res = ','.join([self._to_newick_aux(c) for c in node.children])
        if is_root:
            res = '({})n{}:{}'.format(res, node.id, 1)
        else:
            res = '({})n{}:{}'.format(res, node.id, 1)
        return res

    def _to_newick_leaf_bottom_up(self, node):
        if type(node) == int or type(node) == np.int32:
            return 'n{}'.format(node, 1)
        if len(node.vs) == 1:
            return 'n{}'.format(node.id, 1)

        return ','.join([self._to_newick_leaf_bottom_up(self.node_list[v]) for v in node.vs])

    def _to_newick_leaf_top_down(self, node):
        if type(node) == int or type(node) == np.int32:
            return 'n{}:{}'.format(node, 1)
        if len(node.vs) == 1:
            return 'n{}:{}'.format(node.id, 1)

        return ','.join([self._to_newick_leaf_top_down(c) for c in node.children])


class SEAT(AgglomerativeClustering):

    def __init__(self, min_k=1, max_k=10,
                 max_g_ratio=0,
                 affinity='precomputed',
                 strategy='top_down',
                 objective='structure_entropy',
                 n_neighbors=10,
                 corr_cut_off=0.8,
                 ):
        self.min_k = min_k
        self.max_k = max_k
        self.ks = range(min_k, max_k+1)
        self.max_g_ratio = max_g_ratio
        self.affinity = affinity
        self.strategy = strategy
        self.objective = objective
        self.n_neighbors = n_neighbors
        self.corr_cut_off = corr_cut_off

    def get_affinity(self, X):
        if self.affinity == 'precomputed':
            aff_m = X
        elif self.affinity == 'knn_neighbors':
            aff_m = kneighbors_graph(X, self.n_neighbors).toarray()
            aff_m = (aff_m + aff_m.T)/2
            aff_m[np.nonzero(aff_m)] = 1

        self.affinity_m = aff_m

    def fit(self, X, y=None):

        X = self._validate_data(X, ensure_min_samples=2, estimator=self)

        if self.min_k is not None and self.min_k <= 0:
            raise ValueError("min_k should be an integer greater than 0."
                             " %s was provided." % str(self.min_k))

        if self.max_k is not None and self.max_k <= 2:
            raise ValueError("max_k should be an integer greater than 2."
                             " %s was provided." % str(self.max_k))

        if self.affinity not in ['precomputed', 'knn_neighbors', 'T10', 'T16']:
            raise ValueError("affinity should be precomputed, knn_neighbors, correlation."
                             " %s was provided." % str(self.affinity))

        if self.strategy not in ['bottom_up', 'top_down']:
            raise ValueError("affinity should be bottom_up, top_down."
                             " %s was provided." % str(self.strategy))

        self.get_affinity(X)

        # build the tree

        setree_class = pySETree

        se_tree = setree_class(self.affinity_m, self.min_k, self.max_k,
                        self.max_g_ratio,
                        self.objective,
                        self.strategy)
        self.se_tree = se_tree
        Z = se_tree.build_tree()
        self.affinity_m = se_tree.affinity_m
        se_tree.cut_tree(Z, self.ks)
        self.vertex_num = se_tree.vertex_num
        self.ks = list(se_tree.ks)
        self.se_scores = se_tree.se_scores
        self.delta_se_scores = se_tree.delta_se_scores
        self.optimal_k = se_tree.optimal_k
        self.labels_ = se_tree.optimal_clusters
        self.Z_ = Z[:, :4]
        self.leaves_list = hierarchy.leaves_list(self.Z_)
        self.order = self._order()
        self.ks_clusters = se_tree.ks_clusters
        self.Z_clusters = se_tree.Z_clusters
        self.submodules = self._get_submodules()
        self.submodule_k = len(se_tree.leafs)

        self.newick = se_tree.to_newick()

        return self

    def _order(self):
        order = [(l, i) for i, l in enumerate(self.leaves_list)]
        order.sort()
        return [i for l, i in order]

    def _get_submodules(self):
        leafs = sorted([(self.order[l.vs[0]], l.id) for l in self.se_tree.leafs])
        order = [(v, i) for i, l in enumerate(leafs) for v in self.se_tree.node_list[l[1]].vs]
        order.sort()
        return [i for n, i in order]

    def oval_embedding(self, a=3, b=2, k=0.2):
        angle = np.array([self.order])*(2*np.pi/len(self.order))
        xcor = a*np.cos(angle)
        ycor = b*np.sqrt(np.exp(k*a*np.cos(angle)))*np.sin(angle)
        plane_coordinate = np.concatenate((xcor, ycor), axis=0).T
        return plane_coordinate
