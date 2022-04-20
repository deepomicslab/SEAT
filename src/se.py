import numpy as np
from math import log2


def get_v(M, sparse_m, vs):
    v = np.sum(M[vs, :])
    if v == 0:
        v += 1
    return v


def get_s(M, sparse_m, vs):
    return np.sum(M[np.ix_(vs, vs)])


def get_g(M, sparse_m, vs):
    # inter-affinity
    return get_v(M, sparse_m, vs) - get_s(M, sparse_m, vs)


def get_se(vG, g, V, pV):
    return float(g)/float(vG)*log2(float(pV)/float(V))


def get_se_for_vertices(M, sparse_m, vG, vs, pV):
    g = get_g(M, sparse_m, vs)
    V = get_v(M, sparse_m, vs)
    return get_se(vG, sparse_m, g, V, pV)


def get_se_for_vertex(M, sparse_m, d, vG, bin, pV):
    if get_se(vG, d[bin], d[bin], pV) != get_se_for_vertices(M, vG, [bin], pV):
        raise ValueError("bin se error")
    return get_se_for_vertex(M, sparse_m, vG, [bin], pV)


def get_delta_se_plus(M, sparse_m, vG, d, parent, node1, node2):
    # leaf merge
    new_V = node1.V + node2.V
    new_s = node1.s + node2.s
    new_s_tmp = 0
    for b1 in node1.vs:
        for b2 in node2.vs:
            new_s_tmp += M[b1, b2]
    new_s += 2*new_s_tmp
    new_se = (new_V-new_s)/vG*log2(vG/new_V) + new_V*log2(new_V)/vG
    delta = node1.se + node2.se - new_se \
        + (node1.V*node1.log_V + node2.V*node2.log_V)/vG
    return delta


def get_delta_se(M, sparse_m, vG, parent, node1, node2):
    new_node_V = node1.V + node2.V
    new_node_s = node1.s + node2.s
    new_s_tmp = 0
    for b1 in node1.vs:
        for b2 in node2.vs:
            new_s_tmp += M[b1, b2]
    new_node_s += 2*new_s_tmp
    new_node_g = new_node_V - new_node_s
    new_node_se = get_se(vG, new_node_g, new_node_V, parent.V)
    new_node1_se = get_se(vG, node1.g, node1.V, new_node_V)
    new_node2_se = get_se(vG, node2.g, node2.V, new_node_V)
    return node1.se + node2.se - new_node_se - new_node1_se - new_node2_se


def get_delta_nm(M, sparse_m, vG, parent, node1, node2, r=1):
    if len(node1.vs) == 1:
        node1_q = - r*np.power(node1.V/vG, 2)
    else:
        node1_q = node1.s/vG - r*np.power(node1.V/vG, 2)
    if len(node2.vs) == 1:
        node2_q = - r*np.power(node2.V/vG, 2)
    else:
        node2_q = node2.s/vG - r*np.power(node2.V/vG, 2)
    new_node_V = node1.V + node2.V
    new_node_s = node1.s + node2.s
    new_s_tmp = 0
    for b1 in node1.vs:
        for b2 in node2.vs:
            new_s_tmp += M[b1, b2]
    new_node_s += 2*new_s_tmp

    after = new_node_s/vG - r*np.power(new_node_V/vG, 2)
    return after - (node1_q + node2_q)
