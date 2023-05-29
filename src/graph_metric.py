# -*- coding: utf-8 -*-
"""
    src.HE
    ~~~~~~~~~~~

    @Copyright: (c) 2022-07 by Lingxi Chen (chanlingxi@gmail.com).
    @License: LICENSE_NAME, see LICENSE for more details.
"""

import numpy as np
from math import log2


def get_v(M, sparse_m, vs):
    # intra- and inter- affinity
    # return np.sum(M[vs, :]) - np.sum([M[i, i] for i in vs])
    v = np.sum(M[vs, :])
    if v == 0:
        v += 1
    return v
    # return sparse_m[vs, :].sum()  # slow


def get_s(M, sparse_m, vs):
    # intra-affinity
    return np.sum(M[np.ix_(vs, vs)])
    ''' slow
    c = 0
    for i in vs:
        for j in vs:
            c += M[i, j]
    return c
    '''


def get_g(M, sparse_m, vs):
    # inter-affinity
    return get_v(M, sparse_m, vs) - get_s(M, sparse_m, vs)


# structure entropy
def get_node_se(vG, g, V, pV):
    return float(g)/float(vG)*log2(float(pV)/float(V))


def get_node_vertices_se(vG, d_log_d, V):
    return - (d_log_d - V*log2(V))/vG


def get_delta_merge_se(M, sparse_m, vG, d, parent, node1, node2):
    # leaf merge
    new_V = node1.V + node2.V
    new_s = node1.s + node2.s
    new_s_tmp = 0
    for b1 in node1.vs:
        for b2 in node2.vs:
            new_s_tmp += M[b1, b2]
    # new_s_tmp = np.sum(M[np.ix_(node1.vs, node2.vs)])   # slow
    new_s += 2*new_s_tmp
    new_se = -(new_V-new_s)/vG*log2(new_V/vG)
    new_se += (new_V*log2(new_V) - node1.d_log_d - node2.d_log_d) / vG
    old_se = node1.se + node2.se \
        + (node1.V*node1.log_V - node1.d_log_d + node2.V*node2.log_V - node2.d_log_d)/vG
    delta = old_se - new_se
    return delta


def get_delta_combine_se(M, sparse_m, vG, parent, node1, node2):
    new_node_V = node1.V + node2.V
    new_node_s = node1.s + node2.s
    new_s_tmp = 0
    for b1 in node1.vs:
        for b2 in node2.vs:
            new_s_tmp += M[b1, b2]
    # new_s_tmp = np.sum(M[np.ix_(node1.vs, node2.vs)])  # slow
    new_node_s += 2*new_s_tmp
    new_node_g = new_node_V - new_node_s
    new_node_se = get_node_se(vG, new_node_g, new_node_V, parent.V)
    new_node1_se = get_node_se(vG, node1.g, node1.V, new_node_V)
    new_node2_se = get_node_se(vG, node2.g, node2.V, new_node_V)
    return node1.se + node2.se - new_node_se - new_node1_se - new_node2_se


# topology entropy
def get_node_te(vG, g, V, pV, eta=1):
    vG, g, V, pV = float(vG), float(g), float(V), float(pV)
    s = V - g
    if s == 0:
        return 0
    diff = log2(s/pV) - 2*eta*log2(V/pV)
    return -s/vG*diff


def get_node_vertices_te(vG, d_log_d, V):
    return 0


# topology entropy louvain like
def get_node_lte(vG, g, V, pV, eta=1):
    vG, g, V, pV = float(vG), float(g), float(V), float(pV)
    s = V - g
    if s == 0:
        return 0
    diff = log2(s/pV) - log2(eta) - 2*log2(V/pV)
    return -s/vG*diff


def get_node_vertices_lte(vG, d_log_d, V):
    return 0


def get_node_score(objective, vG, g, V, pV, eta=1):
    if objective == 'SE':
        return get_node_se(vG, g, V, pV)
    if objective == 'TE':
        return get_node_te(vG, g, V, pV, eta)
    if objective == 'LTE':
        return get_node_lte(vG, g, V, pV, eta)


def get_node_vertices_score(objective, vG, d_log_d, V):
    if objective == 'SE':
        return get_node_vertices_se(vG, d_log_d, V)
    if objective == 'TE':
        return get_node_vertices_te(vG, d_log_d, V)
    if objective == 'LTE':
        return get_node_vertices_lte(vG, d_log_d, V)


# modularity
def get_delta_nm(M, sparse_m, vG, parent, node1, node2, r=1):
    if len(node1.vs) == 1:
        node1_q = - r*np.power(node1.V/vG, 2)
        # node1_q = node1.g/vG - r*np.power(node1.V/vG, 2)
    else:
        node1_q = node1.s/vG - r*np.power(node1.V/vG, 2)
    if len(node2.vs) == 1:
        node2_q = - r*np.power(node2.V/vG, 2)
        # node2_q = node2.g/vG - r*np.power(node2.V/vG, 2)
    else:
        node2_q = node2.s/vG - r*np.power(node2.V/vG, 2)
    new_node_V = node1.V + node2.V
    new_node_s = node1.s + node2.s
    new_s_tmp = 0
    for b1 in node1.vs:
        for b2 in node2.vs:
            new_s_tmp += M[b1, b2]
    new_node_s += 2*new_s_tmp
    # new_node_g = new_node_V - new_node_s

    after = new_node_s/vG - r*np.power(new_node_V/vG, 2)
    return after - (node1_q + node2_q)
