import networkx as nx
import numpy as np

def make_even(seq):
    seq = np.array(seq, dtype=int)
    if seq.sum() % 2 != 0:
        idx = np.random.randint(0, len(seq))
        seq[idx] += 1
    return seq

def generate_gaussian_network(N, mean_k, std_k, N_b):
    # Sample degrees
    seq = np.random.normal(loc=mean_k, scale=std_k, size=N)
    seq = np.round(seq).astype(int)
    seq[seq <= 0] = 1 # k > 0
    seq = make_even(seq)
    
    G = nx.configuration_model(seq, create_using=nx.Graph)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    _assign_boundary_nodes(G, N_b)
    return G

def generate_uniform_network(N, k_min, k_max, N_b):
    seq = np.random.randint(k_min, k_max + 1, size=N)
    seq = make_even(seq)
    
    G = nx.configuration_model(seq, create_using=nx.Graph)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    _assign_boundary_nodes(G, N_b)
    return G

def _assign_boundary_nodes(G, N_b):
    nodes = list(G.nodes())
    boundary_nodes = np.random.choice(nodes, size=N_b, replace=False)
    
    for n in G.nodes():
        G.nodes[n]['is_boundary'] = False
        G.nodes[n]['n_dissipative'] = 0
        
    for bn in boundary_nodes:
        k_i = G.degree(bn)
        if k_i > 0:
            n_i = np.random.randint(1, k_i + 1)
            G.nodes[bn]['is_boundary'] = True
            G.nodes[bn]['n_dissipative'] = n_i

def generate_scale_free_network(N, mean_k, gamma):
    if gamma >= 100.0: # Approximation for infinity
        # When gamma goes to infinity, alpha goes to 0 -> weights are uniform -> ER graph
        seq = np.random.poisson(lam=mean_k, size=N)
        seq[seq <= 0] = 1
        seq = make_even(seq)
    else:
        # gamma = 1 + 1/alpha => alpha = 1 / (gamma - 1)
        alpha = 1.0 / (gamma - 1.0)
        
        # weights w_i = i^{-alpha} for i=1..N
        indices = np.arange(1, N + 1)
        w = np.power(indices, -alpha, dtype=float)
        
        # We want mean_k * N edges in total (so sum of degrees is mean_k * N)
        # Expected degree k_i = C * w_i
        C = (mean_k * N) / np.sum(w)
        expected_degrees = C * w
        
        # Probabilistic rounding to integers to avoid bias
        floor_k = np.floor(expected_degrees)
        prob = expected_degrees - floor_k
        rand_vals = np.random.random(size=N)
        
        seq = floor_k + (rand_vals < prob)
        seq = seq.astype(int)
        seq[seq <= 0] = 1
        seq = make_even(seq)
    
    G = nx.configuration_model(seq, create_using=nx.Graph)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # Initialize node attributes for sandpile
    for n in G.nodes():
        G.nodes[n]['is_boundary'] = False
        G.nodes[n]['n_dissipative'] = 0
        
    return G
