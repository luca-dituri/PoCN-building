import numpy as np
import networkx as nx
from numba import njit
import random

@njit
def run_sandpile_numba(adj_indptr, adj_indices, thresholds, f, num_drops):
    N = len(thresholds)
    loads = np.zeros(N, dtype=np.int32)
    for i in range(N):
        if thresholds[i] > 0:
            loads[i] = np.random.randint(0, thresholds[i])
            
    avalanche_sizes = np.zeros(num_drops, dtype=np.int64)
    lifetimes = np.zeros(num_drops, dtype=np.int64)
    
    for drop in range(num_drops):
        # Drop grain on random node
        target = np.random.randint(0, N)
        loads[target] += 1
        
        # Check if avalanche starts
        if loads[target] >= thresholds[target]:
            # Avalanche!
            # Use arrays to keep track of active nodes (queue)
            active_nodes = np.zeros(N, dtype=np.int32)
            active_count = 0
            
            active_nodes[active_count] = target
            active_count += 1
            
            in_queue = np.zeros(N, dtype=np.bool_)
            in_queue[target] = True
            
            size = 0
            lifetime = 0
            
            while active_count > 0:
                lifetime += 1
                
                next_active = np.zeros(N, dtype=np.int32)
                next_count = 0
                
                # Topple all active in parallel-like fashion for this timestep
                for i in range(active_count):
                    node = active_nodes[i]
                    if loads[node] >= thresholds[node]:
                        # Number of times it topples
                        n_topples = loads[node] // thresholds[node]
                        loads[node] = loads[node] % thresholds[node]
                        size += n_topples
                        
                        # distribute to neighbors
                        start = adj_indptr[node]
                        end = adj_indptr[node+1]
                        
                        for _ in range(n_topples):
                            for idx in range(start, end):
                                neighbor = adj_indices[idx]
                                if f > 0.0:
                                    if np.random.random() >= f:
                                        loads[neighbor] += 1
                                        if loads[neighbor] >= thresholds[neighbor] and not in_queue[neighbor]:
                                            next_active[next_count] = neighbor
                                            next_count += 1
                                            in_queue[neighbor] = True
                                else:
                                    loads[neighbor] += 1
                                    if loads[neighbor] >= thresholds[neighbor] and not in_queue[neighbor]:
                                        next_active[next_count] = neighbor
                                        next_count += 1
                                        in_queue[neighbor] = True
                
                for i in range(next_count):
                    in_queue[next_active[i]] = False
                
                # Copy next_active to active_nodes
                for i in range(next_count):
                    active_nodes[i] = next_active[i]
                active_count = next_count
                
            avalanche_sizes[drop] = size
            lifetimes[drop] = lifetime
            
    return avalanche_sizes, lifetimes

def simulate_sandpile(G, num_drops=100000, f=0.0):
    # Convert graph to CSR format for Numba
    adj = nx.to_scipy_sparse_array(G, format='csr')
    adj_indptr = adj.indptr
    adj_indices = adj.indices
    
    N = G.number_of_nodes()
    thresholds = np.zeros(N, dtype=np.int32)
    
    for n in G.nodes():
        n_dissipative = G.nodes[n].get('n_dissipative', 0)
        # Threshold = degree in graph + n_dissipative
        thresholds[n] = G.degree(n) + n_dissipative
        if thresholds[n] == 0:
            thresholds[n] = 1 # Avoid division by zero, though degree 0 won't topple anywhere
            
    # Run simulation
    sizes, lifetimes = run_sandpile_numba(adj_indptr, adj_indices, thresholds, f, num_drops)
    
    # Discard first 20% as transient (burn-in period)
    burn_in = int(0.2 * num_drops)
    return sizes[burn_in:], lifetimes[burn_in:]
