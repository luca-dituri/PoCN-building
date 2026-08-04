import sys
from pathlib import Path
import numpy as np
from multiprocessing import Pool
import time

# Add root project folder to sys.path
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir / 'code'))

from common.utils import load_config, set_seed
from task_15.network_generation import generate_gaussian_network, generate_uniform_network
from task_15.sandpile import simulate_sandpile
from common.plot_utils import get_log_binned_distribution

def process_realization_gaussian(args):
    seed, config = args
    set_seed(seed)
    N = config['part1']['N']
    N_b = config['part1']['N_b']
    mean_k = config['part1']['gaussian']['mean_k']
    std_k = config['part1']['gaussian']['std_k']
    
    G = generate_gaussian_network(N, mean_k, std_k, N_b)
    sizes, lifetimes = simulate_sandpile(G, num_drops=100000, f=0.0)
    return sizes, lifetimes

def process_realization_uniform(args):
    seed, config = args
    set_seed(seed)
    N = config['part1']['N']
    N_b = config['part1']['N_b']
    k_min = config['part1']['uniform']['k_min']
    k_max = config['part1']['uniform']['k_max']
    
    G = generate_uniform_network(N, k_min, k_max, N_b)
    sizes, lifetimes = simulate_sandpile(G, num_drops=100000, f=0.0)
    return sizes, lifetimes

def main():
    config = load_config("task_15")
    num_realizations = config['part1']['realizations']
    base_seed = config['seed']
    
    data_dir = base_dir / "data" / "task_15"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running Part 1 with {num_realizations} realizations per network type...")
    
    args_gaussian = [(base_seed + i, config) for i in range(num_realizations)]
    args_uniform = [(base_seed + 100 + i, config) for i in range(num_realizations)]
    
    with Pool() as pool:
        print("Processing Gaussian Networks...")
        t0 = time.time()
        results_gaussian = pool.map(process_realization_gaussian, args_gaussian)
        print(f"Done in {time.time()-t0:.2f}s")
        
        print("Processing Uniform Networks...")
        t0 = time.time()
        results_uniform = pool.map(process_realization_uniform, args_uniform)
        print(f"Done in {time.time()-t0:.2f}s")
        
    # Aggregate and bin
    sizes_g = [r[0] for r in results_gaussian]
    lifetimes_g = [r[1] for r in results_gaussian]
    
    s_c_g, s_p_g = get_log_binned_distribution(sizes_g)
    l_c_g, l_p_g = get_log_binned_distribution(lifetimes_g)
    
    np.savez(data_dir / "part1_gaussian.npz", 
             size_centers=s_c_g, size_probs=s_p_g,
             lifetime_centers=l_c_g, lifetime_probs=l_p_g)
             
    sizes_u = [r[0] for r in results_uniform]
    lifetimes_u = [r[1] for r in results_uniform]
    
    s_c_u, s_p_u = get_log_binned_distribution(sizes_u)
    l_c_u, l_p_u = get_log_binned_distribution(lifetimes_u)
    
    np.savez(data_dir / "part1_uniform.npz", 
             size_centers=s_c_u, size_probs=s_p_u,
             lifetime_centers=l_c_u, lifetime_probs=l_p_u)

    print("Part 1 complete. Data saved.")

if __name__ == "__main__":
    main()
