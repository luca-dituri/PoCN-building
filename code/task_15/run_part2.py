import sys
from pathlib import Path
import numpy as np
from multiprocessing import Pool
import time

base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir / 'code'))

from common.utils import load_config, set_seed
from task_15.network_generation import generate_scale_free_network
from task_15.sandpile import simulate_sandpile
from common.plot_utils import get_log_binned_distribution

def process_realization_scalefree(args):
    seed, gamma, config = args
    set_seed(seed)
    
    N = config['part2']['N']
    mean_k = config['part2']['mean_k']
    f = config['part2']['f']
    
    G = generate_scale_free_network(N, mean_k, gamma)
    sizes, lifetimes = simulate_sandpile(G, num_drops=100000, f=f)
    return sizes, lifetimes

def main():
    config = load_config("task_15")
    num_realizations = config['part2']['realizations']
    base_seed = config['seed']
    gammas = config['part2']['gammas']
    
    data_dir = base_dir / "data" / "task_15"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running Part 2 with {num_realizations} realizations per gamma...")
    
    for gamma in gammas:
        print(f"Processing Gamma = {gamma}...")
        args = [(base_seed + 1000*int(gamma) + i, gamma, config) for i in range(num_realizations)]
        
        with Pool() as pool:
            t0 = time.time()
            results = pool.map(process_realization_scalefree, args)
            print(f"  Done in {time.time()-t0:.2f}s")
            
        sizes = [r[0] for r in results]
        lifetimes = [r[1] for r in results]
        
        s_c, s_p = get_log_binned_distribution(sizes)
        l_c, l_p = get_log_binned_distribution(lifetimes)
        
        np.savez(data_dir / f"part2_gamma_{gamma}.npz", 
                 size_centers=s_c, size_probs=s_p,
                 lifetime_centers=l_c, lifetime_probs=l_p)

    print("Part 2 complete. Data saved.")

if __name__ == "__main__":
    main()
