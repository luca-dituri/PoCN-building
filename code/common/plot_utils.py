import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Shared theme constants (applied by setup_plot_style)
FONT_SIZE = 15
LEGEND_SIZE = 11
TITLE_SIZE = 13

def _default_style_name():
    base_dir = Path(__file__).parent.parent.parent
    with open(base_dir / "config" / "shared.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config.get("plot_style", "seaborn-v0_8-paper")

def get_figure_dir(task_name="task_15"):
    base_dir = Path(__file__).parent.parent.parent
    fig_dir = base_dir / "latex" / "figures" / task_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir

def setup_plot_style(style_name=None):
    style = style_name or _default_style_name()
    try:
        plt.style.use(style)
    except:
        pass  # fallback to default
    plt.rcParams.update({
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 3,
        'ytick.labelsize': FONT_SIZE - 3,
        'legend.fontsize': LEGEND_SIZE,
        'axes.titlesize': TITLE_SIZE,
    })

def get_log_binned_distribution(runs_data, num_bins=50):
    if not runs_data:
        return np.array([]), np.array([])
        
    all_data = np.concatenate(runs_data)
    all_data = all_data[all_data > 0]
    
    if len(all_data) == 0:
        return np.array([]), np.array([])
        
    min_val, max_val = np.min(all_data), np.max(all_data)
    
    if max_val <= min_val:
        return np.array([min_val]), np.array([1.0])
        
    # Smooth monotonic log-binning
    base = (max_val / min_val) ** (1.0 / num_bins)
    base = max(base, 1.05)
    
    edges = [int(min_val)]
    # Ensure the last edge is strictly greater than max_val so np.histogram's 
    # last-bin inclusive behavior doesn't artificially inflate the last bin's density
    while edges[-1] <= max_val:
        next_edge = max(edges[-1] + 1, int(np.floor(edges[-1] * base)))
        edges.append(next_edge)
        
    edges = np.array(edges)
    bin_widths = np.diff(edges)
    bin_centers = np.sqrt(edges[1:] * edges[:-1]) # geometric mean
    
    counts, _ = np.histogram(all_data, bins=edges)
    probs = counts / (len(all_data) * bin_widths)
    
    valid = counts > 0
    return bin_centers[valid], probs[valid]
