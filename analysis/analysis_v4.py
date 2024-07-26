import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
import numpy as np 

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # or "sans-serif" depending on your document's font
    "font.serif": ["Palatino"],  # or any other font available in your LaTeX distribution
    "pgf.rcfonts": False,  # to prevent pgf from overriding rc settings
})

# Function to perform sparse sampling
def sparse_sample(x, y, num_samples):
    # Step 1: Detect peaks and valleys
    peaks, _ = find_peaks(y)
    valleys, _ = find_peaks(-y)
    important_points = np.concatenate((peaks, valleys))

    # Step 2: Add start and end points
    important_points = np.concatenate(([0], important_points, [len(y) - 1]))

    # Step 3: Uniformly sample the remaining points
    if len(important_points) < num_samples:
        remaining_samples = num_samples - len(important_points)
        all_indices = np.arange(len(y))
        remaining_indices = np.setdiff1d(all_indices, important_points)
        sampled_remaining_indices = np.linspace(0, len(remaining_indices) - 1, remaining_samples).astype(int)
        sampled_remaining_indices = remaining_indices[sampled_remaining_indices]
        sampled_indices = np.concatenate((important_points, sampled_remaining_indices))
    else:
        sampled_indices = important_points

    # Sort sampled indices
    sampled_indices = np.sort(sampled_indices)

    return sampled_indices

def read_csv_files(num_samples=150):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    NAMES = ["N43W080",  "N45W123", "N47W124", "FIU MMC Lake"]

    for j, filename in enumerate(sorted(Path('.').glob("*rmse.csv"))):
        data = pd.read_csv(filename)
        xx = data["index"]
        for i in range(3, 6):
            mean = data[f"team{i}_mean"]
            lb = data[f"team{i}_lb"]
            ub = data[f"team{i}_ub"]

            # sparse sample 
            indxes = sparse_sample(xx, mean, num_samples)
            x = xx[indxes]
            mean = mean[indxes]
            lb = lb[indxes]
            ub = ub[indxes]
            axs[j].plot(x, mean, label=f'team {i}')
            axs[j].fill_between(x, lb, ub, alpha=0.2)

        axs[j].set_xlabel('Simulation Step')
        axs[j].set_ylabel("Average RMSE")

        if j == 3:
            axs[j].set_ylim(0.1, 0.3)

        axs[j].set_title(NAMES[j])
        axs[j].legend()
        axs[j].grid()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # plt.show()
    plt.savefig('rmse.pgf')



if __name__ == '__main__':
    read_csv_files()
