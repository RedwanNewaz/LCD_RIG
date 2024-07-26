import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from scipy.signal import find_peaks

def read_files(args, team_size):
    filepath = f"{args.folder}/{args.env}/distributed"
    for filename in Path(filepath).glob(f"ak_team{team_size}_log_*.csv"):
        data = pd.read_csv(filename)
        rmse = data[args.param].to_numpy()
        if len(rmse) >= args.ub:
            yield rmse[args.lb:args.ub]

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
def generate_latex_history(data, args):

    # Create a DataFrame with custom index
    history = {}
    x = np.arange(args.lb, args.ub)

    # extract results 
    for key, val in data.items():
        scale = 100 if args.param == "coverage" else 1
        val = np.array(val) * scale
        mean = np.mean(val, axis=0)
        yerr = np.std(val, axis=0)
        print(key, val.shape, mean.shape)
        sampled_indices = sparse_sample(x, mean, args.num_samples)

        mean = mean[sampled_indices]
        yerr = yerr[sampled_indices]
        history[f"team{key}_lb"] = mean  - yerr
        history[f"team{key}_mean"] = mean
        history[f"team{key}_ub"] = mean + yerr

    index = x[sampled_indices]
    df = pd.DataFrame(history, index=index)
    df = df.rename_axis("index")
    #save data
    outfile = f"history_{args.env}_{args.param}.csv"
    df.to_csv(outfile)
    print("[+] result saved to ", outfile)

def generate_plot(data, args):
    # plot results
    x = np.arange(args.lb, args.ub)

    for key, val in data.items():
        val = np.array(val)

        mean = np.mean(val, axis=0)
        yerr = np.std(val, axis=0)

        sampled_indices = sparse_sample(x, mean, args.num_samples)
        print(key, yerr.shape, mean.shape)
        mean = mean[sampled_indices]
        yerr = yerr[sampled_indices]
        xx = x[sampled_indices]




        # Plotting
        plt.plot(xx, mean, label=f'team {key}')
        plt.fill_between(xx, mean - yerr, mean + yerr, alpha=0.2)
    plt.xlabel('Simulation Step')
    plt.ylabel(args.param)
    plt.legend()
    plt.savefig(f"result_{args.env}_{args.param}.png")

def main(args):
    # read all csv files and convert them
    data = {}
    for team_size in range(3, 6):
        data[team_size] = [item for item in read_files(args, team_size)]
    
    if args.plot:
        generate_plot(data, args)
    else:
        generate_latex_history(data, args)

  


if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--env", type=str, default="N45W123")
    parser.add_argument("--param", type=str, default="avg_rmse")
    parser.add_argument("--folder", type=str, default="2025")
    parser.add_argument("--ub", type=int, default=3150)
    parser.add_argument("--lb", type=int, default=150)
    parser.add_argument("--num-samples", type=int, default=150)
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()
    main(args)
