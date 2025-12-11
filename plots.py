import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    runtimes = dict()
    with open(args.path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            print('hi')
            # parse to numbers so we can sort/plot
            p = str(row["p"])
            n = int(row["n"])
            time_us = float(row["time_us"])

            p_dict = runtimes.setdefault(p, dict())
            p_dict.setdefault('n', []).append(n)
            p_dict.setdefault('t', []).append(time_us)
    plot_runtime(runtimes, False)
    plot_runtime(runtimes, True)
    plot_speedup(runtimes, False)
    plot_speedup(runtimes, True)
    # plot_efficiency(runtimes, False)
    # plot_efficiency(runtimes, True)


def plot_runtime(runtimes, log=False):
    plt.figure()

    for p in runtimes.keys():
        n_vals = runtimes[p]['n']
        t_vals = runtimes[p]['t']
        plt.plot(n_vals, t_vals, marker='o', label=f"p={p}")

    plt.xlabel("n")
    plt.ylabel("t (ns)")
    plt.legend()
    plt.grid(True)
    if log:
        plt.title("Runtime vs n by p (log)")
        plt.yscale("log", base=2)
        plt.xscale("log", base=2)
    else:
        plt.title("Runtime vs n by p (linear)")
    plt.tight_layout()
    plt.show()

def plot_speedup(runtimes, log=False):
    plt.figure()

    p_1_t = np.array(runtimes['serial_host']['t'])
    for p in runtimes.keys():
        n_vals = np.array(runtimes[p]['n'])
        t_vals = np.array(runtimes[p]['t'])
        plt.plot(n_vals, p_1_t/t_vals, marker='o', label=f"p={p}")

    plt.xlabel("n")
    plt.ylabel("speedup")
    plt.legend()
    plt.grid(True)
    if log:
        plt.title("Speedup (log)")
        plt.yscale("log", base=2)
        plt.xscale("log", base=2)
    else:
        plt.title("Speedup (linear)")
    plt.tight_layout()
    plt.show()

def plot_efficiency(runtimes, log=False):
    plt.figure()

    p_1_t = np.array(runtimes[1]['t'])
    for p in runtimes.keys():
        n_vals = np.array(runtimes[p]['n'])
        t_vals = np.array(runtimes[p]['t'])
        plt.plot(n_vals, p_1_t/t_vals/p, marker='o', label=f"p={p}")

    plt.xlabel("n")
    plt.ylabel("efficiency")
    plt.legend()
    plt.grid(True)
    if log:
        plt.title("Efficiency (log)")
        plt.yscale("log", base=2)
        plt.xscale("log", base=2)
    else:
        plt.title("Efficiency (linear)")
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
