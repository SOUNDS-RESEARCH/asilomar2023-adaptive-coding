import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

# from pythonutils import utils
from pythonutils import dockersim

# The parameters of this script, the number of relizations per parameter combination,
# the seed of the random number generator, the number of processes to be used on parallel
runs = int(sys.argv[1])
seed = int(sys.argv[2])
num_processes = int(sys.argv[3])


# This is the actual simulation function. We simulate the ADMM algorithm
# (imported from admm_fq_*) for different algorithm onfigurations and parameter combinations.
def simulate(a, b, rng=np.random.default_rng()):
    data = rng.normal(loc=a, scale=b, size=(6000,))
    # time.sleep(0.4)
    for val in data:
        yield {"val": val, "sq_val": val * val}


# This has to match the dictionary keys that simulate yields
return_value_names = ["val", "sq_val"]

# Define the tasks, which are basically all the parameter combinations for
# which the algorithms are supposed to be run
tasks = [{"a": a, "b": b} for a in range(1, 10) for b in range(1, 10)]

# Create and run the simulation
sim = dockersim.DockerSim(simulate, tasks, return_value_names, seed, datadir="data")
sim.run(runs=runs, num_processes=num_processes)

# Read the data from the resulting csv files (we are using polars lazy api)
df = pl.scan_csv("data/results_*.csv")

# Plot and save relevant figures
fig = plt.figure()
sns.lineplot(
    df.filter(pl.col("b") == 5).collect(),
    x="series",
    y="val",
    hue="a",
    errorbar=("sd", 1.96),
)
plt.xlabel("Series")
plt.ylabel("Value")
plt.title("Random value")

fig.savefig("plots/figure.png")

fig = plt.figure()
sns.lineplot(
    df.filter(pl.col("b") == 5).collect(),
    x="series",
    y="sq_val",
    hue="a",
    errorbar=("sd", 1.96),
)
plt.xlabel("Series")
plt.ylabel("Value")
plt.title("Random value")

fig.savefig("plots/figure1.png")
