import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

# from pythonutils import utils
from pythonutils import dockersim

runs = int(sys.argv[1])
nprocesses = int(sys.argv[2])

print(f"Running {runs} in {nprocesses} processes.")


def simulate(a, b, rng=np.random.default_rng()):
    data = rng.normal(loc=a, size=(b,))
    # time.sleep(0.4)
    return data


tasks = [{"run_nr": run, "a": 2, "b": 100} for run in range(runs)]
sim = dockersim.DockerSim(simulate, tasks, 1234)
sim.run(num_processes=nprocesses)

df = pl.scan_csv("data/results_*.csv")

fig = plt.figure()
sns.lineplot(
    df.collect(),
    x="series",
    y="value",
    errorbar=("sd", 1.96),
)
plt.xlabel("Series")
plt.ylabel("Value")
plt.title("Random value")

fig.savefig("plots/figure.png")
