import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

# from pythonutils import utils
from pythonutils import dockersim

runs = int(sys.argv[1])
nprocesses = int(sys.argv[2])


def simulate(a, b, rng=np.random.default_rng()):
    data = rng.normal(loc=a, scale=b, size=(6000,))
    # time.sleep(0.4)
    return data


tasks = [{"a": a, "b": b} for a in range(1, 10) for b in range(1, 10)]
sim = dockersim.DockerSim(simulate, tasks, 555, datadir="data")
print(f"Running {runs} of {len(tasks)} tasks each in {nprocesses} processes.")
sim.run(runs=runs, num_processes=nprocesses)

df = pl.scan_csv("data/results_*.csv")

fig = plt.figure()
sns.lineplot(
    df.filter(pl.col("b") == 5).collect(),
    x="series",
    y="value",
    hue="a",
    errorbar=("sd", 1.96),
)
plt.xlabel("Series")
plt.ylabel("Value")
plt.title("Random value")

fig.savefig("plots/figure.png")
