import numpy as np
import matplotlib.pyplot as plt
import time
from pythonutils import utils

data = np.random.normal(0, 2, (10000,))
for n in utils.progressBar(
    range(10),
    prefix="Progress:",
    length=50,
    printEnd="",
):
    data += np.random.normal(0, 2, (10000,))
    time.sleep(0.1)

fig = plt.figure()
plt.hist(data, 100)
plt.xlabel("Value")
plt.ylabel("Count")
plt.title("Normal Distribution")

fig.savefig("data/figure.png")
