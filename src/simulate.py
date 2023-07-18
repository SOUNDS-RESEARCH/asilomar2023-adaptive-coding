import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(0, 2, (10000,))
fig = plt.figure()
plt.hist(data, 100)
plt.xlabel("Value")
plt.ylabel("Count")
plt.title("Normal Distribution")

fig.savefig("data/figure.png")
