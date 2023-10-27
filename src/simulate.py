import sys
import static_convergence
import dynamic_tracking
import plot


# The parameters of this script, the number of relizations per parameter combination,
# the seed of the random number generator, the number of processes to be used on parallel
runs = int(sys.argv[1])
seed = int(sys.argv[2])
num_processes = int(sys.argv[3])
execute = sys.argv[4] if len(sys.argv) > 4 else "--all"

# simulate
if execute == "--all" or execute == "--dynamic":
    dynamic_tracking.dynamic_tracking(runs, seed, num_processes)

# plot the relevant figures (in script plot.py)
if execute == "--all" or execute == "--plot":
    plot.plot()
