import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from main import Swarm


def cost(x):
    return (1.49 - x[0] - x[0] * x[1]) ** 2 +\
           (2.25 - x[0] + (x[0] * x[1]) ** 2) ** 2 +\
           (2.625 - x[0] + (x[0] * x[1]) ** 3) ** 2


# swarm (epoch: int, n_particles: int, lb: list, ub: list, c: list, w: float, fun: callable)
swarm = Swarm(40, 20, [-4.5, -4.5], [4.5, 4.5], [0.4, 0.2], np.random.uniform(0.1, 0.8), cost)
epoch = 50
filenames = []
bests = []
for e in range(epoch):
    fig, axs = plt.subplots(2, 1, figsize=(5, 10))
    fig.tight_layout()
    x = np.array(swarm.get_swarm()[0:])[:, 0]
    y = np.array(swarm.get_swarm()[0:])[:, 1]

    axs[0].scatter(x, y, alpha=0.5)
    axs[0].set_xlim([-5, 5])
    axs[0].set_ylim([-5, 5])

    bests.append(swarm.swarm_best[1])
    bests_plt = np.array(bests)
    axs[1].set_title("Function minimum")
    axs[1].set_ylabel('Value')
    axs[1].set_xlabel('Epoch')
    axs[1].set_yscale('log')
    axs[1].plot(np.linspace(0, len(bests_plt), len(bests_plt)), bests_plt)

    # create file
    filename = f'{e}.png'
    filenames.append(filename)

    # save frame
    plt.savefig(filename)
    plt.close()

    if e < 20:
        swarm.step(True)
    else:
        swarm.step()
print(swarm.swarm_best)
with imageio.get_writer('pso.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    os.remove(filename)
