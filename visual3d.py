import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from main import Swarm


def cost3(x):
    return (1.49 - x[2] - x[0] * x[1]) ** 2 +\
           (2.25 - x[2] + (x[0] * x[1]) ** 2) ** 2 +\
           (2.625 - x[0] + (x[0] * x[1]) ** 3) ** 2


# swarm (epoch: int, n_particles: int, lb: list, ub: list, c: list, w: float, fun: callable)
swarm = Swarm(40, 20, [-4.5, -4.5, -4.5], [4.5, 4.5, 4.5], [0.4, 0.2], np.random.uniform(0.1, 0.8), cost3)

filenames = []
bests = []
epoch = 50
explore = 20
for e in range(epoch):

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    fig.tight_layout()
    x = np.array(swarm.get_swarm()[0:])[:, 0]
    y = np.array(swarm.get_swarm()[0:])[:, 1]
    z = np.array(swarm.get_swarm()[0:])[:, 2]

    ax.scatter3D(x, y, z, alpha=0.5)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    ax = fig.add_subplot(1, 2, 2,)
    bests.append(swarm.swarm_best[1])
    bests_plt = np.array(bests)
    ax.set_title("Function minimum")
    ax.set_ylabel('Value')
    ax.set_xlabel('Epoch')
    ax.set_yscale('log')
    ax.plot(np.linspace(0, len(bests_plt), len(bests_plt)), bests_plt)

    # create file
    filename = f'{e}.png'
    filenames.append(filename)

    # save frame
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    if e < explore:
        swarm.step(True)
    else:
        swarm.step()
    print(f'Epoch: {e}')
print(swarm.swarm_best)
with imageio.get_writer('pso3d.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    os.remove(filename)
