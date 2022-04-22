import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


class Swarm:
    def __init__(self, epoch: int, n_particles: int, lb: list, ub: list, c: list, w: float, fun: callable):
        self._epoch = epoch
        self._n_particles = n_particles
        self._lb = lb
        self._ub = ub
        self._c = c
        self._w = w
        self._fun = fun
        self._swarm_best = None
        self._swarm = []
        for _ in range(self._n_particles):
            self._swarm.append(Particle(self._lb, self._ub, self._c, self._w, self._fun, self))

    @property
    def swarm_best(self):
        return self._swarm_best

    @swarm_best.setter
    def swarm_best(self, x):
        self._swarm_best = x

    def optimize(self):
        for e in range(self._epoch):
            for particle in self._swarm:
                particle.update()
            print(self._swarm_best)
        return self._swarm_best


class Particle:
    def __init__(self, lb: list, ub: list, c: list, w: float, fun: callable, parent: Swarm):
        self._lb = lb
        self._ub = ub
        self._c = c
        self._w = w
        self._fun = fun
        self._parent = parent
        self._cost = None
        self._cexplore = [0.05, 0.05]
        self._velocity = np.random.uniform(np.array(self._lb) / 2, np.array(self._ub) / 2, len(self._lb))
        self._position = np.random.uniform(self._lb, self._ub, len(self._lb))

        self.cost = cost_fun(self._position)
        self._particle_best = [self._position, self.cost]

        if self._parent.swarm_best is None:
            self._parent._swarm_best = [self._position, self.cost]
        if self._parent.swarm_best[1] > self.cost:
            self._parent._swarm_best = [self._position, self.cost]

    def update(self, explore=False):

        # check boundaries
        possible_position = self._position + self._velocity
        for i in range(len(self._ub)):
            if possible_position[i] > self._ub[i] or possible_position[i] < self._lb[i]:
                self._velocity[i] = -self._velocity[i]

        # count cost
        self._cost = self._fun(self._position)

        # update velocity
        # vx = w * x + c1 * (global - x) + c2(local - x)
        r = np.random.rand(2)
        if explore:
            self._velocity = self._velocity \
                             + self._cexplore[0] * r[0] * (self._parent.swarm_best[0] - self._position) \
                             + self._cexplore[1] * r[1] * (self._particle_best[0] - self._position)
        else:
            self._velocity = self._w * self._velocity \
                             + self._c[0] * r[0] * (self._parent.swarm_best[0] - self._position) \
                             + self._c[1] * r[1] * (self._particle_best[0] - self._position)

        # update position
        self._position = self._position + self._velocity

        # update particle best
        if self._particle_best[1] > self._cost:
            self._particle_best = [self._position, self._cost]

        # update swarm best
        if self._parent.swarm_best[1] > self._cost:
            self._parent.swarm_best = [self._position, self._cost]


def cost_fun(x):
    return (1.5 - x[0] - x[0] * x[1]) ** 2 +\
           (2.25 - x[0] + (x[0] * x[1]) ** 2) ** 2 +\
           (2.625 - x[0] + (x[0] * x[1]) ** 3) ** 2


# swarm (epoch: int, n_particles: int, lb: list, ub: list, c: list, w: float, fun: callable)
swarm = Swarm(40, 20, [-4.5, -4.5], [4.5, 4.5], [0.4, 0.2], np.random.uniform(0.1, 0.8), cost_fun)
swarm.optimize()

'''swarm_best = None

swarm = []
for _ in range(20):
    swarm.append(particle([-4.5, -4.5], [4.5, 4.5], [0.4, 0.2], np.random.uniform(0.1, 0.8)))

epoch = 100
bests = []
filenames = []

for e in range(epoch):
    fig.tight_layout()
    fig, axs = plt.subplots(2, 1, figsize=(5, 10))
    x = []
    [x.append(pos.position()[0]) for pos in swarm]
    y = []
    [y.append(pos.position()[1]) for pos in swarm]
    w = []
    [w.append(pos._w) for pos in swarm]
    axs[0].scatter(x, y, alpha=0.5)
    axs[0].set_xlim([-5, 5])
    axs[0].set_ylim([-5, 5])
    bests.append(swarm_best[1])
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

    for particle in swarm:
        if e < epoch / 3:
            particle.update(explore=True)
        else:
            particle.update()

print(swarm_best)

with imageio.get_writer('pso.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    os.remove(filename)'''