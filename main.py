import numpy as np


class Swarm:
    def __init__(self, epoch: int, n_particles: int, lb: list, ub: list, c: list, w: float, fun: callable):
        self._epoch = epoch
        self._n_particles = n_particles
        self._swarm_best = None
        self._swarm = []
        for _ in range(self._n_particles):
            self._swarm.append(Particle(lb, ub, c, w, fun, self))

    def get_swarm(self):
        return [particle.position for particle in self._swarm]

    @property
    def swarm_best(self):
        return self._swarm_best

    @swarm_best.setter
    def swarm_best(self, x):
        self._swarm_best = x

    def optimize(self):
        for e in range(self._epoch):
            for particle in self._swarm:
                if e < self._epoch/3:
                    particle.update(explore=True)
                else:
                    particle.update()
        return True

    def step(self, explore=False):
        for particle in self._swarm:
            particle.update(explore=explore)


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

        self.cost = fun(self._position)
        self._particle_best = [self._position, self.cost]

        if self._parent.swarm_best is None:
            self._parent._swarm_best = [self._position, self.cost]
        if self._parent.swarm_best[1] > self.cost:
            self._parent._swarm_best = [self._position, self.cost]

    @property
    def position(self):
        return self._position

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


if __name__ == '__main__':
    def cost_fun(x):
        return (1.5 - x[0] - x[0] * x[1]) ** 2 +\
               (2.25 - x[0] + (x[0] * x[1]) ** 2) ** 2 +\
               (2.625 - x[0] + (x[0] * x[1]) ** 3) ** 2

    # swarm (epoch: int, n_particles: int, lb: list, ub: list, c: list, w: float, fun: callable)
    swarm = Swarm(40, 20, [-4.5, -4.5], [4.5, 4.5], [0.4, 0.2], np.random.uniform(0.4, 0.9), cost_fun)
    if swarm.optimize():
        print(swarm.swarm_best)


    def cost3(x):
        return (1.5 - x[0] - x[2] * x[1]) ** 2 +\
               (2.25 - x[0] + (x[2] * x[1]) ** 2) ** 2 +\
               (2.625 - x[0] + (x[2] * x[1]) ** 3) ** 2

    # swarm (epoch: int, n_particles: int, lb: list, ub: list, c: list, w: float, fun: callable)
    swarm = Swarm(40, 20, [-4.5, -4.5, -4.5], [4.5, 4.5, 4.5], [0.4, 0.2], np.random.uniform(0.4, 0.9), cost3)
    if swarm.optimize():
        print(swarm.swarm_best)


    def bird_function(x):
        return np.sin(x[0])*np.e**((1-np.cos(x[1]))**2) + \
               np.cos(x[1])*np.e**((1-np.sin(x[0]))**2) + \
               (x[0]+x[1])**2

    bird = Swarm(50, 40, [-10, -10], [10,10], [0.4, 0.2], np.random.uniform(0.4, 0.9), bird_function)
    if bird.optimize():
        print(f'Mishra\'s Bird function{bird.swarm_best} global best[-3.1302, -1.5821] -106,76')

    '''
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    y = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    X, Y = np.meshgrid(x, y)
    Z = bird_function([X, Y])
    ax.set_xlabel("(x-axis)")
    ax.set_ylabel("(y-axis)")
    ax.set_zlabel("(z-axis)")
    ax.plot_surface(X, Y, Z, cmap='plasma')
    ax.contour3D(X, Y, Z)
    plt.show()
    '''