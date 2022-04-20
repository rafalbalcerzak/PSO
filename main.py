import numpy as np
import matplotlib.pyplot as plt


def cost_fun(x, y):
    return (1.5 - x - x * y) ** 2 + (2.25 - x + (x * y) ** 2) ** 2 + (2.625 - x + (x * y) ** 3) ** 2


# settings
n_particles = 20
x_min, x_max = -4.5, 4.5
y_min, y_max = -4.5, 4.5
iterations = 30
c1 = 0.4
c2 = 0.4


# coordinates
x = np.random.uniform(x_min, x_max, size=n_particles)
y = np.random.uniform(y_min, y_max, size=n_particles)

# weight
w = np.random.uniform(0.6, 0.9, size=n_particles)

# velocities
vx = np.random.uniform(0, 5, size=n_particles)
vy = np.random.uniform(0, 5, size=n_particles)

particles = np.column_stack((x, y, vx, vy, w))
costs = np.ones(n_particles)

# cost for current locations
for p in range(n_particles):
    costs[p] = cost_fun(particles[p][0], particles[p][1])

local_best = np.column_stack((x, y, costs))
gi = costs.argmin()
global_best = [particles[gi][0], particles[gi][1], costs[gi]]

plt.scatter(particles[:, 0], particles[:, 1])
plt.xlim([-4.5, 4.5])
plt.ylim([-4.5, 4.5])
plt.show()

# for each epoch
for epoch in range(iterations):
    for i, particle in enumerate(particles):
        # new positon
        particle[0] = particle[0] + particle[2]  # x = x + vx
        particle[1] = particle[1] + particle[3]  # y = y + vy
        r1, r2 = np.random.rand(2)
        # new velocity vx = w * x + c1 * (global - x) + c2(local - x)
        particle[2] = particle[4] * particle[2] + c1 * r1 * (global_best[0] - particle[0]) + c2 * r2 * (local_best[i][0] - particle[0])  # vx
        particle[3] = particle[4] * particle[3] + c1 * r1 * (global_best[1] - particle[1]) + c2 * r2 * (local_best[i][1] - particle[1])  # vy

        costs[i] = cost_fun(particle[0], particle[1])

        if costs[i] < local_best[i][2]:
            local_best[i][2] = costs[i]


    gi = costs.argmin()
    if costs[gi] < global_best[2]:
        global_best = [particles[gi][0], particles[gi][1], costs[gi]]

    plt.scatter(particles[:,0], particles[:,1])
    plt.xlim([-4.5, 4.5])
    plt.ylim([-4.5, 4.5])
    plt.show()
