import numpy as np
import random


class particle:
    def __init__(self, x, v):
        self.x = x
        self.v = v
        self.x_best = self.x


def particle_swarm_optimization(func, population, k_max, w=0.5, c1=0.25, c2=2.0):
    n = population[0].x.size
    shape = population[0].x.shape
    result_by_iteration = []

    x_best, y_best = population[0].x_best.copy(), float("inf")
    w_max, w_min = w, w
    for P in population:
        y = func(np.argsort(P.x))
        if y < y_best:
            x_best = P.x.copy()
            y_best = y

    for k in range(k_max):
        k_sum = 0
        for P in population:
            r1 = np.random.uniform(0, 1, size=shape)
            r2 = np.random.uniform(0, 1, size=shape)

            P.x = np.copy(P.x) + np.copy(P.v)
            P.v = w * np.copy(P.v) + c1 * r1 * (np.copy(P.x_best) - np.copy(P.x)) + c2 * r2 * (x_best - np.copy(P.x))
            P.v = np.clip(P.v, -(shape[1] - 1), shape[1] - 1)

            y = func(np.argsort(P.x))
            if y < y_best:
                x_best = P.x.copy()
                y_best = y
            if y < func(np.argsort(P.x_best)):
                P.x_best = P.x.copy()

            w = w_max - (w_max - w_min) / k_max * (k + 1)
            if w < w_min:
                w_min = w

            k_sum += y
            k_mean = k_sum / len(population)

        result_by_iteration.append(k_mean)

    return population, np.array(result_by_iteration)

if __name__ == '__main__':
    print ("main")
