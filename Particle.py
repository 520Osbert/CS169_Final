import numpy as np
import random
DIM = 2

class particle:
    def __init__(self, dim):
        self.x = np.array([random.uniform(-3, 4) for _ in range(DIM)])
        self.v = np.array([random.uniform(-0.5, 0.5) for _ in range(DIM)])
        self.x_best = self.x

def particle_swarm_optimization(func, population, k_max, w=0.1, c1=0.25, c2=2.0):
    n = DIM
    x_best, y_best = np.copy(population[0].x_best), float("inf")
    for P in population:
        y = func(np.copy(P.x))
        if y < y_best:
            x_best = np.copy(P.x)
            y_best = y
    
    for k in range(k_max):
        for P in population:
            r1 = np.array([random.uniform(0, 1) for _ in range(DIM)])
            r2 = np.array([random.uniform(0, 1) for _ in range(DIM)])
            
            P.x = np.copy(P.x) + np.copy(P.v)
            P.v = w * np.copy(P.v) + c1 * r1 * (np.copy(P.x_best) - np.copy(P.x)) + c2 * r2 * (x_best - np.copy(P.x))
            
            y = func(np.copy(P.x))
            if y < y_best:
                x_best = np.copy(P.x)
                y_best = y
            if y < func(P.x_best):
                P.x_best = np.copy(P.x)
    
    return population

if __name__ == '__main__':
    func1 = lambda x: (x[0] - 2.25) ** 2 + (x[1] + 1.75) ** 2
    starting_points = [particle(DIM) for _ in range(10)]

    population1 = particle_swarm_optimization(func1, starting_points, 5, w=0.1, c1=0.25, c2=2.0)

    for p in population1:
        print (p.x, func1(p.x))
