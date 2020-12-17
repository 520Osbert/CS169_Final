import numpy as np
import random


class particle:
    def __init__(self, x, v):
        self.x = x
        self.v = v
        self.x_best = self.x

def particle_swarm_optimization(func, population, k_max, w=0.1, c1=0.25, c2=2.0):
    n = population[0].x.size
    shape = population[0].x.shape
    
    x_best, y_best = population[0].x_best.copy(), float("inf")
    for P in population:
        y = func(np.argsort(P.x))
        if y < y_best:
            x_best = P.x.copy()
            y_best = y
    
    for k in range(k_max):
        for P in population:
            r1 = np.random.uniform(0, 1, size=shape)
            r2 = np.random.uniform(0, 1, size=shape)
            
            P.x = np.copy(P.x) + np.copy(P.v)
            P.v = w * np.copy(P.v) + c1 * r1 * (np.copy(P.x_best) - np.copy(P.x)) + c2 * r2 * (x_best - np.copy(P.x))
            
            y = func(np.argsort(P.x))
            if y < y_best:
                x_best = P.x.copy()
                y_best = y
            if y < func(np.argsort(P.x_best)):
                P.x_best = P.x.copy()
    
    return population

if __name__ == '__main__':
    print ("main")
