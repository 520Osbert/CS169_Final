import numpy as np
import random
import math
from JobShopScheduling import *


def findNeighbours(seq):
    nbrs = []
    M,N = seq.shape
    for i in range(M):
        for j in range(N - 1):
            nbr = seq[:]
            nbr[i][j], nbr[i][j+1] = nbr[i][j+1], nbr[i][j]
            nbrs.append(nbr)
    return nbrs


def simulatedAnnealing(job, T, maxIter, halting, decrease):
    seq = job.generate_rand_seq()

    for i in range(halting):
        T = decrease * float(T)

        for j in range(maxIter):
            cost = job.get_end_time(seq)

            for k in findNeighbours(seq):
                k_cost = job.get_end_time(k)
                if k_cost < cost:
                    seq = k
                    cost = k_cost
                else:
                    p = math.exp(-k_cost / T)
                    if random.random() < p:
                        seq = k
                        cost = k_cost
    return cost, seq