import numpy as np
import random
import math
import time
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


def simulatedAnnealing(job, T, maxIter, halting, decrease, mode="normal"):
    seq = job.generate_rand_seq()

    for i in range(halting):
        T = decrease * float(T)

        for j in range(maxIter):
            cost = job.get_end_time(seq)

            nlist = findNeighbours(seq)

            if mode == "random":
                k = random.choice(nlist)
                k_cost = job.get_end_time(k)
                if k_cost < cost:
                    seq = k
                    cost = k_cost
                else:
                    p = math.exp(-k_cost / T)
                    if random.random() < p:
                        seq = k
                        cost = k_cost

            elif mode == "normal":
                for k in nlist:
                    k_cost = job.get_end_time(k)
                    if k_cost < cost:
                        seq = k
                        cost = k_cost
                    else:
                        p = math.exp(-k_cost / T)
                        if random.random() < p:
                            seq = k
                            cost = k_cost

    return cost, seq, T


def SAsearch(job, loopcount=100, T=200, maxIter=10, halting=10, decrease=0.8, mode="normal"):
    costs = []
    sols = []
    best = math.inf
    best_seq = None
    # start = time.time()

    seq = job.generate_rand_seq()
    for i in range(loopcount):
        cost, seq, _ = simulatedAnnealing(job, T, maxIter, halting, decrease)
        if cost < best:
            best = cost
            best_seq = seq
        costs.append(best)

    return best, best_seq, costs