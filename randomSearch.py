import numpy as np
import random
import math


def randomSearch(job, maxIter=None):
    '''
    random search for a given time, and find the best solutions
    '''
    i = 0
    solutions = []
    best_list = []
    best = math.inf
    while True:
        if i > maxIter:
            return solutions, best_list

        seq = job.generate_rand_seq()
        makespan = job.get_end_time(seq)

        if makespan < best:
            best = makespan
            best_list.append(best)
            solutions.append(seq)
        i += 1