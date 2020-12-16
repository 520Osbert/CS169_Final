import numpy as np
from time import time
from random import choices, random
from JobShopScheduling import JSSP
from copy import deepcopy


def init_rand_population(size, m, n):
    basic = [i for i in range(m) for _ in range(n)]
    return [np.random.permutation(basic) for _ in range(size)]


def fitness_function_JSSP(chromosome, model):
    m = model.m_machine
    n = model.n_job
    seq = np.zeros((n, m))
    job_end = np.zeros(m).astype(int)
    order_end = np.zeros(n).astype(int)
    for c in chromosome:
        job = job_end[c]
        order = order_end[job]
        seq[job, order] = c
        job_end[c] += 1
        order_end[job] += 1
    return model.get_end_time(seq)


def select(population, fitnesses, num):
    return [tuple(choices(population, weights=fitnesses, k=2)) for _ in num]

def crossover(a, b):
    child = deepcopy(a)
    for i in range(len(a)):
        if random() < 0.5:
            child[i] = b[i]
    return child


def mutate():
    pass


def genetic_algorithm(model, max_iter=1000, pop_size=10, select_rate=0.5, crossover_rate=0.8):
    population = init_rand_population(pop_size, model.m_machine, model.n_job)
    for k in range(max_iter):
        fitnesses = np.array([fitness_function_JSSP(p, model) for p in population])
        parents = select(population, fitnesses, int(select_rate*pop_size))
        children = [crossover(population[p[1]], population[p[2]])
                    for p in parents]
    # population = mutate()


if __name__ == '__main__':
    model = JSSP(2, 3, randopt=True)
    genetic_algorithm(model)