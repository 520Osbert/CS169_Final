import numpy as np


def generate_rand_seq(m, n):
    return np.array([np.random.permutation(m) for _ in range(n)])

def init_rand_population(size, m, n):
    return [generate_rand_seq(m,n) for _ in range(size)]

def select():
    pass

def crossover():
    pass

def mutate():
    pass

def genetic_algorithm(f, m_machine, n_job, max_iter):
    population = init_rand_population(m_machine, n_job)
    for k in range(max_iter):
        parents = select(f, population)
        # children = [crossover(C, population[p[1]], population[p[2]])
        #             for p in parents]
        # population = mutate()

