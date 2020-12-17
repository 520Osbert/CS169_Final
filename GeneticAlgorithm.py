import numpy as np
from time import time
from random import choices, random, randint
from JobShopScheduling import JSSP
from copy import deepcopy


def init_rand_population(size, m, n):
    basic = [i for i in range(m) for _ in range(n)]
    return [np.random.permutation(basic) for _ in range(size)]


def c_to_seq(chromosome, model):
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
    return seq

def fitness_function_JSSP(chromosome, model):
    seq = c_to_seq(chromosome, model)
    return model.schedule_efficiency(seq)


def select(population, fitnesses, num):
    fitnesses = np.max(fitnesses) - fitnesses
    return [tuple(choices(population, weights=fitnesses, k=2)) for _ in range(num)]


def fix(c, n, m):
    count = np.zeros(m).astype(int)
    for i in range(len(c)):
        count[c[i]] += 1
    less = [i for i in range(m) for _ in range(n-count[i]) if count[i] < n]
    for i in range(len(c)):
        if count[c[i]] > n:
            count[c[i]] -= 1
            c[i] = less.pop()
    return c


def Uniformcrossover(a, b, cross_rate=0.5):
    child = deepcopy(a)
    for i in range(len(a)):
        if random() < cross_rate:
            child[i] = b[i]
    return child


def TwoPointcrossover(a, b):
    i = randint(0, len(a)-1)
    j = randint(0, len(a)-1)
    if i > j:
        i, j = j, i
    child = np.hstack((a[:i], b[i:j], a[j:]))
    return child


def OnePointcrossover(a, b):
    i = randint(0, len(a)-1)
    child = np.hstack((a[:i], b[i:]))
    return child


def crossover(method, a, b, n, m, cross_rate=0.5):
    if method == 'Uniform':
        child = Uniformcrossover(a, b, cross_rate)
    elif method == 'TwoPoint':
        child = TwoPointcrossover(a, b)
    else:
        child = OnePointcrossover(a, b)
    child = fix(child, n, m)
    return child


def mutate(child, mutation_rate):
    for i in range(len(child)):
        if random() > mutation_rate:
            j = randint(0, len(child)-1)
            child[i], child[j] = child[j], child[i]
    return child



def genetic_algorithm(model, max_iter=1000, pop_size=1000,  mutation_rate=0.8, cross_method='OnePoint', cross_rate=0.5):
    population = init_rand_population(pop_size, model.m_machine, model.n_job)
    best_fit = np.inf
    ys = []
    start = time()
    best = None
    for k in range(max_iter):
        fitnesses = np.array([fitness_function_JSSP(p, model) for p in population])

        if np.min(fitnesses) < best_fit:
            best_fit = np.min(fitnesses)
            best = population[np.argmin(fitnesses)]
        ys.append(best_fit)

        parents = select(population, fitnesses, pop_size)
        children = [crossover(cross_method, p[0], p[1], model.n_job, model.m_machine, cross_rate) for p in parents]
        population = [mutate(child, mutation_rate) for child in children]
    return c_to_seq(best, model), [ys, time() - start]

if __name__ == '__main__':
    model = JSSP(2, 5, Processing_time=0, randopt=True)
    res, stats = genetic_algorithm(model, max_iter=1, pop_size=4)
    print(res)
    print(model.get_end_time(res))