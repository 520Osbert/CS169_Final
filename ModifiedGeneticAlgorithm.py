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
    return model.get_end_time(seq)


def select(population, fitnesses, num):
    fitnesses = np.max(fitnesses) - fitnesses
    return [tuple(choices(population, weights=fitnesses, k=3)) for _ in range(num)]


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


def TwoPointcrossover(a, b):
    i = randint(0, len(a)-1)
    j = randint(0, len(a)-1)
    if i > j:
        i, j = j, i
    return np.hstack((a[:i], b[i:j], a[j:])), np.hstack((b[:i], a[i:j], b[j:]))


def OnePointcrossover(a, b):
    i = randint(0, len(a)-1)
    return np.hstack((a[:i], b[i:])), np.hstack((b[:i], a[i:]))


def multi_crossover(a, b, c, model, Rc):
    ''' Algorithm 1'''
    m = model.m_machine
    n = model.n_job
    ci = [a, b, c]
    for k in range(3):
        if k == 0:
            p1, p2 = a, b
        elif k == 1:
            p1, p2 = a, c
        else:
            p1, p2 = b, c
        fp1, fp2 = fitness_function_JSSP(p1, model), fitness_function_JSSP(p2, model)
        fi = np.inf
        for i in range(Rc):
            method = randint(0, 1)
            if method == 1:
                c1, c2 = TwoPointcrossover(p1, p2)
            else:
                c1, c2 = OnePointcrossover(p1, p2)
            c1, c2 = fix(c1, n, m), fix(c2, n, m)
            fc1, fc2 = fitness_function_JSSP(c1, model), fitness_function_JSSP(c2, model)
            if fc1 < fc2:
                fi = fc1
                ci[k] = c1
            elif fc2 < fc1:
                fi = fc2
                ci[k] = c2
            if fi < fp1 or fi < fp2:
                break
    return ci


def Swapmutate(child, i, j):
    child[i], child[j] = child[j], child[i]
    return child


def insertmutate(child, i, j):
    c = child[j]
    child = np.insert(np.delete(child, j), i, c)
    return child


def mutate(children, model, max_mutat, els=0.95):
    '''Algorithm 2'''
    ret = []
    func = Swapmutate if randint(0,1)==0 else insertmutate
    for i in range(len(children)):
        c = children[i]
        if random() > els:
            fc = fitness_function_JSSP(c, model)
            for i in range(max_mutat):
                r1 = randint(0, len(c)-1)
                r2 = randint(0, len(c)-1)
                while r1 == r2:
                    r2 = randint(0, len(c) - 1)
                cc = func(c, r1, r2)
                fcc = fitness_function_JSSP(cc, model)
                if fcc < fc:
                    c = cc
                    fc = fcc
        else:
            r1 = randint(0, len(c) - 1)
            r2 = randint(0, len(c) - 1)
            while r1 == r2:
                r2 = randint(0, len(c) - 1)
            c = func(c, r1, r2)
        ret.append(c)
    return ret


def improve_crossover(population, model):
    func = Swapmutate if randint(0,1)==0 else insertmutate
    N = model.m_machine*model.n_job
    for p in range(len(population)):
        fc = fitness_function_JSSP(population[p], model)
        c = population[p]
        for i in range(N):
            for j in range(N):
                cc = func(c, i, j)
                fcc = fitness_function_JSSP(cc, model)
                if fcc < fc:
                    c = cc
                    fc = fcc
        population[p] = c
    return population


def modified_genetic_algorithm(model, max_iter=50, pop_size=100, max_crossover=10, best_k=2):
    population = init_rand_population(pop_size, model.m_machine, model.n_job)
    best_fit = np.inf
    ys = []
    start = time()
    best = None
    fitnesses = np.array([fitness_function_JSSP(p, model) for p in population])
    for k in range(max_iter):
        if np.min(fitnesses) < best_fit:
            best_fit = np.min(fitnesses)
            best = population[np.argmin(fitnesses)]
        ys.append(best_fit)

        parents = select(population, fitnesses, pop_size//3)
        children = []
        for p in parents:
            children += multi_crossover(p[0], p[1], p[2], model, Rc=max_crossover)
        population = mutate(children, model, 10, els=0.95)
        fitnesses = np.array([fitness_function_JSSP(p, model) for p in population])
        best_index = np.argsort(fitnesses)[:best_k]
        best_children = [population[i] for i in best_index]
        best_children = improve_crossover(best_children, model)
        for i, j in enumerate(best_index):
            population[j] = best_children[i]
    return c_to_seq(best, model), [ys, time() - start]


if __name__ == '__main__':
    model = JSSP(10, 10, Processing_time=0, randopt=True)
    res, stats = modified_genetic_algorithm(model)
    print(res)
    print(model.get_end_time(res))
    print(stats[1])