import numpy as np
import random, operator
import pandas as pd
import matplotlib.pyplot as plt
import struct

MAXIMUM = 10
MINIMUM = -10
EQUATION = lambda x : np.multiply(np.power(x, 5), 9) - np.multiply(np.power(x, 4), 194.7) + np.multiply(np.power(x, 3), 1680.1) - np.multiply(np.power(x, 2), 7227.94) + np.multiply(x, 15501.2) - 13257.2
# EQUATION = lambda x : np.subtract(np.add(np.subtract(np.add(np.subtract(np.multiply(np.power(x, 5), 9), np.multiply(np.power(x, 4), 194.7)), np.multiply(np.power(x, 3), 1680.1)), np.multiply(np.power(x, 2), 7227.94)), np.multiply(x, 15501.2)), 13257.2)
np.seterr(over='ignore')
np.seterr(invalid='ignore')
def bin2float(b):
    ''' Convert binary string to a float.

    Attributes:
        :b: Binary string to transform.
    '''
    h = int(b, 2).to_bytes(8, byteorder="big")
    return struct.unpack('>d', h)[0]

def float2bin(f):
    ''' Convert float to 64-bit binary string.

    Attributes:
        :f: Float number to transform.
    '''
    [d] = struct.unpack(">Q", struct.pack(">d", f))
    return f'{d:064b}'

class Fitness:
    def __init__(self, value):
        self.equation = EQUATION
        self.value = value
        # self.result = self.equation(value)
        self.fitness = float(0)

    def compute_fitness(self):
        if self.fitness == 0:
            self.fitness = np.divide(1, abs(float(self.equation(self.value))))
        return self.fitness

def initial_population(pop_size):
    population = []
    for i in range(0, pop_size):
        population.append(np.random.uniform(MINIMUM, MAXIMUM))
    return population

def rank_by_fitness(population):
    fitness_results = {}
    for i in range(0,len(population)):
        fitness_results[i] = Fitness(population[i]).compute_fitness()
    return sorted(fitness_results.items(), key = operator.itemgetter(0), reverse = True)

def selection(pop_rank, elite_size):
    selection_results = []
    df = pd.DataFrame(np.array(pop_rank), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    for i in range(0, elite_size):
        selection_results.append(pop_rank[i][0])
    for i in range(0, len(pop_rank) - elite_size):
        pick_prob = 100 * random.random()
        for i in range(0, len(pop_rank)):
            if pick_prob <= df.iat[i,3]:
                selection_results.append(pop_rank[i][0])
                break
    return selection_results

def create_mating_pool(population, selection_results):
    mating_pool = []
    # print(type(selection_results[0]))
    for i in range(0, len(selection_results)):
        index = selection_results[i]
        mating_pool.append(population[index])
    return mating_pool


def breed(parent1, parent2):
    child = ''
    child_part1 = ''
    child_part2 = ''

    # p1 = format(abs(parent1), '07b')
    # p2 = format(abs(parent2), '07b')

    p1 = float2bin(parent1)
    p2 = float2bin(parent2)

    # print(p1)
    # print(p2)

    gene_a = np.random.randint(0,len(p1))
    gene_b = np.random.randint(0,len(p2))

    # print(gene_a)
    # print(gene_b)

    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    # print(start_gene)
    # print(end_gene)

    for i in range(0, start_gene):
        child += p2[i]
    for i in range(start_gene, end_gene):
        child += p1[i]
    for i in range(end_gene, len(p2)):
        child += p2[i]


    # print(child)
    # print(len(child))
    child = bin2float(child)
    return child

# print(breed(127,0))

def breed_population(mating_pool, elite_size):
    children = []
    length = len(mating_pool) - elite_size
    pool = random.sample(mating_pool, len(mating_pool))

    for i in range(0,elite_size):
        children.append(mating_pool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(mating_pool) - i - 1])
        children.append(child)
    return children

def mutate(individual, mutation_rate):
    ind = float2bin(individual)
    # print(ind)
    for swap1 in range(len(ind)):
        if(random.random() < mutation_rate):
            swap2 = np.random.randint(0,len(ind))

            # print(swap1, swap2)
            ind = swap(ind, swap1, swap2)
            # point1 = ind[swap1]
            # point2 = ind[swap2]
            #
            # ind[swap1] = point2
            # ind[swap2] = point1
    # print(ind)
    individual = bin2float(ind)
    return individual

def swap(string, i, j):
    c = list(string)
    c[i], c[j] = c[j], c[i]
    return ''.join(c)

# print(mutate(5, 0.2))


def mutate_population(population, mutation_rate):
    mutated_population = []

    for index in range(0, len(population)):
        mutated_individual = mutate(population[index], mutation_rate)
        mutated_population.append(mutated_individual)
    return mutated_population

def next_generation(current_gen, elite_size, mutation_rate):
    elite_fitness = rank_by_fitness(current_gen)
    selection_results = selection(elite_fitness, elite_size)
    mating_pool = create_mating_pool(current_gen, selection_results)
    children = breed_population(mating_pool, elite_size)
    next_gen = mutate_population(children, mutation_rate)
    return next_gen

def GA(pop_size, elite_size, mutation_rate, no_of_generations):
    pop = initial_population(pop_size)
    print("Initial error: " + str(1 / rank_by_fitness(pop)[0][1]))

    progress = []
    progress.append(1 / rank_by_fitness(pop)[0][1])

    for i in range(0, no_of_generations):
        pop = next_generation(pop, elite_size, mutation_rate)
        progress.append(1 / rank_by_fitness(pop)[0][1])

    print("Final error: " + str(1 / rank_by_fitness(pop)[0][1]))

    best_val_index = rank_by_fitness(pop)[0][0]
    best_val = pop[best_val_index]
    print("Best root: " + str(best_val))

    plt.plot(progress)
    plt.ylabel('Error')
    plt.xlabel('Generation')
    plt.show()

    return best_val

GA(pop_size=100, elite_size=20, mutation_rate=0.01, no_of_generations=50)
