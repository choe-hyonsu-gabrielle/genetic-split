import random
from itertools import permutations
import numpy as np
from scipy.special import softmax

random.seed(0)
np.random.seed(0)


class Config:
    def __init__(self, data_size=100):
        self.LENGTH_OF_CHROMOSOME = data_size       # dimension of encoding: data size to be split
        self.POPULATION_SIZE = 5000
        self.SPLIT_RATIO = 0.5                      # ratio of 0 and 1
        self.ELITES_RATE = 0.15
        self.FERTILE_PARENTS_RATE = 0.5
        self.REPRODUCTION_RATE = 3.0
        self.MUTATION_RATE = 0.05


class Individual:
    def __init__(self, fitness, chromosome):
        self.fitness = fitness
        self.chromosome = chromosome

    def __repr__(self):
        return f"<{self.__class__.__name__} - fitness: {self.fitness} | chromosome: {self.chromosome}>"


def reproduce(parents, reproduction_rate=2.0, mutation_rate=0.05):
    feasible_couples = list(permutations(parents, 2))
    target_births = int(len(parents) * reproduction_rate)
    offsprings = []
    while len(offsprings) < target_births:
        mom, dad = random.choice(feasible_couples)
        elder, younger = one_point_crossover(mom, dad, mutation_rate=mutation_rate)
        offsprings.extend([elder, younger])
    return offsprings


def one_point_crossover(parent_a, parent_b, mutation_rate=0.05):
    assert len(parent_a.chromosome) == len(parent_b.chromosome)
    split_point = random.randrange(len(parent_a.chromosome))
    new_seq_a = mutation(seq_a[:split_point] + seq_b[split_point:], rate=mutation_rate)
    new_seq_b = mutation(seq_b[:split_point] + seq_a[split_point:], rate=mutation_rate)
    child_a = Individual(fitness=0.0, chromosome=new_seq_a)
    child_b = Individual(fitness=0.0, chromosome=new_seq_b)
    return child_a, child_b


def mutation(seq, rate=0.05):
    mutant = list(seq)
    target_indices = random.sample(range(len(seq)), k=int(len(seq)*rate))
    for target in target_indices:
        mutant[target] = 1 if mutant[target] == 0 else 0
    return mutant


def ratio_loss_objectives(individual):
    # do something here for fitness of each individual
    ratio_loss = abs(self.config.SPLIT_RATIO - (sum(individual.chromosome) / len(individual.chromosome))) ** 2
    return -ratio_loss
    

class GeneticAlgorithm:
    def __init__(self, objectives, config):
        self.objectives = objectives
        self.config = config
        self.generation = 0
        self.fitness_sum = 0.0
        self.population = []

        # Population initialization
        for _ in range(self.config.POPULATION_SIZE):
            individual = Individual(0.0, [random.randint(0, 1) for _ in range(self.config.LENGTH_OF_CHROMOSOME)])
            self.population.append(individual)
        self.compute_fitness()

    def __repr__(self):
        return f"<{self.__class__.__name__} - Gen: {self.generation} | Pop: {len(self.population)} | FitSum: {self.fitness_sum}>"

    def evolve(self):
        self.natural_selection()    # decide next generation population following POPULATION_SIZE
        self.generation += 1
        self.fitness_sum = sum([individual.fitness for individual in self.population])
        print(self.__repr__())

    def compute_fitness(self):
        for individual in self.population:
            individual.fitness = self.objectives(individual.chromosome)
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

    def natural_selection(self):
        # selecting elites: they shall survive
        elites = self.population[:int(len(self.population) * self.config.ELITES_RATE)]
        # selecting reproductive parents including elites
        fertile_parents = list(np.random.choice(self.population,
                                                size=int(len(self.population) * self.config.FERTILE_PARENTS_RATE),
                                                p=softmax([individual.fitness ** 2 for individual in self.population])))
        # offspring from fertile parents
        offsprings = reproduce(fertile_parents,
                               reproduction_rate=self.config.REPRODUCTION_RATE,
                               mutation_rate=self.config.MUTATION_RATE)
        # Replacing population with next generation individuals
        candidates = random.sample(fertile_parents + offsprings, k=self.config.POPULATION_SIZE - len(elites))
        self.population = elites + candidates
        self.compute_fitness()


if __name__ == '__main__':
    config = Config(data_size=100)
    engine = GeneticAlgorithm(objectives=ratio_loss_objectives, config=config)

    # for rank, entity in enumerate(generation.population):
    #     print(rank, entity.fitness, sum(entity.chromosome), entity.chromosome)

    for _ in range(100):
        engine.evolve()

    # for rank, entity in enumerate(generation.population):
    #     print(rank, entity.fitness, sum(entity.chromosome), entity.chromosome)
