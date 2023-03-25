import numpy as np
from sklearn.metrics import mean_squared_error
import random


class LinearRegression:

    # Create range of the genes that can be choice
    __mutation_sigma = 300

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.n_features = len(chromosome) - 1
        # self.fitness = self.cal_fitness()

    @classmethod
    def __mutated_genes(self):
        '''
        create random genes for mutation
        '''
        gene = np.random.normal(scale=self.__mutation_sigma)
        return gene

    @classmethod
    def spawn(self, n_features):
        '''
        create chromosome or string of genes
        '''
        return LinearRegression([self.__mutated_genes() for _ in range(n_features+1)])  # +1 (intercept)

    def mate(self, parent2):
        '''
        Perform mating and produce new offspring
        '''
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, parent2.chromosome):

            prob = random.random()

            if prob < 0.45:
                child_chromosome.append(gp1)
            elif prob < 0.90:
                child_chromosome.append(gp2)
            else:
                child_chromosome.append(self.__mutated_genes())

                # create new Individual(offspring) using
                # generated chromosome for offspring
        return LinearRegression(child_chromosome)

    def predict(self, X):
        params = np.array(self.chromosome)
        coef = params[:-1]
        intercept = params[-1]
        return np.dot(X, coef) + intercept

    def cal_fitness(self, X, y_true):
        y_pred = self.predict(X)
        return mean_squared_error(y_true, y_pred)

    def set_mutation_sigma(self, mutation_sigma):
        if mutation_sigma < 0:
            raise Exception(
                "mutation_sigma scale can not be number that less than Zero.")
        else:
            __mutation_sigma = mutation_sigma

    def __str__(self):
        return "  ".join([str(np.round(param, decimals=2)) for param in self.chromosome])
