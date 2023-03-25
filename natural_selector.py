from linear_regression import LinearRegression
import random

def natural_selector(X, y, population_size, num_generations, length_of_convergent_list, mutation_sigma, fitness_goal):
    generation = 1

    # found = False
    population = []
    last_best_pop_fitnesses = []

    # Create initial population
    for _ in range(population_size):
        offspring = LinearRegression.spawn(1)
        offspring.set_mutation_sigma(mutation_sigma)
        population.append(offspring)

    for gen in range(num_generations):

        # X = X.reshape(X.shape[0], 1)
        population = sorted(population, key=lambda x: x.cal_fitness(X, y))

        fitness = population[0].cal_fitness(X, y)

        if fitness <= fitness_goal:
            break

        if len(last_best_pop_fitnesses) <= length_of_convergent_list:
            last_best_pop_fitnesses.append(fitness)
        else:
            last_best_pop_fitnesses = last_best_pop_fitnesses[1:]
            last_best_pop_fitnesses.append(fitness)

            if last_best_pop_fitnesses[0] == last_best_pop_fitnesses[length_of_convergent_list-1]:
                # found = True
                break

        new_generation = []

        # Perform Elitism, that mean 10% of fittest population
        # goes to the next generation
        s = int((10*population_size)/100)
        new_generation.extend(population[:s])

        # From 50% of fittest population, Individuals
        # will mate to produce offspring
        # crossover 90% individuals for next generation
        s = int((90*population_size)/100)
        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation

        print(
            f"Generation: {generation}\tParams: {population[0].chromosome}\tFitness: {fitness}")

        generation += 1

    print(
        f"Generation: {generation}\tParams: {population[0].chromosome}\tFitness: {fitness}")

    return population[0]
