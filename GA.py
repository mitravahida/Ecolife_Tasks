import random
import utils
import numpy as np

class GA:
    def __init__(self, parameters, server_pair, function_name, ci_avg, ci, intervals):
        self.parameters = parameters
        self.server_pair = server_pair
        self.function_name = function_name
        self.ci_avg = ci_avg
        self.ci = ci
        self.intervals = intervals
        self.population_size = 10
        self.generations = 10
        self.mutation_rate = 0.1
        self.selection_size = 3
        self.lam = parameters[2]

        # Compute max values for normalization in fitness function
        old_cold, _ = utils.get_st(function_name, server_pair[0])
        new_cold, _ = utils.get_st(function_name, server_pair[1])
        cold_carbon_max, _ = utils.compute_exe(function_name, server_pair, ci_avg)
        self.max_st = max(old_cold, new_cold)
        self.max_carbon_st = max(cold_carbon_max)
        self.max_carbon_kat = max(utils.compute_kat(function_name, server_pair[0], 1, ci_avg),
                                  utils.compute_kat(function_name, server_pair[1], 1, ci_avg))

    def create_population(self):
        # chromosome is represented as value of [ka_loc, k_last] to be consistent with pso algo
        population = [(random.randint(0, 1), random.choice(range(1, 25))) for _ in range(self.population_size)]
        return population

    def ga_fitness(self, chromosome):
        # I reused pso fitness function with a minor change in intervals
        ka_loc, ka_last = chromosome

        # kat carbon
        score = 0
        old_kat_carbon = utils.compute_kat(self.function_name, self.server_pair[0], ka_last, self.ci)
        new_kat_carbon = utils.compute_kat(self.function_name, self.server_pair[1], ka_last, self.ci)
        cold_carbon, warm_carbon = utils.compute_exe(self.function_name, self.server_pair, self.ci)

        old_st = utils.get_st(self.function_name, self.server_pair[0])
        new_st = utils.get_st(self.function_name, self.server_pair[1])
        score += (1 - self.lam) * (((1 - ka_loc) * old_kat_carbon + ka_loc * new_kat_carbon) / self.max_carbon_kat)

        cold_prob, warm_prob = self.prob_cold(self.intervals, ka_last)
        part_time_prob = cold_prob * ((1 - ka_loc) * old_st[0] + ka_loc * new_st[0]) + warm_prob * (
                (1 - ka_loc) * old_st[1] + ka_loc * new_st[1])
        part_carbon_prob = cold_prob * ((1 - ka_loc) * cold_carbon[0] + ka_loc * cold_carbon[1]) + warm_prob * (
                (1 - ka_loc) * warm_carbon[0] + ka_loc * warm_carbon[1])
        score += self.lam * (part_time_prob) / self.max_st
        score += (1 - self.lam) * (part_carbon_prob) / (self.max_carbon_st)
        return score

    def prob_cold(self, cur_interval, kat):
        # this is also reused from pso algo
        if len(self.intervals) == 0:
            # no invocation
            return 0.5, 0.5
        else:
            cold = 0
            warm = 0
            for interval in cur_interval:
                if interval <= kat:
                    # hit
                    warm += 1
                else:
                    cold += 1
            return cold / (cold + warm), warm / (cold + warm)

    def parent_selection(self, fitness_scores, population):
        parents = []
        for _ in range(self.population_size):
            candidate = random.sample(list(enumerate(fitness_scores)), self.selection_size)
            selected = min(candidate, key=lambda x: x[1])[0]
            parents.append(population[selected])
        return parents

    def crossover(self, parent1, parent2):
        if random.random() < 0.5:
            return (parent1[0], parent2[1]), (parent2[0], parent1[1])
        else:
            return parent1, parent2

    def mutate(self, chromosome):
        ka_loc, ka_last = chromosome
        if random.random() < self.mutation_rate:
            ka_loc = 1 - ka_loc
        if random.random() < self.mutation_rate:
            ka_last = random.choice(range(1, 25))
        return ka_loc, ka_last

    def main(self, ci, interval):
        self.ci = ci
        self.interval = interval

        population = self.create_population()
        for generation in range(self.generations):
            fitness_scores = [self.ga_fitness(chromosome) for chromosome in population]
            selected_parents = self.parent_selection(fitness_scores, population)

            offsprings = []
            for i in range(0, self.population_size, 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[(i + 1) % self.population_size]
                child1, child2 = self.crossover(parent1, parent2)
                offsprings.extend([child1, child2])
            population = [self.mutate(chromosome) for chromosome in offsprings]

        new_fitness_scores = [self.ga_fitness(chromosome) for chromosome in population]
        best_solution = population[np.argmin(new_fitness_scores)]
        return best_solution, min(new_fitness_scores)
