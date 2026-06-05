import numpy as np

class GeneticAlgorithm:

    def __init__(self, obj_func, dim, bounds, pop_size=30, generations=100):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations

    def optimize(self):

        population = np.random.uniform(
            self.bounds[0],
            self.bounds[1],
            (self.pop_size, self.dim)
        )

        for _ in range(self.generations):

            fitness = np.array([self.obj_func(ind) for ind in population])

            selected = population[np.argsort(fitness)[:self.pop_size//2]]

            children = []
            for i in range(len(selected)//2):
                child = (selected[i] + selected[-i-1]) / 2
                children.append(child)

            population = np.vstack((selected, children))

        best = min(population, key=self.obj_func)
        return best, self.obj_func(best)