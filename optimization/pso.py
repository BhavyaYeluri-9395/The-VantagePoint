import numpy as np

class PSO:

    def __init__(self, obj_func, dim, bounds, particles=30, iterations=100):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = bounds
        self.particles = particles
        self.iterations = iterations
        self.history = []

    def optimize(self):

        lb, ub = self.bounds

        X = np.random.uniform(lb, ub, (self.particles, self.dim))
        V = np.zeros_like(X)

        pbest = X.copy()
        pbest_scores = np.array([self.obj_func(x) for x in X])

        gbest = pbest[np.argmin(pbest_scores)]
        gbest_score = min(pbest_scores)

        for _ in range(self.iterations):

            r1, r2 = np.random.rand(), np.random.rand()

            V = 0.5*V + 1.5*r1*(pbest-X) + 1.5*r2*(gbest-X)
            X = X + V

            X = np.clip(X, lb, ub)

            scores = np.array([self.obj_func(x) for x in X])

            for i in range(self.particles):
                if scores[i] < pbest_scores[i]:
                    pbest[i] = X[i]
                    pbest_scores[i] = scores[i]

            if min(scores) < gbest_score:
                gbest = X[np.argmin(scores)]
                gbest_score = min(scores)

            self.history.append(gbest_score)

        return gbest, gbest_score