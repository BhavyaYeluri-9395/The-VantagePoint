import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from copy import deepcopy


class PSOOptimizer:

    def __init__(
        self,
        n_particles=30,
        n_iterations=10,
        inertia_weight=0.7,
        cognitive_coeff=1.5,
        social_coeff=1.5,
        validation_split=0.2,
        random_state=42
    ):

        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = inertia_weight
        self.c1 = cognitive_coeff
        self.c2 = social_coeff
        self.validation_split = validation_split
        self.random_state = random_state

        self.history = []

    def optimize(self, X, y, model):
        """
        Optimizes feature weights using PSO.
        """

        np.random.seed(self.random_state)

        n_features = X.shape[1]

        # Split once for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.validation_split,
            random_state=self.random_state
        )

        # Initialize particles
        particles = np.random.rand(self.n_particles, n_features)
        velocities = np.zeros((self.n_particles, n_features))

        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.n_particles, np.inf)

        global_best_position = None
        global_best_score = np.inf

        for iteration in range(self.n_iterations):

            for i in range(self.n_particles):

                weights = particles[i]

                X_train_weighted = X_train * weights
                X_val_weighted = X_val * weights

                # Use fresh model copy each time (avoid contamination)
                model_copy = deepcopy(model)

                model_copy.fit(X_train_weighted, y_train)
                predictions = model_copy.predict(X_val_weighted)

                mse = mean_squared_error(y_val, predictions)

                # Update personal best
                if mse < personal_best_scores[i]:
                    personal_best_scores[i] = mse
                    personal_best_positions[i] = weights.copy()

                # Update global best
                if mse < global_best_score:
                    global_best_score = mse
                    global_best_position = weights.copy()

            # Update velocity and position
            r1 = np.random.rand(self.n_particles, n_features)
            r2 = np.random.rand(self.n_particles, n_features)

            velocities = (
                self.w * velocities
                + self.c1 * r1 * (personal_best_positions - particles)
                + self.c2 * r2 * (global_best_position - particles)
            )

            particles = particles + velocities

            # Optional: keep weights in [0,1]
            particles = np.clip(particles, 0, 1)

            self.history.append(global_best_score)

            print(
                f"Iteration {iteration+1}/{self.n_iterations} "
                f"- Best Validation MSE: {global_best_score:.8f}"
            )

        return global_best_position