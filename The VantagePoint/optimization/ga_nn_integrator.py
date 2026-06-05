import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class GANeuralNetworkIntegrator:
    """
    Genetic Algorithm for Neural Network Weight Optimization
    Integrates GA with Neural Network training
    """
    
    def __init__(self, base_model, population_size=50, generations=30,
                 mutation_rate=0.1, crossover_rate=0.8, validation_split=0.2,
                 random_state=42):
        
        self.base_model = base_model
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.validation_split = validation_split
        self.random_state = random_state
        
        self.best_weights = None
        self.best_fitness = float('inf')
        self.history = []
    
    def extract_weights(self, model):
        """
        Extract weights from neural network model
        Supports sklearn and custom models
        """
        weights = []
        
        if hasattr(model, 'coef_'):
            # Linear models
            weights.extend(model.coef_.flatten())
            if hasattr(model, 'intercept_'):
                weights.append(model.intercept_)
        elif hasattr(model, 'feature_importances_'):
            # Tree-based models
            weights.extend(model.feature_importances_)
        elif hasattr(model, 'get_params'):
            # For other models, use coefficients if available
            if hasattr(model, 'estimators_'):
                for est in model.estimators_:
                    if hasattr(est, 'coef_'):
                        weights.extend(est.coef_.flatten())
        
        return np.array(weights) if weights else np.random.randn(10)
    
    def set_weights(self, model, weights):
        """
        Set weights to neural network model
        """
        try:
            if hasattr(model, 'coef_'):
                # For linear models
                n_features = model.coef_.shape[0]
                model.coef_ = weights[:n_features].reshape(model.coef_.shape)
                if hasattr(model, 'intercept_') and len(weights) > n_features:
                    model.intercept_ = weights[-1]
            elif hasattr(model, 'feature_importances_'):
                # For tree models
                n_features = len(model.feature_importances_)
                model.feature_importances_ = weights[:n_features]
            elif hasattr(model, 'estimators_'):
                # For ensemble models
                idx = 0
                for est in model.estimators_:
                    if hasattr(est, 'coef_'):
                        n_feat = est.coef_.shape[0]
                        est.coef_ = weights[idx:idx+n_feat].reshape(est.coef_.shape)
                        idx += n_feat
            return True
        except Exception as e:
            print(f"Warning: Could not set weights: {e}")
            return False
    
    def initialize_population(self, initial_weights):
        """
        Initialize population with variations of initial weights
        """
        population = []
        n_params = len(initial_weights)
        
        for _ in range(self.population_size):
            # Add random noise to initial weights
            noise = np.random.normal(0, 0.1, n_params)
            individual = initial_weights + noise
            population.append(individual)
        
        return np.array(population)
    
    def fitness_function(self, weights, model, X_train, y_train, X_val, y_val):
        """
        Calculate fitness (negative MSE for maximization)
        """
        model_copy = deepcopy(model)
        self.set_weights(model_copy, weights)
        
        try:
            model_copy.fit(X_train, y_train)
            predictions = model_copy.predict(X_val)
            mse = mean_squared_error(y_val, predictions)
            fitness = -mse  # Negative because we want to minimize MSE
        except:
            fitness = -float('inf')
        
        return fitness
    
    def selection(self, population, fitness_scores):
        """
        Tournament selection for parent selection
        """
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Tournament selection
            tournament_indices = np.random.choice(
                len(population), tournament_size, replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return np.array(selected)
    
    def crossover(self, parent1, parent2):
        """
        Uniform crossover with probability crossover_rate
        """
        if np.random.random() < self.crossover_rate:
            mask = np.random.random(len(parent1)) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutation(self, individual, mutation_strength=0.1):
        """
        Gaussian mutation
        """
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                individual[i] += np.random.normal(0, mutation_strength)
        return individual
    
    def optimize(self, X, y, model=None):
        """
        Main GA optimization loop
        """
        np.random.seed(self.random_state)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=self.random_state
        )
        
        # Get initial weights
        target_model = model or self.base_model
        initial_weights = self.extract_weights(target_model)
        n_params = len(initial_weights)
        
        print(f"\n🧬 Starting GA Optimization for Neural Network Weights...")
        print(f"   Optimizing {n_params} parameters")
        print(f"   Population Size: {self.population_size}")
        print(f"   Generations: {self.generations}")
        
        # Initialize population
        population = self.initialize_population(initial_weights)
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = np.array([
                self.fitness_function(ind, target_model, X_train, y_train, X_val, y_val)
                for ind in population
            ])
            
            # Track best individual
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_individual = population[best_idx].copy()
            
            if -best_fitness < self.best_fitness:
                self.best_fitness = -best_fitness
                self.best_weights = best_individual.copy()
            
            self.history.append(-best_fitness)
            
            # Selection
            selected_population = self.selection(population, fitness_scores)
            
            # Crossover
            new_population = []
            for i in range(0, len(selected_population), 2):
                if i + 1 < len(selected_population):
                    child1, child2 = self.crossover(
                        selected_population[i], selected_population[i+1]
                    )
                    new_population.extend([child1, child2])
                else:
                    new_population.append(selected_population[i])
            
            # Mutation
            population = np.array([self.mutation(ind) for ind in new_population])
            
            # Elitism: keep best individual
            population[0] = best_individual
            
            if (generation + 1) % 5 == 0:
                print(f"   Generation {generation+1}/{self.generations} - Best MSE: {self.history[-1]:.6f}")
        
        # Apply best weights to model
        self.set_weights(target_model, self.best_weights)
        
        print(f"\nGA Optimization Complete!")
        print(f"   Best MSE: {self.best_fitness:.6f}")
        
        return self.best_weights, self.best_fitness