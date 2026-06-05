import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class MFParticleSwarmOptimizer:
    """
    PSO Optimizer for Fuzzy Membership Functions
    Optimizes the parameters of triangular membership functions
    """
    
    def __init__(self, fuzzy_system, n_particles=30, n_iterations=50,
                 inertia_weight=0.7, cognitive_coeff=1.5, social_coeff=1.5,
                 validation_split=0.2, random_state=42):
        
        self.fuzzy_system = fuzzy_system
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = inertia_weight
        self.c1 = cognitive_coeff
        self.c2 = social_coeff
        self.validation_split = validation_split
        self.random_state = random_state
        
        self.history = []
        self.best_params = None
        self.best_fitness = float('inf')
    
    def objective_function(self, params, X_train, y_train, X_val, y_val):
        """
        Objective function: Minimize MSE between fuzzy predictions and actual ratings
        """
        # Get parameter names
        _, param_names = self.fuzzy_system.get_membership_parameters()
        
        # Set membership parameters
        self.fuzzy_system.set_membership_parameters(params, param_names)
        
        # Compute fuzzy scores for validation set
        predictions = []
        for i in range(len(X_val)):
            budget = X_val[i, 0]  # Assuming budget is first column
            popularity = X_val[i, 2]  # Assuming popularity is third column
            fuzzy_score = self.fuzzy_system.compute_fuzzy_score(budget, popularity)
            predictions.append(fuzzy_score)
        
        predictions = np.array(predictions)
        
        # Calculate MSE
        mse = mean_squared_error(y_val, predictions)
        
        return mse
    
    def optimize(self, X, y):
        """
        Optimize membership function parameters using PSO
        """
        np.random.seed(self.random_state)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=self.random_state
        )
        
        # Get initial parameters and bounds
        initial_params, param_names = self.fuzzy_system.get_membership_parameters()
        n_params = len(initial_params)
        
        # Define bounds for parameters (based on variable ranges)
        bounds_low = []
        bounds_high = []
        
        for i, param_name in enumerate(param_names):
            if "budget" in param_name:
                if "p0" in param_name:  # a parameter (min)
                    bounds_low.append(0)
                    bounds_high.append(100000000)
                elif "p1" in param_name:  # b parameter (peak)
                    bounds_low.append(50000000)
                    bounds_high.append(300000000)
                else:  # p2 (max)
                    bounds_low.append(200000000)
                    bounds_high.append(500000000)
            elif "popularity" in param_name:
                bounds_low.append(0)
                bounds_high.append(100)
            else:
                bounds_low.append(0)
                bounds_high.append(10)
        
        # Initialize particles
        particles = np.array([
            initial_params + np.random.uniform(-0.1, 0.1, n_params) * initial_params
            for _ in range(self.n_particles)
        ])
        
        # Clip to bounds
        for i in range(self.n_particles):
            for j in range(n_params):
                particles[i, j] = np.clip(particles[i, j], bounds_low[j], bounds_high[j])
        
        velocities = np.zeros((self.n_particles, n_params))
        
        # Personal best
        pbest_positions = particles.copy()
        pbest_scores = np.array([
            self.objective_function(p, X_train, y_train, X_val, y_val)
            for p in particles
        ])
        
        # Global best
        gbest_idx = np.argmin(pbest_scores)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]
        
        print("\n🔹 Starting PSO Optimization for Membership Functions...")
        print(f"   Optimizing {n_params} parameters")
        
        for iteration in range(self.n_iterations):
            # Update velocity and position
            r1 = np.random.rand(self.n_particles, n_params)
            r2 = np.random.rand(self.n_particles, n_params)
            
            velocities = (self.w * velocities +
                         self.c1 * r1 * (pbest_positions - particles) +
                         self.c2 * r2 * (gbest_position - particles))
            
            particles = particles + velocities
            
            # Clip to bounds
            for i in range(self.n_particles):
                for j in range(n_params):
                    particles[i, j] = np.clip(particles[i, j], bounds_low[j], bounds_high[j])
            
            # Evaluate new positions
            scores = np.array([
                self.objective_function(p, X_train, y_train, X_val, y_val)
                for p in particles
            ])
            
            # Update personal bests
            improved = scores < pbest_scores
            pbest_positions[improved] = particles[improved]
            pbest_scores[improved] = scores[improved]
            
            # Update global best
            current_best_idx = np.argmin(scores)
            if scores[current_best_idx] < gbest_score:
                gbest_score = scores[current_best_idx]
                gbest_position = particles[current_best_idx].copy()
            
            self.history.append(gbest_score)
            
            if (iteration + 1) % 10 == 0:
                print(f"   Iteration {iteration+1}/{self.n_iterations} - Best MSE: {gbest_score:.6f}")
        
        # Save best parameters
        self.best_params = gbest_position
        self.best_fitness = gbest_score
        
        # Apply best parameters to fuzzy system
        self.fuzzy_system.set_membership_parameters(self.best_params, param_names)
        
        print(f"\n✅ PSO Optimization Complete!")
        print(f"   Best MSE: {gbest_score:.6f}")
        
        return self.best_params, self.best_fitness