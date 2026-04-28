"""
Advanced Optimization Module
Gradient descent, Newton's method, genetic algorithms, particle swarm, etc.
"""

import math
import random
from typing import Callable, List, Tuple, Optional


class GradientDescent:
    """Gradient Descent optimization."""
    
    @staticmethod
    def optimize(f: Callable[[float], float], x0: float, 
                learning_rate: float = 0.01, max_iterations: int = 1000,
                tolerance: float = 1e-6) -> Tuple[float, float]:
        """Minimize function using gradient descent.
        
        Args:
            f: Function to minimize
            x0: Initial point
            learning_rate: Step size
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of (x_min, f_min)
        """
        x = x0
        h = 1e-5
        
        for i in range(max_iterations):
            # Numerical gradient
            grad = (f(x + h) - f(x - h)) / (2 * h)
            
            x_new = x - learning_rate * grad
            
            if abs(x_new - x) < tolerance:
                break
            
            x = x_new
        
        return (x, f(x))
    
    @staticmethod
    def optimize_adaptive(f: Callable[[float], float], x0: float,
                         initial_lr: float = 0.1, max_iterations: int = 1000,
                         tolerance: float = 1e-6) -> Tuple[float, float]:
        """Gradient descent with adaptive learning rate."""
        x = x0
        lr = initial_lr
        h = 1e-5
        
        for i in range(max_iterations):
            f_prev = f(x)
            grad = (f(x + h) - f(x - h)) / (2 * h)
            
            x_new = x - lr * grad
            f_new = f(x_new)
            
            # Adjust learning rate
            if f_new < f_prev:
                lr *= 1.1  # Increase learning rate
            else:
                lr *= 0.5  # Decrease learning rate
            
            if abs(x_new - x) < tolerance:
                break
            
            x = x_new
        
        return (x, f(x))


class NewtonsMethod:
    """Newton's method for optimization."""
    
    @staticmethod
    def optimize(f: Callable[[float], float], x0: float,
                max_iterations: int = 100, tolerance: float = 1e-10) -> Tuple[float, float]:
        """Optimize using Newton's method on f'(x) = 0.
        
        Args:
            f: Function to minimize (finds where f'(x) = 0)
            x0: Initial point
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of (x_optimal, f_optimal)
        """
        x = x0
        h = 1e-5
        
        for _ in range(max_iterations):
            # First derivative (gradient)
            f_prime = (f(x + h) - f(x - h)) / (2 * h)
            
            # Second derivative (Hessian)
            f_double_prime = (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
            
            if abs(f_double_prime) < 1e-15:
                break
            
            x_new = x - f_prime / f_double_prime
            
            if abs(x_new - x) < tolerance:
                break
            
            x = x_new
        
        return (x, f(x))


class ConjugateGradient:
    """Conjugate Gradient method for multidimensional optimization."""
    
    @staticmethod
    def optimize(f: Callable[[List[float]], float], x0: List[float],
                max_iterations: int = 100, tolerance: float = 1e-6) -> Tuple[List[float], float]:
        """Minimize multidimensional function using Conjugate Gradient.
        
        Args:
            f: Function to minimize (takes list of floats)
            x0: Initial point
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of (x_optimal, f_optimal)
        """
        x = x0[:]
        h = 1e-5
        n = len(x)
        
        # Compute initial gradient
        grad = []
        for i in range(n):
            x_plus = x[:]
            x_plus[i] += h
            x_minus = x[:]
            x_minus[i] -= h
            grad.append((f(x_plus) - f(x_minus)) / (2 * h))
        
        d = [-g for g in grad]  # Initial direction
        
        for iteration in range(max_iterations):
            # Line search
            alpha = 0.01
            x_new = [x[i] + alpha * d[i] for i in range(n)]
            
            # Compute new gradient
            grad_new = []
            for i in range(n):
                x_plus = x_new[:]
                x_plus[i] += h
                x_minus = x_new[:]
                x_minus[i] -= h
                grad_new.append((f(x_plus) - f(x_minus)) / (2 * h))
            
            # Check convergence
            grad_norm = math.sqrt(sum(g**2 for g in grad_new))
            if grad_norm < tolerance:
                break
            
            # Compute beta (Polak-Ribiere)
            denom = sum(g**2 for g in grad)
            if abs(denom) < 1e-15:
                break
            
            numerator = sum((grad_new[i] - grad[i]) * grad_new[i] for i in range(n))
            beta = numerator / denom
            
            # Update direction
            d = [-grad_new[i] + beta * d[i] for i in range(n)]
            
            x = x_new
            grad = grad_new
        
        return (x, f(x))


class SimulatedAnnealing:
    """Simulated Annealing optimization."""
    
    @staticmethod
    def optimize(f: Callable[[float], float], x0: float, 
                temp_initial: float = 100, cooling_rate: float = 0.95,
                max_iterations: int = 10000) -> Tuple[float, float]:
        """Minimize function using Simulated Annealing.
        
        Args:
            f: Function to minimize
            x0: Initial point
            temp_initial: Initial temperature
            cooling_rate: Temperature decay rate
            max_iterations: Maximum iterations
            
        Returns:
            Tuple of (x_optimal, f_optimal)
        """
        x = x0
        f_x = f(x)
        
        x_best = x
        f_best = f_x
        
        temperature = temp_initial
        
        for iteration in range(max_iterations):
            # Generate neighbor
            delta = random.gauss(0, 1)
            x_new = x + delta
            f_x_new = f(x_new)
            
            # Acceptance criterion
            delta_f = f_x_new - f_x
            if delta_f < 0 or random.random() < math.exp(-delta_f / temperature):
                x = x_new
                f_x = f_x_new
                
                if f_x < f_best:
                    x_best = x
                    f_best = f_x
            
            # Cool down
            temperature *= cooling_rate
        
        return (x_best, f_best)


class ParticleSwarmOptimization:
    """Particle Swarm Optimization (PSO)."""
    
    @staticmethod
    def optimize(f: Callable[[List[float]], float], bounds: List[Tuple[float, float]],
                num_particles: int = 30, max_iterations: int = 100,
                w: float = 0.7, c1: float = 1.5, c2: float = 1.5) -> Tuple[List[float], float]:
        """Minimize multidimensional function using PSO.
        
        Args:
            f: Function to minimize
            bounds: List of (min, max) for each dimension
            num_particles: Number of particles
            max_iterations: Maximum iterations
            w, c1, c2: PSO parameters
            
        Returns:
            Tuple of (x_optimal, f_optimal)
        """
        n_dims = len(bounds)
        
        # Initialize particles
        particles = []
        velocities = []
        best_positions = []
        best_scores = []
        
        for _ in range(num_particles):
            pos = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(n_dims)]
            vel = [random.uniform(-1, 1) for _ in range(n_dims)]
            
            particles.append(pos)
            velocities.append(vel)
            best_positions.append(pos[:])
            best_scores.append(f(pos))
        
        # Global best
        best_idx = best_scores.index(min(best_scores))
        global_best = best_positions[best_idx][:]
        global_best_score = best_scores[best_idx]
        
        for iteration in range(max_iterations):
            for i in range(num_particles):
                # Update velocity
                for d in range(n_dims):
                    r1, r2 = random.random(), random.random()
                    velocities[i][d] = (
                        w * velocities[i][d] +
                        c1 * r1 * (best_positions[i][d] - particles[i][d]) +
                        c2 * r2 * (global_best[d] - particles[i][d])
                    )
                
                # Update position
                for d in range(n_dims):
                    particles[i][d] += velocities[i][d]
                    # Enforce bounds
                    particles[i][d] = max(bounds[d][0], min(bounds[d][1], particles[i][d]))
                
                # Evaluate
                score = f(particles[i])
                if score < best_scores[i]:
                    best_positions[i] = particles[i][:]
                    best_scores[i] = score
                    
                    if score < global_best_score:
                        global_best = particles[i][:]
                        global_best_score = score
        
        return (global_best, global_best_score)


class GeneticAlgorithm:
    """Genetic Algorithm for optimization."""
    
    @staticmethod
    def optimize(f: Callable[[List[float]], float], bounds: List[Tuple[float, float]],
                population_size: int = 50, generations: int = 100,
                mutation_rate: float = 0.1) -> Tuple[List[float], float]:
        """Minimize function using Genetic Algorithm.
        
        Args:
            f: Function to minimize (fitness = -f for maximization)
            bounds: List of (min, max) for each dimension
            population_size: Population size
            generations: Number of generations
            mutation_rate: Mutation probability
            
        Returns:
            Tuple of (best_individual, best_fitness)
        """
        n_dims = len(bounds)
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = [random.uniform(bounds[i][0], bounds[i][1]) 
                         for i in range(n_dims)]
            population.append(individual)
        
        for generation in range(generations):
            # Evaluate fitness
            fitness = [f(ind) for ind in population]
            
            # Selection (tournament)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 3
                indices = random.sample(range(population_size), tournament_size)
                winner_idx = min(indices, key=lambda i: fitness[i])
                new_population.append(population[winner_idx][:])
            
            # Crossover and mutation
            offspring = []
            for i in range(0, population_size, 2):
                parent1 = new_population[i]
                parent2 = new_population[(i + 1) % population_size]
                
                # Single point crossover
                crossover_point = random.randint(1, n_dims - 1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
                
                # Mutation
                for child in [child1, child2]:
                    for d in range(n_dims):
                        if random.random() < mutation_rate:
                            child[d] = random.uniform(bounds[d][0], bounds[d][1])
                
                offspring.extend([child1, child2])
            
            population = offspring[:population_size]
        
        # Return best
        fitness = [f(ind) for ind in population]
        best_idx = fitness.index(min(fitness))
        return (population[best_idx], fitness[best_idx])
