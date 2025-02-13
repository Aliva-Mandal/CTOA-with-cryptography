import time
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import numpy as np

# Sample CTOA Implementation for demonstration
def ctoa_optimize(objective_function, population_size, generations):
    np.random.seed(0)
    best_solution = None
    best_fitness = float('inf')
    
    # Initialize population
    population = np.random.uniform(-10, 10, (population_size, 2))
    
    for _ in range(generations):
        for i in range(population_size):
            fitness = objective_function(population[i])
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = population[i]
        
        # Update population
        for i in range(population_size):
            if np.random.rand() < 0.5:
                population[i] = best_solution + np.random.normal(0, 0.1, 2)
    
    return best_solution, best_fitness

# Sample objective function (minimization problem)
def objective_function(x):
    return x[0]**2 + x[1]**2

# Compare CTOA and RSA key generation
def compare_ctoa_rsa():
    # CTOA Parameters
    population_size = 50
    generations = 100
    
    # Measure CTOA performance
    start_time = time.time()
    best_solution, best_fitness = ctoa_optimize(objective_function, population_size, generations)
    ctoa_time = time.time() - start_time
    
    # RSA Key Generation
    start_time = time.time()
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    rsa_time = time.time() - start_time
    
    # Display results
    print("CTOA Optimization:")
    print(f"  Best Solution: {best_solution}")
    print(f"  Best Fitness: {best_fitness}")
    print(f"  Execution Time: {ctoa_time:.4f} seconds")
    
    print("\nRSA Key Generation:")
    print(f"  Key Size: {private_key.key_size} bits")
    print(f"  Execution Time: {rsa_time:.4f} seconds")

# Run comparison
compare_ctoa_rsa()
