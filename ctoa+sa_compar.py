import numpy as np
import matplotlib.pyplot as plt

# Objective Function (Example: Sphere Function)
def objective_function(x):
    return np.sum(x**2)

# Simulated Annealing (SA) Algorithm
def SimulatedAnnealing(obj_func, pop_size, max_iter, dim, lower_bound, upper_bound, initial_temp=1000, cooling_rate=0.95):
    # Step 1: Initialize Population
    current_solution = np.random.uniform(low=lower_bound, high=upper_bound, size=dim)
    current_fitness = obj_func(current_solution)
    
    best_solution = np.copy(current_solution)
    best_fitness = current_fitness
    
    temperature = initial_temp
    
    for iteration in range(max_iter):
        # Generate a neighbor solution
        new_solution = current_solution + np.random.uniform(low=-1, high=1, size=dim)
        # Apply bounds
        new_solution = np.clip(new_solution, lower_bound, upper_bound)
        
        new_fitness = obj_func(new_solution)
        
        # Accept the new solution based on the acceptance probability
        if new_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - new_fitness) / temperature):
            current_solution = np.copy(new_solution)
            current_fitness = new_fitness
        
        # Update best solution
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = np.copy(current_solution)
        
        # Cool down
        temperature *= cooling_rate
        
        # Print the progress
        print(f"SA Iteration {iteration+1}/{max_iter}, Best Fitness: {best_fitness}")
    
    return best_solution, best_fitness

# CTOA Algorithm
def CTOA(obj_func, pop_size, max_iter, dim, lower_bound, upper_bound):
    # Step 1: Initialize Population
    population = np.random.uniform(low=lower_bound, high=upper_bound, size=(pop_size, dim))
    
    # Step 2: Evaluate Initial Population
    fitness = np.apply_along_axis(obj_func, 1, population)
    
    # Step 3: Main CTOA Loop
    best_solution = population[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    
    for iteration in range(max_iter):
        # Update each learner based on the topper
        topper = population[np.argmin(fitness)]
        
        for i in range(pop_size):
            # Exploration Phase: Move solution towards topper
            r = np.random.rand(dim)
            new_solution = population[i] + np.random.rand() * (topper - population[i]) * r
            # Apply bounds
            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            new_fitness = obj_func(new_solution)
            
            # Accept the new solution if it's better
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
        
        # Update best solution
        current_best_fitness = np.min(fitness)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[np.argmin(fitness)]
        
        # Print the progress
        print(f"CTOA Iteration {iteration+1}/{max_iter}, Best Fitness: {best_fitness}")
    
    return best_solution, best_fitness

# Parameters for both algorithms
pop_size = 50
max_iter = 100
dim = 2  # Number of dimensions (variables)
lower_bound = -5
upper_bound = 5

# Run Simulated Annealing (SA)
print("Running Simulated Annealing...")
best_solution_sa, best_fitness_sa = SimulatedAnnealing(objective_function, pop_size, max_iter, dim, lower_bound, upper_bound)

# Run CTOA
print("\nRunning CTOA...")
best_solution_ctoa, best_fitness_ctoa = CTOA(objective_function, pop_size, max_iter, dim, lower_bound, upper_bound)

# Define comparison criteria
criteria = ['Convergence Rate', 'Exploration Capability', 'Exploitation Ability', 'Complexity', 'Flexibility']
sa_scores = [8, 8, 7, 5, 7]  # Hypothetical scores for Simulated Annealing (SA)
ctoa_scores = [7, 8, 9, 6, 8]  # Hypothetical scores for CTOA

# Create a comparison bar plot
x = np.arange(len(criteria))
width = 0.35

# Create a figure and axes for the bar plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the bars for SA and CTOA
bars_sa = ax.bar(x - width/2, sa_scores, width, label='Simulated Annealing (SA)')
bars_ctoa = ax.bar(x + width/2, ctoa_scores, width, label='CTOA')

# Labeling
ax.set_xlabel('Criteria')
ax.set_ylabel('Scores')
ax.set_title('Comparison between Simulated Annealing (SA) and CTOA')
ax.set_xticks(x)
ax.set_xticklabels(criteria)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()

# Output best solutions and fitness values
print("\nBest Solution by SA:", best_solution_sa)
print("Best Fitness by SA:", best_fitness_sa)

print("\nBest Solution by CTOA:", best_solution_ctoa)
print("Best Fitness by CTOA:", best_fitness_ctoa)
