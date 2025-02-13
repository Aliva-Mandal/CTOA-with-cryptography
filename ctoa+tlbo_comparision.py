import numpy as np
import matplotlib.pyplot as plt

# Objective Function (Example: Sphere Function)
def objective_function(x):
    return np.sum(x**2)

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

# TLBO Algorithm
def TLBO(obj_func, pop_size, max_iter, dim, lower_bound, upper_bound):
    # Step 1: Initialize Population
    population = np.random.uniform(low=lower_bound, high=upper_bound, size=(pop_size, dim))
    
    # Step 2: Evaluate Initial Population
    fitness = np.apply_along_axis(obj_func, 1, population)
    
    # Step 3: Main TLBO Loop
    best_solution = population[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    
    for iteration in range(max_iter):
        # Teacher Phase
        teacher = population[np.argmin(fitness)]
        
        for i in range(pop_size):
            # Exploration: Move solution towards the teacher
            r = np.random.rand(dim)
            new_solution = population[i] + np.random.rand() * (teacher - population[i]) * r
            # Apply bounds
            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            new_fitness = obj_func(new_solution)
            
            # Accept the new solution if it's better
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
        
        # Learner Phase
        for i in range(pop_size):
            # Learners update based on other learners
            j = np.random.randint(pop_size)
            while j == i:  # Ensure j is not equal to i
                j = np.random.randint(pop_size)
            
            r = np.random.rand(dim)
            new_solution = population[i] + np.random.rand() * (population[j] - population[i]) * r
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
        print(f"TLBO Iteration {iteration+1}/{max_iter}, Best Fitness: {best_fitness}")
    
    return best_solution, best_fitness

# Parameters for both algorithms
pop_size = 50
max_iter = 100
dim = 2  # Number of dimensions (variables)
lower_bound = -5
upper_bound = 5

# Run both algorithms and collect the best fitness scores
print("Running CTOA...")
best_solution_ctoa, best_fitness_ctoa = CTOA(objective_function, pop_size, max_iter, dim, lower_bound, upper_bound)

print("\nRunning TLBO...")
best_solution_tlbo, best_fitness_tlbo = TLBO(objective_function, pop_size, max_iter, dim, lower_bound, upper_bound)

# Define comparison criteria
criteria = ['Convergence Rate', 'Exploration Capability', 'Exploitation Ability', 'Complexity', 'Flexibility']
ctoa_scores = [7, 8, 9, 6, 8]  # Hypothetical scores for CTOA
tlbo_scores = [8, 7, 7, 5, 7]  # Hypothetical scores for TLBO

# Create a comparison bar plot
x = np.arange(len(criteria))
width = 0.35

# Create a figure and axes for the bar plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the bars for CTOA and TLBO
bars_ctoa = ax.bar(x - width/2, ctoa_scores, width, label='CTOA')
bars_tlbo = ax.bar(x + width/2, tlbo_scores, width, label='TLBO')

# Labeling
ax.set_xlabel('Criteria')
ax.set_ylabel('Scores')
ax.set_title('Comparison between CTOA and TLBO')
ax.set_xticks(x)
ax.set_xticklabels(criteria)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()

# Output best solutions and fitness values
print("\nBest Solution by CTOA:", best_solution_ctoa)
print("Best Fitness by CTOA:", best_fitness_ctoa)

print("\nBest Solution by TLBO:", best_solution_tlbo)
print("Best Fitness by TLBO:", best_fitness_tlbo)
