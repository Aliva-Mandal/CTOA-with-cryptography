import matplotlib.pyplot as plt
import numpy as np

# Criteria for comparison
criteria = ['Convergence Speed', 'Solution Quality', 'Parameter Sensitivity', 'Computational Cost', 'Scalability']

# Sample scores for CTOA and PSO (on a scale from 1 to 10)
ctoa_scores = [8, 9, 7, 6, 8]
pso_scores = [7, 8, 6, 7, 7]

# Position of bars on X-axis
ind = np.arange(len(criteria))
width = 0.35  # Width of the bars

# Plotting the data
fig, ax = plt.subplots(figsize=(10, 6))

# Bar plots for CTOA and PSO
ctoa_bars = ax.bar(ind - width/2, ctoa_scores, width, label='CTOA', color='blue')
pso_bars = ax.bar(ind + width/2, pso_scores, width, label='PSO', color='green')

# Adding labels, title, and legend
ax.set_xlabel('Criteria')
ax.set_ylabel('Scores')
ax.set_title('Comparison of CTOA and PSO based on Different Criteria')
ax.set_xticks(ind)
ax.set_xticklabels(criteria)
ax.legend()

# Displaying the plot
plt.tight_layout()
plt.show()
