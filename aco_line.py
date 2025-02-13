import matplotlib.pyplot as plt
import numpy as np

# Criteria for comparison
criteria = ['Convergence Speed', 'Solution Quality', 'Flexibility', 'Scalability', 'Robustness']

# Hypothetical scores for each criterion (out of 10)
aco_scores = [7, 8, 6, 7, 9]  # Ant Colony Optimization
ctoa_scores = [8, 9, 7, 8, 8]  # Class Topper Optimization Algorithm

# Positions of the criteria (x-axis values)
index = np.arange(len(criteria))

# Plotting the lines
fig, ax = plt.subplots()
ax.plot(index, aco_scores, marker='o', label='ACO', color='b')
ax.plot(index, ctoa_scores, marker='o', label='CTOA', color='g')

# Adding labels, title, and custom x-axis tick labels
ax.set_xlabel('Criteria')
ax.set_ylabel('Scores')
ax.set_title('Comparison of ACO and CTOA across different criteria')
ax.set_xticks(index)
ax.set_xticklabels(criteria)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()
