import matplotlib.pyplot as plt
import numpy as np

# Criteria for comparison
criteria = ['Convergence Speed', 'Solution Quality', 'Flexibility', 'Scalability', 'Robustness']

# Hypothetical scores for each criterion (out of 10)
aco_scores = [7, 8, 6, 7, 9]  # Ant Colony Optimization
ctoa_scores = [8, 9, 7, 8, 8]  # Class Topper Optimization Algorithm

# Bar width
bar_width = 0.35

# Positions of the bars
index = np.arange(len(criteria))

# Plotting the bars
fig, ax = plt.subplots()
bar1 = ax.bar(index, aco_scores, bar_width, label='ACO', color='b')
bar2 = ax.bar(index + bar_width, ctoa_scores, bar_width, label='CTOA', color='g')

# Adding labels, title, and custom x-axis tick labels
ax.set_xlabel('Criteria')
ax.set_ylabel('Scores')
ax.set_title('Comparison of ACO and CTOA across different criteria')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(criteria)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()