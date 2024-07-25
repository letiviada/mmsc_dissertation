import matplotlib.pyplot as plt

# Sample data
x = [0.05, 0.15, 0.2, 0.05, 0.3]
y = [0.1, 0.2, 0.05, 0.3, 0.4]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', linestyle='-')

# Add placeholders for text
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sample Plot')

# Save as EPS with placeholders
plt.savefig('plot.pgf', format='pgf')