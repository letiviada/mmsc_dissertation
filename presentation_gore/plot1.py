import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the x values
x = np.linspace(0, 1, 11)

# Define the y values for three different trends with more randomness
y1 = np.exp(-x * 3) + 1
y2 = np.exp(-x * 2.5) + 1.5
y3 = np.exp(-x * 2) + 2

# Add some noise to the y values with different levels
noise1 = np.random.normal(0, 0.05, size=y1.shape)
noise2 = np.random.normal(0, 0.08, size=y2.shape)
noise3 = np.random.normal(0, 0.1, size=y3.shape)
y_noisy1 = y1 + noise1
y_noisy2 = y2 + noise2
y_noisy3 = y3 + noise3

# Define a function for regression (exponential decay in this case)
def exp_decay(x, a, b):
    return a * np.exp(-b * x)

# Perform curve fitting for each set of noisy data
params1, _ = curve_fit(exp_decay, x, y_noisy1)
params2, _ = curve_fit(exp_decay, x, y_noisy2)
params3, _ = curve_fit(exp_decay, x, y_noisy3)
fitted_y1 = exp_decay(x, *params1)
fitted_y2 = exp_decay(x, *params2)
fitted_y3 = exp_decay(x, *params3)

# Create the plot with colors from the 'tab20' colormap and thinner lines
def data():
    plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-v0_8')
    colors = plt.get_cmap('tab20').colors

    # Add dots on the fitted lines at x = 0.45
    x_dot = 0.45
    y_dot1 = exp_decay(x_dot, *params1)
    y_dot2 = exp_decay(x_dot, *params2)
    y_dot3 = exp_decay(x_dot, *params3)

    #plt.scatter([x_dot], [y_dot1], color='black', marker='.', zorder = 5,s=100)
    #plt.scatter([x_dot], [y_dot2], color='black', marker='.', zorder = 5,s=100)
    #plt.scatter([x_dot], [y_dot3], color='black', marker='.',zorder=5, s=100)

    plt.scatter(x, y_noisy3, color=colors[4], marker='x',label = '$\\beta = 0.04$', s=100)  
    plt.scatter(x, y_noisy2, color=colors[2], marker='x', label = '$\\beta = 0.06$', s=100)  
    plt.scatter(x, y_noisy1, color=colors[0], marker='x', label = '$\\beta = 0.08$', s=100)  


    #plt.plot(x, fitted_y1, color=colors[0], linewidth=2)  
    #plt.plot(x, fitted_y2, color=colors[2], linewidth=2)  
    #plt.plot(x, fitted_y3, color=colors[4], linewidth=2)  


    # Add an arrow
    #plt.annotate('Increasing $\\beta$', xy=(0, 1), xytext=(0.5, 2.75),
               # arrowprops=dict(facecolor='black', width = 1.5,shrink=0.05),
               # fontsize=12, color='black', ha='center')
    plt.xlabel('Stickiness, $\\alpha$')
    plt.xticks(np.linspace(0,1,11))
    plt.ylabel('Throughput')

    plt.title('Value of throughput for fixed initial pore conductance')
    plt.grid(True)
    plt.legend()
    plt.savefig('presentation_gore/figures/data.svg')
    plt.savefig('presentation_gore/figures/data.pdf')
    plt.show()

def before_ML():
    plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-v0_8')
    colors = plt.get_cmap('tab20').colors

    # Add dots on the fitted lines at x = 0.45
    x_dot = 0.45
    y_dot1 = exp_decay(x_dot, *params1)
    y_dot2 = exp_decay(x_dot, *params2)
    y_dot3 = exp_decay(x_dot, *params3)

    #plt.scatter([x_dot], [y_dot1], color='black', marker='.', zorder = 5,s=100)
    #plt.scatter([x_dot], [y_dot2], color='black', marker='.', zorder = 5,s=100)
    #plt.scatter([x_dot], [y_dot3], color='black', marker='.',zorder=5, s=100)

    plt.scatter(x, y_noisy3, color=colors[4], marker='x',label = '$\\beta = 0.04$', s=100)  
    plt.scatter(x, y_noisy2, color=colors[2], marker='x', label = '$\\beta = 0.06$', s=100)  
    plt.scatter(x, y_noisy1, color=colors[0], marker='x', label = '$\\beta = 0.08$', s=100)  


    plt.plot(x, fitted_y1, color=colors[0], linewidth=2)  
    plt.plot(x, fitted_y2, color=colors[2], linewidth=2)  
    plt.plot(x, fitted_y3, color=colors[4], linewidth=2)  


    # Add an arrow
    plt.annotate('Increasing $\\beta$', xy=(0, 1), xytext=(0.5, 2.75),
                arrowprops=dict(facecolor='black', width = 1.5,shrink=0.05),
                fontsize=12, color='black', ha='center')
    plt.xlabel('Stickiness, $\\alpha$')
    plt.xticks(np.linspace(0,1,11))
    plt.ylabel('Throughput')

    plt.title('Value of throughput for fixed initial pore conductance')
    plt.grid(True)
    plt.legend()
    plt.savefig('presentation_gore/figures/before_ML.svg')
    plt.savefig('presentation_gore/figures/before_ML.pdf')
    plt.show()

def after_ML():
    plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-v0_8')
    colors = plt.get_cmap('tab20').colors

    # Add dots on the fitted lines at x = 0.45
    x_dot = 0.45
    y_dot1 = exp_decay(x_dot, *params1)
    y_dot2 = exp_decay(x_dot, *params2)
    y_dot3 = exp_decay(x_dot, *params3)

    plt.scatter([x_dot], [y_dot1], color='black', marker='.', zorder = 5,s=100)
    plt.scatter([x_dot], [y_dot2], color='black', marker='.', zorder = 5,s=100)
    plt.scatter([x_dot], [y_dot3], color='black', marker='.',zorder=5, s=100)

    plt.scatter(x, y_noisy3, color=colors[4], marker='x',label = '$\\beta = 0.04$', s=100)  
    plt.scatter(x, y_noisy2, color=colors[2], marker='x', label = '$\\beta = 0.06$', s=100)  
    plt.scatter(x, y_noisy1, color=colors[0], marker='x', label = '$\\beta = 0.08$', s=100)  

    # Add another cross near the black dots but not quite
    plt.scatter([x_dot], [y_dot1 + 0.05], color='red', marker='x', s=100, zorder=6)
    plt.scatter([x_dot], [y_dot2 - 0.1], color='red', marker='x', s=100, zorder=6)
    plt.scatter([x_dot], [y_dot3 + 0.1], color='red', marker='x', s=100, zorder=6) 

    plt.plot(x, fitted_y1, color=colors[0], linewidth=2)  
    plt.plot(x, fitted_y2, color=colors[2], linewidth=2)  
    plt.plot(x, fitted_y3, color=colors[4], linewidth=2)  


    # Add an arrow
    plt.annotate('Increasing $\\beta$', xy=(0, 1), xytext=(0.5, 2.75),
                arrowprops=dict(facecolor='black', width = 1.5,shrink=0.05),
                fontsize=12, color='black', ha='center')
    plt.xlabel('Stickiness, $\\alpha$')
    plt.xticks(np.linspace(0,1,11))
    plt.ylabel('Throughput')

    plt.title('Value of throughput for fixed initial pore conductance')
    plt.grid(True)
    plt.legend()
    plt.savefig('presentation_gore/figures/after_ML.svg')
    plt.savefig('presentation_gore/figures/after_ML.pdf')
    plt.show()

data()
before_ML()
after_ML()



