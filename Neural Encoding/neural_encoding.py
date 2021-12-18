import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

if __name__ == '__main__':
    # Constants
    N = 100000  # Number of samples for Gaussian distributions

    # Gaussian distribution parameters of the response rate under stimulus s1 (-) and s2 (+)
    mean_s1 = 5
    std_s1 = 0.5
    mean_s2 = 7
    std_s2 = 1

    # Distributions
    x_min = min(mean_s1 - 3, mean_s2 - 3)
    x_max = max(mean_s1 + 3, mean_s2 + 3)
    x = np.linspace(x_min, x_max, N)
    s1_pdf = norm.pdf(x, mean_s1, std_s1)
    s2_pdf = norm.pdf(x, mean_s2, std_s2)

    # Likelihood ratio
    loss_plus = 2
    loss_minus = 1
    likelihood_r = s2_pdf / s1_pdf
    lr_threshold = loss_plus / loss_minus

    # Plot distributions
    plt.figure(facecolor='white', dpi=200)
    plt.subplot(1, 2, 1)
    plt.plot(x, s1_pdf, color='blue', label='Stimulus 1 (s1)')
    plt.plot(x, s2_pdf, color='red', label='Stimulus 2 (s2)')
    plt.title('Stimulus conditional distributions')
    plt.xlabel('Stimulus')
    plt.ylabel('pdf')
    plt.axvline(x=mean_s1, linestyle='--', color='blue')
    plt.axvline(x=mean_s2, linestyle='--', color='red')
    plt.legend()

    # Plot likelihood ratio
    plt.subplot(1, 2, 2)
    plt.plot(x, likelihood_r, color='green', label='Likelihood ratio')
    plt.axhline(lr_threshold, color='green', linestyle='--')
    plt.title('Likelihood ratio')
    plt.xlabel('Stimulus')
    plt.ylabel('Likelihood ratio')
    plt.ylim((0, 4))
    plt.legend()

    # Plot
    plt.show()
    plt.close()
    