#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, axs = plt.subplots(3, 2, figsize=(10, 15))
fig.suptitle('All in One', fontsize=16)


axs[0, 0].plot(np.arange(0, 11), y0, color='red')
axs[0, 0].set_title('Plot 1', fontsize='x-small')

axs[0, 1].scatter(x1, y1, color='magenta')
axs[0, 1].set_xlabel('Height (in)', fontsize='x-small')
axs[0, 1].set_ylabel('Weight (lbs)', fontsize='x-small')
axs[0, 1].set_title("Men's Height vs Weight", fontsize='x-small')

axs[1, 0].plot(x2, y2)
axs[1, 0].set_xlabel('Time (years)', fontsize='x-small')
axs[1, 0].set_ylabel('Fraction Remaining', fontsize='x-small')
axs[1, 0].set_title('Exponential Decay of C-14', fontsize='x-small')
axs[1, 0].set_xscale('linear')
axs[1, 0].set_yscale('log')
axs[1, 0].set_xlim(0, 28650)

axs[1, 1].plot(x3, y31, 'r--', label='C-14')
axs[1, 1].plot(x3, y32, 'g-', label='Ra-226')
axs[1, 1].set_xlabel('Time (years)', fontsize='x-small')
axs[1, 1].set_ylabel('Fraction Remaining', fontsize='x-small')
axs[1, 1].set_title(
    'Exponential Decay of Radioactive Elements', fontsize='x-small')
axs[1, 1].set_xlim(0, 20000)
axs[1, 1].set_ylim(0, 1)
axs[1, 1].legend(loc='upper right', fontsize='x-small')


bins = np.arange(0, 101, 10)
axs[2, 0].hist(student_grades, bins, edgecolor='black')
axs[2, 0].set_xlabel('Grades', fontsize='x-small')
axs[2, 0].set_ylabel('Number of Students', fontsize='x-small')
axs[2, 0].set_title('Project A', fontsize='x-small')
axs[2, 0].set_ylim(0, 30)
axs[2, 0].set_xticks(np.arange(0, 101, 10))
axs[2, 0].set_xlim(0, 100)

axs[2, 1].axis('off')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
