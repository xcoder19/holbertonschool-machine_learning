#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

people = ['Farrah', 'Fred', 'Felicia']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruit_labels = ['apples', 'bananas', 'oranges', 'peaches']

bar_width = 0.5

bottoms = np.zeros(len(people))
for i in range(len(fruit)):
    plt.bar(
        people,
        fruit[i],
        bar_width,
        color=colors[i],
        label=fruit_labels[i],
        bottom=bottoms)
    bottoms += fruit[i]


plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.legend()
plt.ylim(0, 80)
plt.yticks(np.arange(0, 81, 10))
plt.show()
