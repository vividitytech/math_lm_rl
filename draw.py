import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ce = np.array([21,21, 39,42, 62,66,57,70, 73,64,67,74,80,83,75,79,79,84,81,86])

ce_rl = np.array([20, 32,32,36,54,40,65, 68,74,92,76, 89, 82,90,96,105,102,93,107,105])

ce_rl2 = np.array([2, 23,28,34,60,57,59, 70,65,75 ,82,82, 90 ,98,98,93, 99,105,111,110])

total = 1319

ce = 100*ce/total
ce_rl = ce_rl/total
ce_rl2 = 100*ce_rl2/total
# data to be plotted
x = np.arange(1, 11)
y = np.array([100, 10, 300, 20, 500, 60, 700, 80, 900, 100])

x = np.arange(1, 21)
# plotting
plt.title("Accuray via the number of epochs")
plt.xlabel("Number of Epochs")
plt.ylabel("% Problems Solved")
plt.plot(x, ce, color ="green", label="CE")
#plt.plot(x, ce_rl, color ="blue", label="CE+RL")
plt.plot(x, ce_rl2, color ="red", label="Our method")

plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

plt.legend()
plt.show()
print("done")