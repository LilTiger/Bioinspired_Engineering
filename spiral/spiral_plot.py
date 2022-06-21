import numpy as np
import matplotlib.pyplot as plt

a = 0.1
b = 0.003183
theta = np.linspace(0, 30*np.pi, 1000)
r = a + b * theta
plt.axes(polar=True)
plt.plot(theta, r)
# # set yaxis visiable
# frame = plt.gca()
# frame.axes.get_yaxis().set_visible(False)
plt.yticks(np.linspace(0.1, 0.4, 4))
# plt.savefig('spiral.tif')
plt.show()

