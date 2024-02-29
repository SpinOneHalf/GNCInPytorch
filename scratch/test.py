import matplotlib.pyplot as plt
import torch


plt.plot(x_new, interpolate_to_support(x, y, x_new), "go", label="Custom interpolation")
plt.plot(x_new, np.interp(x_new, x, y, left=0, right=0), "y-", label="np.interp")
plt.plot(x, y, "b--", label="original values")
plt.legend()
plt.show()