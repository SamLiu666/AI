import numpy as np
import matplotlib.pyplot as plt

# 基本图表
x  = np.linspace(0, 10, 30)
plt.plot(x, np.sin(x))
plt.show()

plt.plot(x, np.sin(x), '-o')
plt.show()
