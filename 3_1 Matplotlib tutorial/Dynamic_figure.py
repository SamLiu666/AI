import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
"""https://matplotlib.org/3.1.1/api/animation_api.html"""
def parameters():
    """fig 是用来 「绘制图表」的 figure 对象；
    chartfunc 是一个以数字为输入的函数，其含义为时间序列上的时间；
    interval 这个更好理解，是帧之间的间隔延迟，以毫秒为单位，默认值为 200"""
    fig = None
    chartfunc = None
    animator = ani.FuncAnimation(fig, chartfunc, interval = 100)

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show()