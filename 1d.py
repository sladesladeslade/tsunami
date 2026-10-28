# 1D Tsunami Sim
# Slade Brooks
# brooksl@mail.uc.edu

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# set up grid and time
g = 9.81
H = 10.
c = np.sqrt(g*H)
Lx = 100.
nx = 100
dx = Lx/nx
xs = np.linspace(0. - dx, Lx + dx, nx + 2)
t = 2*Lx/c
dt = dx/c/2
ts = np.arange(0., t + dt, dt)

# set initial conditions
Hx = H
us = np.zeros((len(xs), len(ts)))
ns = np.zeros_like(us)
ns[:,0] = 0.1*np.exp(-((xs - xs[len(xs)//2])**2)/100)
# us[:,0] = 0.1*np.exp(-((xs - xs[len(xs)//2])**2)/3)
# us[len(xs)//2:,0] = -0.1

# solve em
for k, time in enumerate(ts[:-1]):
    # do n update
    ns[1:-1, k + 1] = 0.5*(ns[2:, k] + ns[:-2, k]) - Hx*dt*(us[2:, k] - us[:-2, k])/(2*dx)

    # do u update
    us[1:-1, k + 1] = 0.5*(us[2:, k] + us[:-2, k]) - g*dt*(ns[2:, k] - ns[:-2, k])/(2*dx)

    # update BCs
    ns[0, k + 1] = ns[1, k + 1]
    ns[-1, k + 1] = ns[-2, k + 1]
    us[0, k + 1] = us[1, k + 1]
    us[-1, k + 1] = us[-2, k + 1]

# # output results
# plt.figure()
# plt.plot(xs[1:-1], ns[1:-1,0])
# plt.plot(xs[1:-1], ns[1:-1,-1])
# plt.xlim(0, Lx); plt.ylim(0)
# plt.grid(); plt.tight_layout()
# # plt.show()

# animate results
fig, ax = plt.subplots()
line, = ax.plot(xs[1:-1], ns[1:-1,0])
title = ax.set_title("t = 0.000 s")
ax.set_xlim(0, Lx); ax.set_ylim(ns.min() - 0.05, ns.max() + 0.05)
ax.set_xlabel("x (m)"); ax.set_ylabel("$\\eta$ (m)")
ax.grid(); fig.tight_layout()

def update(k):
    line.set_ydata(ns[1:-1,k])
    title.set_text(f"t = {ts[k]:.3f} s")
    return line, title

ani = FuncAnimation(fig, update, frames=len(ts), interval=50, blit=False)
plt.show()
# ani.save("tsunami.gif", writer="pillow", fps=20)