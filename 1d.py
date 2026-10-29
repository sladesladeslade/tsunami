# 1D Tsunami Sim
# Slade Brooks
# brooksl@mail.uc.edu

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# set up grid (meters)
Lx = 500e3
nx = 500
dx = Lx/nx
xs = np.linspace(0. - dx, Lx + dx, nx + 2)

# constants
g = 9.81

# sea floor depth: 4000 m deep ocean -> continental slope -> 50 m shelf
Hdeep = 4000.
Hshelf = 50.
Hs = Hshelf + (Hdeep - Hshelf)*0.5*(1 - np.tanh((xs - 0.75*Lx)/(0.05*Lx)))
dHdx = np.gradient(Hs, dx)
c = np.sqrt(g*Hs.max())

# time step (CFL set by deepest water, run long enough to cross the slow shelf)
t = 3*Lx/c
dt = dx/c/2
ts = np.arange(0., t + dt, dt)

# set initial conditions
us = np.zeros((len(xs), len(ts)))
ns = np.zeros_like(us)
ns[:,0] = 1.*np.exp(-((xs - 0.3*Lx)**2)/(2*(30e3)**2))
us[:,0] = ns[:,0]*np.sqrt(g/Hs)     # right-going wave only
# us[:,0] = 0.                      # use this instead to split into two waves

# solve em
for k, time in enumerate(ts[:-1]):
    # do n update
    ns[1:-1, k + 1] = 0.5*(ns[2:, k] + ns[:-2, k]) - dt*(Hs[1:-1]*(us[2:, k] - us[:-2, k])/(2*dx) + dHdx[1:-1]*us[1:-1, k])

    # do u update
    us[1:-1, k + 1] = 0.5*(us[2:, k] + us[:-2, k]) - g*dt*(ns[2:, k] - ns[:-2, k])/(2*dx)

    # update BCs
    ns[0, k + 1] = ns[1, k + 1]
    ns[-1, k + 1] = ns[-2, k + 1]
    us[0, k + 1] = us[1, k + 1]
    us[-1, k + 1] = us[-2, k + 1]

# animate results
fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
line, = ax.plot(xs[1:-1]/1e3, ns[1:-1,0])
title = ax.set_title("t = 0.0 min")
ax.set_ylim(0, ns.max() + 0.1)
ax.set_ylabel("$\\eta$ (m)")
ax.grid()

# sea floor
ax2.fill_between(xs[1:-1]/1e3, -Hs[1:-1], -Hdeep*1.05, color="tan")
ax2.fill_between(xs[1:-1]/1e3, 0, -Hs[1:-1], color="lightskyblue", alpha=0.5)
ax2.plot(xs[1:-1]/1e3, -Hs[1:-1], "k")
ax2.axhline(0, color="b", lw=0.8)
ax2.set_xlim(0, Lx/1e3); ax2.set_ylim(-Hdeep*1.05, 0)
ax2.set_xlabel("x (km)"); ax2.set_ylabel("$-H$ (m)")
ax2.grid()
fig.tight_layout()

def update(k):
    line.set_ydata(ns[1:-1,k])
    title.set_text(f"t = {ts[k]/60:.1f} min")
    return line, title

ani = FuncAnimation(fig, update, frames=range(0, len(ts), 20), interval=50, blit=False)
plt.show()
# ani.save("tsunami.gif", writer="pillow", fps=20)