import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import settings
from functions import *

"""
Testing a single neuron under a ramped applied current
"""

# For Yuval Model
settings.C = 7.0    # pF Membrane Capacitance
settings.g = 1.0    # nS Membrane Conductance
settings.L = -70.0  # mV Resting Potential
settings.T = -45.0  # mV Depolarization Threshold
settings.H = -35.0  # mV Plateau Potential
settings.m1 = 0.7
settings.m2 = 1/81.0
settings.m3 = -1/30.0
settings.m4 = 0.17

MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')

def run(save=True):
    t0 = 0
    tf = 10000
    dt = 1
    t = np.linspace(t0, tf, int(tf-t0/dt))

    def ode(t, y):
        V = f(y[0])
        if t < 2500:
            Iapp = 4*t/tf
        elif t <= 5000:
            Iapp = 2-4*t/tf
        elif t >= 9000:
            Iapp = -1
        else:
            Iapp = 0

        return np.array([(settings.g * V + 4*Iapp) / settings.C])

    init = np.array([settings.L]).reshape(1,)
    sol = solve_ivp(ode, [t0, tf], init, method="BDF", t_eval=t)

    Iapp = np.zeros(t.size)
    Iapp[0:2500] = 4*t[0:2500]/tf
    Iapp[2500:5000] = 2-4*(t[2500:5000])/tf
    Iapp[5000:9000] = 0
    Iapp[9000:] = -1

    fig, axes = plt.subplots(2, 1)
    fig.suptitle("Single Neuron Ramp", fontsize=14)

    axes[0].set_title("Voltage")
    axes[0].plot(sol.t, sol.y[0], color="black")
    axes[0].plot(sol.t, settings.L*np.ones(sol.t.shape), color="blue", linestyle="dotted", label="Leak")
    axes[0].plot(sol.t, settings.T*np.ones(sol.t.shape), color="green", linestyle="dotted", label="Threshold")
    axes[0].plot(sol.t, settings.H*np.ones(sol.t.shape), color="orange", linestyle="dotted", label="Plateau")
    axes[0].legend(loc="upper right", fontsize=6)
    axes[0].set_ylabel("Voltage (mV)")

    axes[1].set_title("Applied Current")
    axes[1].plot(sol.t, Iapp, color="red")
    axes[1].set_ylabel("Iapp (pA)")
    axes[1].set_xlabel("Time (ms)")

    fig.tight_layout()

    if save:
        plt.savefig(os.path.join(MEDIA_DIR, "test_ramp_single_neuron.png"))

    plt.show()


if __name__ == "__main__":
    run()
