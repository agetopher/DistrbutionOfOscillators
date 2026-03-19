import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import settings
from functions import *

"""
Testing two neurons for starting point of DE
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
settings.d_I = 0    # pA Normal ECF (3, 1)

# Synapses
settings.G_syne = 0 
settings.G_syni = 0.5
settings.E_syne = 0
settings.E_syni = -100.0
settings.E_conn = np.array([[0, 0], [0, 0]])
settings.I_conn = np.array([[0, 1], [1, 0]])

# Synaptic gating parameters (sigmoid threshold)
settings.k_syn = 0.125 # sigmoid steepness
settings.V_th  = -52.0  # mV half-activation voltage

# Synaptic Fatigue parameters
settings.a = 0.000035
settings.b = 0.005

# Number of Cells
settings.numCells = 2 

# Test synaptic gating variables
'''
V = np.linspace(-100, 20, 100)
s = 1.0 / (1.0 + np.exp(-settings.k_syn * (V - settings.V_th)))
plt.plot(V, s)
plt.xlabel("Voltage")
plt.ylabel("Synaptic Gating Variable")
plt.show()
'''

# Voltage to both neurons
Iapp = 3 # 2, 3, 7
 
# Create ODE
def ode(t, y):
    # y[0]: Neuron 0    
    # y[1]: Neuron 1    
    # y[2]: SF 0     
    # y[3]: SF 1   

    fv = f_vec(y[0:2].reshape(2, 1))
    sf = sf_vec(y[0:2].reshape(2,), y[2:].reshape(2,))
 
    # Presynaptic sigmoid gating variable
    s0 = 1.0 / (1.0 + np.exp(-settings.k_syn * (y[0] - settings.V_th)))
    s1 = 1.0 / (1.0 + np.exp(-settings.k_syn * (y[1] - settings.V_th)))

    I_inh0 = settings.G_syni * s1 * (y[0] - settings.E_syni) * y[3]
    I_inh1 = settings.G_syni * s0 * (y[1] - settings.E_syni) * y[2]

    z = np.empty(4,)
    z[0] = (settings.g * (fv[0] - settings.d_I) - I_inh0 + Iapp) / settings.C
    z[1] = (settings.g * (fv[1] - settings.d_I) - I_inh1 + Iapp) / settings.C
    z[2] = sf[0]
    z[3] = sf[1]

    return z

# Set time
t0 = 0
tf = 5000
dt = 1
t = np.linspace(t0, tf, int((tf-t0)/dt))

# Initial Voltages
V_init = np.array([-45.0, settings.L-0.5]).reshape(settings.numCells, 1)

# Initial Fatigue
F_init = np.array([0.2, 0]).reshape(settings.numCells,1)

inits = np.append(V_init, F_init)

sol = solve_ivp(
        ode, [0.0, tf], inits, 
        method='BDF', t_eval=t
    )

F1_init = np.array([0.3, 0]).reshape(settings.numCells,1)

inits1 = np.append(V_init, F1_init)

sol1 = solve_ivp(
        ode, [0.0, tf], inits1,
        method='BDF', t_eval=t
)

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(sol.t, sol.y[0, :], color="blue")
plt.plot(sol.t, sol.y[1, :], color="red")
plt.ylabel("Voltage")
plt.title(f"Mututal Inhibition: Iapp={Iapp}, sf0={F_init[0,0]}, sf1={F_init[1,0]}")
plt.subplot(2, 2, 3)
plt.plot(sol1.t, sol1.y[2, :], color="blue")
plt.plot(sol1.t, sol1.y[3, :], color="red")
plt.ylabel("Synaptic Fatigue")
plt.xlabel("Time")
plt.subplot(2,2,2)
plt.subplot(2, 2, 2)
plt.plot(sol1.t, sol1.y[0, :], color="blue")
plt.plot(sol1.t, sol1.y[1, :], color="red")
plt.ylabel("Voltage")
plt.title(f"Mututal Inhibition: Iapp={Iapp}, sf0={F1_init[0,0]}, sf1={F1_init[1,0]}")
plt.subplot(2, 2, 4)
plt.plot(sol1.t, sol1.y[2, :], color="blue")
plt.plot(sol1.t, sol1.y[3, :], color="red")
plt.ylabel("Synaptic Fatigue")
plt.xlabel("Time")

plt.show()