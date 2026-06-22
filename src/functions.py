import numpy as np
import settings

# Model Equations

# Synaptic Fatigue
def sf_vec(v, s):
    conditions = [
        (v > settings.L) & (s > 0),
        (v <= settings.L) & (s < 1)
    ]
    choices = [
        -settings.a*(v - settings.L),
        settings.b
    ]
    return np.select(conditions, choices)

# Vector of Neurons Internal dynamics
def f_vec(v):
    conditions = [
        v > settings.H, 
        v >= settings.T, 
        v >= settings.L
    ]
    choices = [
        settings.m4*(-v+settings.H), 
        settings.m3*(v-settings.H)*(v-settings.T), 
        settings.m2*(v-settings.L)*(v-settings.T)
    ]
    return np.select(conditions, choices, default=settings.m1*(-v+settings.L)).reshape(settings.numCells,)

# Slow recovery Variable
def w_inf_vec(v):
    conditions = [
        v > settings.T, 
    ]
    choices = [
        settings.beta*(v - settings.T)
    ]
    return np.select(conditions, choices, default=0).reshape(settings.numCells)

# Single neuron internal dynamics
def f(v):
    if (v > settings.H):
        return settings.m4*(-v+settings.H)
    elif (v >= settings.T):
        return settings.m3*(v-settings.H)*(v-settings.T)
    elif (v >= settings.L):
        return settings.m2*(v-settings.L)*(v-settings.T)
    else:
        return settings.m1*(-v+settings.L)

'''
Loss functions to consider
\distance^\alpha_0 * \kappa(s)^\alpha_1 * \var{\kapaa(s)_s)^{ -\apha_2} * \lambda^\alpha_3
\distance^\alpha_0  +  \kappa(s)^\alpha_1 * \var{\kapaa(s)_s) ^{ -\apha_2} * \lambda ^\alpha_3
'''