import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import math

Jx = 0.5*np.asarray(
    [
        [0, 1],
        [1, 0]
    ],
    dtype = np.complex128
)

Jy = 0.5*np.asarray(
    [
        [0, -1j],
        [1j,  0]
    ],
    dtype = np.complex128
)

Jz = 0.5*np.asarray(
    [
        [1, 0],
        [0, 1]
    ],
    dtype = np.complex128
)

def get_field_larmor(time_sample, field_modifier, field_sample):
   field_sample[0] = 0            # Zero field in x direction
   field_sample[1] = 0            # Zero field in y direction
   field_sample[2] = 1000         # Split spin z eigenstates by 1kHz

def derivative(time_sample, state):
    field_sample = np.empty(3, np.complex128)
    get_field_larmor(time_sample, 0, field_sample)
    field_matrix = -1j*math.tau*(field_sample[0]*Jx + field_sample[1]*Jy + field_sample[2]*Jz)
    return np.matmul(field_matrix, state)

state_init = np.asarray([1/np.sqrt(2), 1/np.sqrt(2)], np.complex128)
time = np.arange(0e-3, 100e-3, 5e-7, dtype = np.complex128)


results = scipy.integrate.solve_ivp(derivative, [0e-3, 100e-3], state_init, t_eval = time)

state = np.transpose(results.y)
# spin = np.empty(length)

plt.figure()
plt.plot(time, state)
plt.legend(["x", "y", "z"])
plt.xlim(0e-3, 2e-3)
plt.xlabel("time (s)")
plt.ylabel("spin expectation (hbar)")
plt.title("Spin projection for Larmor precession")
plt.show()