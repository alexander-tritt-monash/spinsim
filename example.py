import spinsim
import numpy as np
import matplotlib.pyplot as plt

def get_source_larmor(time_sample, source_modifier, source_sample):
   source_sample[0] = 0            # Zero source in x direction
   source_sample[1] = 0            # Zero source in y direction
   source_sample[2] = 1000         # Split spin z eigenstates by 1kHz

simulator_larmor = spinsim.Simulator(get_source_larmor, spinsim.SpinQuantumNumber.HALF)

state_init = np.asarray([1/np.sqrt(2), 1/np.sqrt(2)], np.cdouble)

state, time = simulator_larmor.get_state(0, 0e-3, 100e-3, 100e-9, 500e-9, state_init)
spin = simulator_larmor.get_spin(state)

plt.figure()
plt.plot(time, spin)
plt.legend(["x", "y", "z"])
plt.xlim(0e-3, 2e-3)
plt.xlabel("time (s)")
plt.ylabel("spin expectation (hbar)")
plt.title("Spin projection for Larmor precession")
plt.show()

import spinsim
import numpy as np
import matplotlib.pyplot as plt
import math

def get_source_rabi(time_sample, source_modifier, source_sample):
   # Dress atoms from the x direction, Rabi flopping at 1kHz
   source_sample[0] = 2000*math.cos(math.tau*20e3*source_modifier*time_sample)
   source_sample[1] = 0                        # Zero source in y direction
   source_sample[2] = 20e3*source_modifier     # Split spin z eigenstates by 700kHz
   source_sample[3] = 0                        # Zero quadratic shift, found in spin one systems

simulator_rabi = spinsim.Simulator(get_source_rabi, spinsim.SpinQuantumNumber.ONE)

state_init = np.asarray([1, 0, 0], np.cdouble)

state0, time0 = simulator_rabi.get_state(1, 0e-3, 100e-3, 100e-9, 500e-9, state_init)
spin0 = simulator_rabi.get_spin(state0)

plt.figure()
plt.plot(time0, spin0)
plt.legend(["x", "y", "z"])
plt.xlim(0e-3, 2e-3)
plt.xlabel("time (s)")
plt.ylabel("spin expectation (hbar)")
plt.title("Spin projection for Rabi flopping")
plt.show()

state1, time1 = simulator_rabi.get_state(2, 0e-3, 100e-3, 100e-9, 500e-9, state_init)
spin1 = simulator_rabi.get_spin(state1)

plt.figure()
plt.plot(time1, spin1)
plt.legend(["x", "y", "z"])
plt.xlim(0e-3, 2e-3)
plt.xlabel("time (s)")
plt.ylabel("spin expectation (hbar)")
plt.title("Spin projection for Rabi flopping")
plt.show()