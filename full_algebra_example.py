import spinsim
import numpy as np
import matplotlib.pyplot as plt
import math

def get_field_rabi(time_sample, sweep_parameters, field_sample):
    # Dress atoms from the x direction, Rabi flopping at 1kHz
    field_sample[0] = math.tau*2000*math.cos(sweep_parameters[0]*time_sample)
    field_sample[1] = 0
    field_sample[2] = sweep_parameters[0]   # Split spin z eigenstates by 700kHz
    field_sample[3] = math.tau*1000
    field_sample[4] = math.tau*1000
    field_sample[5] = math.tau*1000
    field_sample[6] = math.tau*1000
    field_sample[7] = math.tau*1000

simulator_rabi = spinsim.Simulator(get_field_rabi, spinsim.SpinQuantumNumber.ONE, exponentiation_method = spinsim.ExponentiationMethod.LIE_TROTTER_8)

result = simulator_rabi.evaluate(0e-3, 100e-3, 100e-9, 500e-9, spinsim.SpinQuantumNumber.ONE.plus_z, [math.tau*20e3])

plt.figure()
plt.plot(result.time, result.spin)
plt.legend(["x", "y", "z"])
plt.xlim(0e-3, 2e-3)
plt.ylim(-1, 1)
plt.xlabel("time (s)")
plt.ylabel("spin expectation (hbar)")
plt.title("Spin projection for Rabi flopping")
plt.show()