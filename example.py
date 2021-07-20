import spinsim
import matplotlib.pyplot as plt
import math

# Define field for spin system
def get_field_larmor(time_sample, sweep_parameters, field_sample):
   field_sample[0] = 0              # Zero field in x direction
   field_sample[1] = 0              # Zero field in y direction
   field_sample[2] = math.tau*1000  # Split spin z eigenstates by 1kHz

# Initialise simulator instance
simulator_larmor = spinsim.Simulator(get_field_larmor, spinsim.SpinQuantumNumber.HALF, device = spinsim.Device.CPU)
# Evaluate a simulation
results_larmor = simulator_larmor.evaluate(0e-3, 100e-3, 100e-9, 500e-9, spinsim.SpinQuantumNumber.HALF.plus_x)

# Plot results
plt.figure()
plt.plot(results_larmor.time, results_larmor.spin)
plt.legend(["x", "y", "z"])
plt.xlim(0e-3, 2e-3)
plt.xlabel("time (s)")
plt.ylabel("spin expectation (hbar)")
plt.title("Spin projection for Larmor precession")
plt.show()

# import spinsim
# import matplotlib.pyplot as plt
# import math

# # Define field for spin system
# def get_field_rabi(time_sample, sweep_parameters, field_sample):
#    # Dress atoms from the x direction, Rabi flopping
#    field_sample[0] = 2*sweep_parameters[1]*math.cos(sweep_parameters[0]*time_sample)
#    field_sample[1] = 0                    # Zero field in y direction
#    field_sample[2] = sweep_parameters[0]  # Split spin z eigenstates
#    field_sample[3] = 0                    # Zero quadratic shift, found in spin one systems

# # Initialise simulator instance
# simulator_rabi = spinsim.Simulator(get_field_rabi, spinsim.SpinQuantumNumber.ONE)
# # Evaluate a simulation
# result0 = simulator_rabi.evaluate(0e-3, 100e-3, 100e-9, 500e-9, spinsim.SpinQuantumNumber.ONE.plus_z, [math.tau*20e3, math.tau*1e3])
# # Evaluate another simulation
# result1 = simulator_rabi.evaluate(0e-3, 100e-3, 100e-9, 500e-9, spinsim.SpinQuantumNumber.ONE.plus_z, [math.tau*40e3, math.tau*1e3])

# # Plot results
# plt.figure()
# plt.plot(result0.time, result0.spin)
# plt.legend(["x", "y", "z"])
# plt.xlim(0e-3, 2e-3)
# plt.xlabel("time (s)")
# plt.ylabel("spin expectation (hbar)")
# plt.title("Spin projection for Rabi flopping (20kHz bias)")
# plt.draw()
# plt.figure()
# plt.plot(result1.time, result1.spin)
# plt.legend(["x", "y", "z"])
# plt.xlim(0e-3, 2e-3)
# plt.xlabel("time (s)")
# plt.ylabel("spin expectation (hbar)")
# plt.title("Spin projection for Rabi flopping (40kHz bias)")
# plt.draw()

# plt.show()