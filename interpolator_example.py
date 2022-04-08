import spinsim
import matplotlib.pyplot as plt
import math
import numpy as np

# Generate noise
time_sampled = np.arange(0, 100e-3, 1e-3)
larmor_sampled = np.random.randn(time_sampled.size)

# Generate the interpolator
larmor_interpolator = spinsim.generate_interpolation_sampler(time_sampled, larmor_sampled)
def get_field_larmor_interpolator(time_sample, user_parameters, field_sample):
  # We can vary the noise amplitude per simulation using user_parameters
  noise_amplitude = user_parameters[0]
  field_sample[2] = math.tau*(200 + noise_amplitude*larmor_interpolator(time_sample))

  field_sample[0] = 0
  field_sample[1] = 0
  field_sample[3] = 0

# Simulate the system with first without noise, then with noise
simulator_larmor_interpolator = spinsim.Simulator(get_field_larmor_interpolator, spinsim.SpinQuantumNumber.ONE)
results_larmor = simulator_larmor_interpolator.evaluate(0, 100e-3, 1e-6, 10e-6, spinsim.SpinQuantumNumber.ONE.plus_x, [0])
results_larmor_interpolator = simulator_larmor_interpolator.evaluate(0, 100e-3, 1e-6, 10e-6, spinsim.SpinQuantumNumber.ONE.plus_x, [50])

# Visualise the interpolation
larmor_interpolator_plot = spinsim.generate_interpolation_sampler(time_sampled, larmor_sampled, device = spinsim.Device.CPU)
time_plot = np.arange(0, 100e-3, 100e-6)
larmor_plot = np.zeros(time_plot.size)
for time_index in range(time_plot.size):
  larmor_plot[time_index] = 200 + 50*larmor_interpolator_plot(time_plot[time_index])

# Plot
plt.figure()
plt.plot(time_sampled, 200 + 50*larmor_sampled, "r.", label = "\"Recorded\" amplitudes")
plt.plot(time_plot, larmor_plot, "b-", label = "Interpolated amplitudes")
plt.xlabel("Time (s)")
plt.ylabel("Z field amplitude (Hz)")
plt.legend()
plt.ylim(0, 400)
plt.title("Noisy Larmor interpolation\nField timeseries")
plt.savefig("interpolation_example_1_timeseries.png")
plt.savefig("interpolation_example_1_timeseries.pdf")
plt.draw()

plt.figure()
plt.plot(results_larmor_interpolator.time, results_larmor.spin[:, 0], "g-", label = "No noise")
plt.plot(results_larmor_interpolator.time, results_larmor_interpolator.spin[:, 0], "m-", label = "Noise")
plt.xlabel("Time (ms)")
plt.ylabel("Expected spin x projection (hbar)")
plt.legend()
plt.title("Noisy Larmor interpolation\nField timeseries")
plt.savefig("interpolation_example_1_results.png")
plt.savefig("interpolation_example_1_results.pdf")
plt.draw()

plt.show()