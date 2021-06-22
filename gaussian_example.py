import spinsim
import math
import numpy as np
import matplotlib.pyplot as plt
import datetime as dtm

time_now_string = dtm.datetime.now().strftime("%Y%m%dT%H%M%S")

def gaussian_pulse(time, sweep_parameters, pulse):
    pulse[0] = (math.pi/math.sqrt(math.tau))*math.exp(-0.5*(time**2))
    pulse[1] = 0.0
    pulse[2] = 0.0

def cumulative_gaussian(t):
    return 0.5*(1 + math.erf(t/math.sqrt(2.0)))

plt.figure()
pulse_sample = np.empty(3, np.float64)

time_continuous = np.arange(-5.0, 5.0005, 1e-3)
pulse_continuous = []
for time_sample in time_continuous:
    gaussian_pulse(time_sample, 0, pulse_sample)
    pulse_continuous += [pulse_sample[0]]
pulse_continuous = np.asarray(pulse_continuous)
plt.plot(time_continuous, pulse_continuous, "k-")

time_step = 0.25
time_midpoint = 0.5*time_step + np.arange(-5.0, 5.0, time_step)
pulse_midpoint = []
for time_sample in time_midpoint:
    gaussian_pulse(time_sample, 0, pulse_sample)
    pulse_midpoint += [pulse_sample[0]]
pulse_midpoint = np.asarray(pulse_midpoint)
plt.plot(time_midpoint, pulse_midpoint, "bo")

time_quadrature = []
pulse_quadrature = []
for time_sample in time_midpoint:
    gaussian_pulse(time_sample - 0.5*time_step/math.sqrt(3), 0, pulse_sample)
    time_quadrature += [time_sample - 0.5*time_step/math.sqrt(3)]
    pulse_quadrature += [pulse_sample[0]]

    gaussian_pulse(time_sample + 0.5*time_step/math.sqrt(3), 0, pulse_sample)
    time_quadrature += [time_sample + 0.5*time_step/math.sqrt(3)]
    pulse_quadrature += [pulse_sample[0]]
time_quadrature = np.asarray(time_quadrature)
pulse_quadrature = np.asarray(pulse_quadrature)
plt.plot(time_quadrature, pulse_quadrature, "m.")

plt.xlabel("Time (standard deviations)")
plt.ylabel("Pulse strength (Hz)")
plt.legend(
    [
        "Pulse shape",
        "Integration steps",
        "Pulse sample points"
    ]
)
plt.title("{}\nSample points for integrating Gaussian pulse".format(time_now_string))
plt.savefig("gaussian_pulse_sample.png")
plt.savefig("gaussian_pulse_sample.pdf")
plt.show()

plt.figure()

time = np.arange(-5.0, 5.1, 2.0)
state_analytic = np.asarray([[math.cos(0.5*math.pi*cumulative_gaussian(t)), -1j*math.sin(0.5*math.pi*cumulative_gaussian(t))] for t in time], dtype = np.complex128)

simulator = spinsim.Simulator(gaussian_pulse, spinsim.SpinQuantumNumber.HALF)

result_compare = simulator.evaluate(-5.0, 7.0, 1e-6, 2.0, spinsim.SpinQuantumNumber.HALF.plus_z)

legend = []

colours = ["r", "g", "b", "c", "m", "y"]
error = []
time_steps = np.asarray([0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
number_of_steps = 10 / time_steps
plot_start_index = 4
for simulation_index, time_step in enumerate(time_steps):
    result_simulated = simulator.evaluate(-5.0, 7.0, time_step, 2.0, np.asarray([1, 0], np.complex128))

    error += [np.sum(np.abs(result_simulated.state - result_compare.state))/5]

    if simulation_index >= plot_start_index:
        plt.plot(time, result_simulated.spin[:, 0], colours[simulation_index - plot_start_index] + "--o")
        plt.plot(time, result_simulated.spin[:, 1], colours[simulation_index - plot_start_index] + "--x")
        plt.plot(time, result_simulated.spin[:, 2], colours[simulation_index - plot_start_index] + "--+")

        legend += [
            "{:d} x".format(int(number_of_steps[simulation_index])),
            "{:d} y".format(int(number_of_steps[simulation_index])),
            "{:d} z".format(int(number_of_steps[simulation_index]))
        ]

plt.legend(legend, loc = "lower left")
plt.xlabel("Time (standard deviations)")
plt.ylabel("Spin")
plt.title("{}\nGaussian pulse at various numbers of steps".format(time_now_string))
plt.show()

plt.figure()
plt.loglog(number_of_steps, error, "-rx")
plt.xlabel("Number of steps")
plt.ylabel("Error")
plt.title("{}\nError in integrating Gaussian pulse".format(time_now_string))
plt.ylim((1e-12, 1e0))
plt.show()