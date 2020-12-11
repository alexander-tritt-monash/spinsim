import spinsim
import math
import numpy as np
import matplotlib.pyplot as plt
import datetime as dtm

time_now_string = dtm.datetime.now().strftime("%Y%m%dT%H%M%S")

def gaussian_pulse(time, modifier, pulse):
    pulse[0] = (math.pi/math.sqrt(math.tau))*math.exp(-0.5*(time**2))/math.tau
    pulse[1] = 0.0
    pulse[2] = 0.0

def cumulative_gaussian(t):
    return 0.5*(1 + math.erf(t/math.sqrt(2.0)))

plt.figure()

time = np.arange(-5.0, 5.1, 2.0)
state_analytic = np.asarray([[math.cos(0.5*math.pi*cumulative_gaussian(t)), -1j*math.sin(0.5*math.pi*cumulative_gaussian(t))] for t in time], dtype = np.complex128)

simulator = spinsim.Simulator(gaussian_pulse, spinsim.SpinQuantumNumber.HALF)

result_compare = simulator.evaluate(0.0, -5.0, 7.0, 1e-6, 2.0, np.asarray([1, 0], np.complex128))

legend = []

colours = ["r", "g", "b", "c", "m", "y"]
error = []
time_steps = np.asarray([0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
number_of_steps = 10 / time_steps
plot_start_index = 4
for simulation_index, time_step in enumerate(time_steps):
    result_simulated = simulator.evaluate(0.0, -5.0, 7.0, time_step, 2.0, np.asarray([1, 0], np.complex128))

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
plt.savefig("gaussian_pulse.png")
plt.savefig("gaussian_pulse.pdf")
plt.show()

plt.figure()
plt.loglog(number_of_steps, error, "-x")
plt.xlabel("Number of steps")
plt.ylabel("Error")
plt.title("{}\nError in integrating Gaussian pulse".format(time_now_string))
plt.ylim((1e-12, 1e0))
plt.savefig("gaussian_pulse_error.png")
plt.savefig("gaussian_pulse_error.pdf")
plt.show()