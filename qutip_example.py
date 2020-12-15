import qutip
import math
import numpy as np

state_initial = qutip.basis(2, 0)
hamiltonion0 = math.tau*700e3*qutip.sigmaz()
hamiltonion1 = math.tau*2e3*qutip.sigmax()
hamiltonion = [hamiltonion0, [hamiltonion1, "cos(2*pi*700e3*t)"]]
time = np.arange(0, 0.1, 5e-7)

result = qutip.sesolve(hamiltonion, state_initial, time)
print(result)