"""
.. _overview_of_simulation_method:

*********************************
Overview of the simulation method
*********************************

The goal here is to evaluate the value of the spin, and thus the quantum state of a 3 level atom in a coarse grained time series with step :math:`\\mathrm{D}t`. The time evolution between time indices is

.. math::
   \\begin{align*}
   \\psi(t + \\mathrm{D}t) &= U(t \\rightarrow t + \\mathrm{D}t) \\psi(t)\\\\
   \\psi(t + \\mathrm{D}t) &= U(t) \\psi(t)
   \\end{align*}

Each :math:`U(t)` is completely independent of :math:`\\psi(t_0)` or :math:`U(t_0)` for any other time value :math:`t_0`. Therefore each :math:`U(t)` can be calculated independently of each other. This is done in parallel using a GPU kernel in the function :func:`getTimeEvolutionCommutatorFree4Rotating_wave()` (the highest performing variant of this solver). Afterwards, the final result of

.. math::
   \\psi(t + \\mathrm{D}t) = U(t) \\psi(t)

is calculated sequentially for each :math:`t` in the function :func:`get_state()`. Afterwards, the spin at each time step is calculated in parallel in the function :func:`get_spin()`.

All magnetic signals fed into the integrator in the form of sine waves, with varying amplitude, frequency, phase, and start and end times. This can be used to simulate anything from the bias and dressing fields, to the fake neural pulses, to AC line and DC detuning noise. These sinusoids are superposed and sampled at any time step needed to for the solver. The magnetic signals are written in high level as :class:`test_signal.TestSignal` objects, and are converted to a parametrisation readable to the integrator in the form of :class:`SourceProperties` objects.

*********
Reference
*********
"""

from . import utilities

from enum import Enum
import numba as nb
from numba import cuda
import math

sqrt2 = math.sqrt(2)
sqrt3 = math.sqrt(3)

class SpinQuantumNumber(Enum):
    """
    The spin quantum number of the system being simulated. 

    Parameters
    ----------
    value : `int`
        Dimension of the hilbert space the states belong to.
    utility_set : `module`
        The module for GPU device functions for that particular spin value.
    label : `string`
        What to write in the HDF5 archive file.
    """

    def __init__(self, value, dimension, utility_set, label):
        super().__init__()
        self._value_ = value
        self.dimension = dimension
        self.utility_set = utility_set
        self.label = label

    HALF = (1/2, 2, utilities.spin_half, "half")
    """
    For two level systems.
    """

    ONE = (1, 3, utilities.spin_one, "one")
    """
    For three level systems.
    """

class IntegrationMethod(Enum):
    """
    Options for describing which method is used during the integration.

    Parameters
    ----------
    value : `string`
        The code name for this method.
    """

    MAGNUS_CF_4 = "magnus_cf4"
    """
    Commutator free fourth order Magnus based.
    """

    MIDPOINT_SAMPLE = "midpoint_sample"
    """
    Naive integration method.
    """

    HALF_STEP = "half_step"
    """
    Integration method from AtomicPy.
    """

class ExponentiationMethod(Enum):
    """
    The implementation to use for matrix exponentiation within the integrator.

    Parameters
    ----------
    value : `string`
        The code name for this method.
    index : `int`
        A reference number, used in the integrator factory where higher level objects cannot be interpreted.
    """
    def __init__(self, value, index):
        super().__init__()
        self._value_ = value
        self.index = index

    ANALYTIC = ("analytic", 0)
    """
    Analytic expression for spin half systems only.
    """

    LIE_TROTTER = ("lie_trotter", 1)
    """
    Approximation using the Lie Trotter theorem.
    """

    # TAYLOR = ("taylor", 2)
    # """
    # Taylor expansion.
    # """

class Simulator:
    def __init__(self, get_source, spin_quantum_number, use_rotating_frame = True, integration_method = IntegrationMethod.MAGNUS_CF_4, exponentiation_method = ExponentiationMethod.LIE_TROTTER, trotter_cutoff = 28, max_registers = 63, threads_per_block = 64):
        """
        Parameters
        ----------
        get_source : `function`
            A python function that describes the source that the spin system is being put under. It must have three arguments:
            
            * **time_sample** (`float`) - the time to sample the source at, int units of s.
            * **simulation_index** (`int`) - a parameter that can be swept over when multiple simulations need to be run. For example, it is used to sweep over dressing frequencies during the simulations that `spinsim` was designed for.
            * **source_sample** (:class:`numpy.ndarray` of `numpy.double` (spatial_index)) the returned value of the source. This is a four dimensional vector, with the first three entries being x, y, z spatial directions (to model a magnetic field, for example), and the optional fourth entry being the amplitude of the quadratic shift (only appearing in spin one systems).

            .. note::
                This function must be :func:`numba.cuda.jit()` compilable, as it will be compiled into a device function.
        trotter_cutoff : `int`
            The number of squares made by the matrix exponentiator, if :obj:`ExponentiationMethod.LIE_TROTTER` is chosen.
        """

        self.threads_per_block = threads_per_block

        self.get_time_evolution_raw = None
        try:
            self.compile_time_evolver(get_source, spin_quantum_number, use_rotating_frame, integration_method, exponentiation_method, trotter_cutoff, max_registers)
        except:
            print("\033[31mspinsim error: numba.cuda could not jit get_source function into a cuda device function.\033[0m\n")
            raise

    def get_time_evolution(self, source_modifier, time_coarse, time_end_points, time_step_fine, time_step_coarse, time_evolution_coarse):
        blocks_per_grid = (time_coarse.size + (self.threads_per_block - 1)) // self.threads_per_block
        try:
            self.get_time_evolution_raw[blocks_per_grid, self.threads_per_block](source_modifier, time_coarse, time_end_points, time_step_fine, time_step_coarse, time_evolution_coarse)
        except:
            print("\033[31mspinsim error: numba.cuda could not jit get_source function into a cuda device function.\033[0m\n")
            raise

    def compile_time_evolver(self, get_source, spin_quantum_number, use_rotating_frame = True, integration_method = IntegrationMethod.MAGNUS_CF_4, exponentiation_method = ExponentiationMethod.LIE_TROTTER, trotter_cutoff = 28, max_registers = 63):
        """
        Parameters
        ----------
        get_source : `function`
            A python function that describes the source that the spin system is being put under. It must have three arguments:
            
            * **time_sample** (`float`) - the time to sample the source at, int units of s.
            * **simulation_index** (`int`) - a parameter that can be swept over when multiple simulations need to be run. For example, it is used to sweep over dressing frequencies during the simulations that `spinsim` was designed for.
            * **source_sample** (:class:`numpy.ndarray` of `numpy.double` (spatial_index)) the returned value of the source. This is a four dimensional vector, with the first three entries being x, y, z spatial directions (to model a magnetic field, for example), and the optional fourth entry being the amplitude of the quadratic shift (only appearing in spin one systems).

            .. note::
                This function must be :func:`numba.cuda.jit()` compilable, as it will be compiled into a device function.
        trotter_cutoff : `int`
            The number of squares made by the matrix exponentiator, if :obj:`ExponentiationMethod.LIE_TROTTER` is chosen.
        """
        dimension = spin_quantum_number.dimension
        lie_dimension = dimension + 1
        utility_set = spin_quantum_number.utility_set

        if integration_method == IntegrationMethod.MAGNUS_CF_4:
            sample_index_max = 3
            sample_index_end = 4
        elif integration_method == IntegrationMethod.HALF_STEP:
            sample_index_max = 3
            sample_index_end = 4
        elif integration_method == IntegrationMethod.MIDPOINT_SAMPLE:
            sample_index_max = 1
            sample_index_end = 1

        exponentiation_method_index = exponentiation_method.index
        if (exponentiation_method == ExponentiationMethod.ANALYTIC) and (spin_quantum_number != SpinQuantumNumber.HALF):
            print("\033[31mspinsim warning!!!\n_attempting to use an analytic exponentiation method outside of spin half. Switching to a Lie Trotter method.\033[0m")
            exponentiation_method = ExponentiationMethod.LIE_TROTTER
            exponentiation_method_index = 1

        @cuda.jit("(float64[:], complex128[:, :], complex128[:, :])", device = True, inline = True)
        def append_exponentiation(source_sample, time_evolution_fine, time_evolution_coarse):
            time_evolution_old = cuda.local.array((dimension, dimension), dtype = nb.complex128)

            # Calculate the exponential
            if exponentiation_method_index == 0:
                utilities.spin_half.matrix_exponential_analytic(source_sample, time_evolution_fine)
            elif exponentiation_method_index == 1:
                utility_set.matrix_exponential_lie_trotter(source_sample, time_evolution_fine, trotter_cutoff)
            # elif exponentiation_method_index == 2:
            #     utility_set.matrix_exponential_taylor(source_sample, time_evolution_fine)

            # Premultiply to the exitsing time evolution operator
            utility_set.set_to(time_evolution_coarse, time_evolution_old)
            utility_set.matrix_multiply(time_evolution_fine, time_evolution_old, time_evolution_coarse)

        if use_rotating_frame:
            if dimension == 3:
                @cuda.jit("(float64[:], float64, complex128)", device = True, inline = True)
                def transform_frame_spin_one_rotating(source_sample, rotating_wave, rotating_wave_winding):
                    X = (source_sample[0] + 1j*source_sample[1])/rotating_wave_winding
                    source_sample[0] = X.real
                    source_sample[1] = X.imag
                    source_sample[2] = source_sample[2] - rotating_wave
        
                transform_frame = transform_frame_spin_one_rotating
            else:
                @cuda.jit("(float64[:], float64, complex128)", device = True, inline = True)
                def transform_frame_spin_half_rotating(source_sample, rotating_wave, rotating_wave_winding):
                    X = (source_sample[0] + 1j*source_sample[1])/(rotating_wave_winding**2)
                    source_sample[0] = X.real
                    source_sample[1] = X.imag
                    source_sample[2] = source_sample[2] - 2*rotating_wave

                transform_frame = transform_frame_spin_half_rotating
        else:
            @cuda.jit("(float64[:], float64, complex128)", device = True, inline = True)
            def transform_frame_lab(source_sample, rotating_wave, rotating_wave_winding):
                return

            transform_frame = transform_frame_lab

        get_source_jit = cuda.jit(get_source, device = True, inline = True)

        if integration_method == IntegrationMethod.MAGNUS_CF_4:
            @cuda.jit("(float64, float64, float64, float64, float64[:, :], float64, complex128[:])", device = True, inline = True)
            def get_source_integration_magnus_cf4(source_modifier, time_fine, time_coarse, time_step_fine, source_sample, rotating_wave, rotating_wave_winding):
                time_sample = ((time_fine + 0.5*time_step_fine*(1 - 1/sqrt3)) - time_coarse)
                rotating_wave_winding[0] = math.cos(math.tau*rotating_wave*time_sample) + 1j*math.sin(math.tau*rotating_wave*time_sample)
                time_sample += time_coarse
                get_source_jit(time_sample, source_modifier, source_sample[0, :])

                time_sample = ((time_fine + 0.5*time_step_fine*(1 + 1/sqrt3)) - time_coarse)
                rotating_wave_winding[1] = math.cos(math.tau*rotating_wave*time_sample) + 1j*math.sin(math.tau*rotating_wave*time_sample)
                time_sample += time_coarse
                get_source_jit(time_sample, source_modifier, source_sample[1, :])

            @cuda.jit("(complex128[:, :], complex128[:, :], float64[:, :], float64, float64, complex128[:])", device = True, inline = True)
            def append_exponentiation_integration_magnus_cf4(time_evolution_fine, time_evolution_coarse, source_sample, time_step_fine, rotating_wave, rotating_wave_winding):
                transform_frame(source_sample[0, :], rotating_wave, rotating_wave_winding[0])
                transform_frame(source_sample[1, :], rotating_wave, rotating_wave_winding[1])

                w0 = (1.5 + sqrt3)/6
                w1 = (1.5 - sqrt3)/6
                
                source_sample[2, 0] = math.tau*time_step_fine*(w0*source_sample[0, 0] + w1*source_sample[1, 0])
                source_sample[2, 1] = math.tau*time_step_fine*(w0*source_sample[0, 1] + w1*source_sample[1, 1])
                source_sample[2, 2] = math.tau*time_step_fine*(w0*source_sample[0, 2] + w1*source_sample[1, 2])
                if dimension > 2:
                    source_sample[2, 3] = math.tau*time_step_fine*(w0*source_sample[0, 3] + w1*source_sample[1, 3])

                append_exponentiation(source_sample[2, :], time_evolution_fine, time_evolution_coarse)

                source_sample[2, 0] = math.tau*time_step_fine*(w1*source_sample[0, 0] + w0*source_sample[1, 0])
                source_sample[2, 1] = math.tau*time_step_fine*(w1*source_sample[0, 1] + w0*source_sample[1, 1])
                source_sample[2, 2] = math.tau*time_step_fine*(w1*source_sample[0, 2] + w0*source_sample[1, 2])
                if dimension > 2:
                    source_sample[2, 3] = math.tau*time_step_fine*(w1*source_sample[0, 3] + w0*source_sample[1, 3])

                append_exponentiation(source_sample[2, :], time_evolution_fine, time_evolution_coarse)

            get_source_integration = get_source_integration_magnus_cf4
            append_exponentiation_integration = append_exponentiation_integration_magnus_cf4

        elif integration_method == IntegrationMethod.HALF_STEP:
            @cuda.jit("(float64, float64, float64, float64, float64[:, :], float64, complex128[:])", device = True, inline = True)
            def get_source_integration_half_step(source_modifier, time_fine, time_coarse, time_step_fine, source_sample, rotating_wave, rotating_wave_winding):
                time_sample = time_fine - time_coarse
                rotating_wave_winding[0] = math.cos(math.tau*rotating_wave*time_sample) + 1j*math.sin(math.tau*rotating_wave*time_sample)
                time_sample += time_coarse
                get_source_jit(time_sample, source_modifier, source_sample[0, :])

                time_sample = time_fine + time_step_fine - time_coarse
                rotating_wave_winding[1] = math.cos(math.tau*rotating_wave*time_sample) + 1j*math.sin(math.tau*rotating_wave*time_sample)
                time_sample += time_coarse
                get_source_jit(time_sample, source_modifier, source_sample[1, :])

            @cuda.jit("(complex128[:, :], complex128[:, :], float64[:, :], float64, float64, complex128[:])", device = True, inline = True)
            def append_exponentiation_integration_half_step(time_evolution_fine, time_evolution_coarse, source_sample, time_step_fine, rotating_wave, rotating_wave_winding):
                transform_frame(source_sample[0, :], rotating_wave, rotating_wave_winding[0])
                transform_frame(source_sample[1, :], rotating_wave, rotating_wave_winding[1])
                
                source_sample[2, 0] = math.tau*time_step_fine*source_sample[0, 0]/2
                source_sample[2, 1] = math.tau*time_step_fine*source_sample[0, 1]/2
                source_sample[2, 2] = math.tau*time_step_fine*source_sample[0, 2]/2
                if dimension > 2:
                    source_sample[2, 3] = math.tau*time_step_fine*source_sample[0, 3]/2

                append_exponentiation(source_sample[2, :], time_evolution_fine, time_evolution_coarse)

                source_sample[2, 0] = math.tau*time_step_fine*source_sample[1, 0]/2
                source_sample[2, 1] = math.tau*time_step_fine*source_sample[1, 1]/2
                source_sample[2, 2] = math.tau*time_step_fine*source_sample[1, 2]/2
                if dimension > 2:
                    source_sample[2, 3] = math.tau*time_step_fine*source_sample[1, 3]/2

                append_exponentiation(source_sample[2, :], time_evolution_fine, time_evolution_coarse)

            get_source_integration = get_source_integration_half_step
            append_exponentiation_integration = append_exponentiation_integration_half_step

        elif integration_method == IntegrationMethod.MIDPOINT_SAMPLE:
            @cuda.jit("(float64, float64, float64, float64, float64[:, :], float64, complex128[:])", device = True, inline = True)
            def get_source_integration_midpoint(source_modifier, time_fine, time_coarse, time_step_fine, source_sample, rotating_wave, rotating_wave_winding):
                time_sample = time_fine + 0.5*time_step_fine - time_coarse
                rotating_wave_winding[0] = math.cos(math.tau*rotating_wave*time_sample) + 1j*math.sin(math.tau*rotating_wave*time_sample)
                time_sample += time_coarse
                get_source_jit(time_sample, source_modifier, source_sample[0, :])

            @cuda.jit("(complex128[:, :], complex128[:, :], float64[:, :], float64, float64, complex128[:])", device = True, inline = True)
            def append_exponentiation_integration_midpoint(time_evolution_fine, time_evolution_coarse, source_sample, time_step_fine, rotating_wave, rotating_wave_winding):
                transform_frame(source_sample[0, :], rotating_wave, rotating_wave_winding[0])
                
                source_sample[0, 0] = math.tau*time_step_fine*source_sample[0, 0]
                source_sample[0, 1] = math.tau*time_step_fine*source_sample[0, 1]
                source_sample[0, 2] = math.tau*time_step_fine*source_sample[0, 2]
                if dimension > 2:
                    source_sample[0, 3] = math.tau*time_step_fine*source_sample[0, 3]

                append_exponentiation(source_sample[0, :], time_evolution_fine, time_evolution_coarse)

            get_source_integration = get_source_integration_midpoint
            append_exponentiation_integration = append_exponentiation_integration_midpoint

        @cuda.jit("(float64, float64[:], float64[:], float64, float64, complex128[:, :, :])", debug = False,  max_registers = max_registers)
        def get_time_evolution(source_modifier, time_coarse, time_end_points, time_step_fine, time_step_coarse, time_evolution_coarse):
            """
            Find the stepwise time evolution opperator.

            Parameters
            ----------
            simulation_index : `int`

            time_coarse : :class:`numpy.ndarray` of :class:`numpy.double` (time_index)
                A coarse grained list of time samples that the time evolution operator is found for. In units of s. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
            time_end_points : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
                The time values for when the experiment is to start and finishes. In units of s.
            time_step_fine : `float`
                The time step used within the integration algorithm. In units of s.
            time_step_coarse : `float`
                The time difference between each element of `time_coarse`. In units of s. Determines the sample rate of the outputs `time_coarse` and `time_evolution_coarse`.
            time_evolution_coarse : :class:`numpy.ndarray` of :class:`numpy.cdouble` (time_index, bra_state_index, ket_state_index)
                Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overview_of_simulation_method`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
            """

            # Declare variables
            time_evolution_fine = cuda.local.array((dimension, dimension), dtype = nb.complex128)

            source_sample = cuda.local.array((sample_index_max, lie_dimension), dtype = nb.float64)
            rotating_wave_winding = cuda.local.array(sample_index_end, dtype = nb.complex128)

            # Run calculation for each coarse timestep in parallel
            time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
            if time_index < time_coarse.size:
                time_coarse[time_index] = time_end_points[0] + time_step_coarse*time_index
                time_fine = time_coarse[time_index]

                # Initialise time evolution operator to 1
                utility_set.set_to_one(time_evolution_coarse[time_index, :])
                source_sample[0, 2] = 0
                if use_rotating_frame:
                    time_sample = time_coarse[time_index] + time_step_coarse/2
                    get_source_jit(time_sample, source_modifier, source_sample[0, :])
                rotating_wave = source_sample[0, 2]
                if dimension == 2:
                    rotating_wave /= 2

                # For every fine step
                for time_fine_index in range(math.floor(time_step_coarse/time_step_fine + 0.5)):
                    get_source_integration(source_modifier, time_fine, time_coarse[time_index], time_step_fine, source_sample, rotating_wave, rotating_wave_winding)
                    append_exponentiation_integration(time_evolution_fine, time_evolution_coarse[time_index, :], source_sample, time_step_fine, rotating_wave, rotating_wave_winding)

                    time_fine += time_step_fine

                if use_rotating_frame:
                    # Take out of rotating frame
                    rotating_wave_winding[0] = math.cos(math.tau*rotating_wave*time_step_coarse) + 1j*math.sin(math.tau*rotating_wave*time_step_coarse)

                    time_evolution_coarse[time_index, 0, 0] /= rotating_wave_winding[0]
                    time_evolution_coarse[time_index, 0, 1] /= rotating_wave_winding[0]
                    if dimension > 2:
                        time_evolution_coarse[time_index, 0, 2] /= rotating_wave_winding[0]

                        time_evolution_coarse[time_index, 2, 0] *= rotating_wave_winding[0]
                        time_evolution_coarse[time_index, 2, 1] *= rotating_wave_winding[0]
                        time_evolution_coarse[time_index, 2, 2] *= rotating_wave_winding[0]
                    else:
                        time_evolution_coarse[time_index, 1, 0] *= rotating_wave_winding[0]
                        time_evolution_coarse[time_index, 1, 1] *= rotating_wave_winding[0]

        self.get_time_evolution_raw = get_time_evolution

    def get_state(self, state_init, state, time_evolution):
        """
        Use the stepwise time evolution operators in succession to find the quantum state timeseries of the 3 level atom.

        Parameters
        ----------
        state_init : :class:`numpy.ndarray` of :class:`numpy.cdouble`
            The state (spin wavefunction) of the system at the start of the simulation.
        state : :class:`numpy.ndarray` of :class:`numpy.cdouble` (time_index, state_index)
            The state (wavefunction) of the spin system in the lab frame, for each time sampled. See :math:`\\psi(t)` in :ref:`overview_of_simulation_method`. This is an output.
        time_evolution : :class:`numpy.ndarray` of :class:`numpy.cdouble` (time_index, bra_state_index, ket_state_index)
            Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overview_of_simulation_method`.
        """

        get_state(state_init, state, time_evolution)

    def get_spin(self, state, spin):
        blocks_per_grid = (state.shape[0] + (self.threads_per_block - 1)) // self.threads_per_block
        get_spin[blocks_per_grid, self.threads_per_block](state, spin)


@nb.jit(nopython = True)
def get_state(state_init, state, time_evolution):
    """
    Use the stepwise time evolution operators in succession to find the quantum state timeseries of the 3 level atom.

    Parameters
    ----------
    state_init : :class:`numpy.ndarray` of :class:`numpy.cdouble`
        The state (spin wavefunction) of the system at the start of the simulation.
    state : :class:`numpy.ndarray` of :class:`numpy.cdouble` (time_index, state_index)
        The state (wavefunction) of the spin system in the lab frame, for each time sampled. See :math:`\\psi(t)` in :ref:`overview_of_simulation_method`. This is an output.
    time_evolution : :class:`numpy.ndarray` of :class:`numpy.cdouble` (time_index, bra_state_index, ket_state_index)
        Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overview_of_simulation_method`.
    """
    for time_index in range(state.shape[0]):
        # State = time evolution * previous state
        for x_index in nb.prange(state.shape[1]):
            state[time_index, x_index] = 0
            if time_index > 0:
                for z_index in range(state.shape[1]):
                    state[time_index, x_index] += time_evolution[time_index - 1, x_index, z_index]*state[time_index - 1, z_index]
            else:
                state[time_index, x_index] += state_init[x_index]

@cuda.jit()
def get_spin(state, spin):
    """
    Calculate each expected spin value in parallel.

    For spin half:

    .. math::
        \\begin{align*}
            \\langle F\\rangle(t) = \\begin{pmatrix}
                \\Re(\\psi_{+\\frac{1}{2}}(t)\\psi_{-\\frac{1}{2}}(t)^*)\\\\
                -\\Im(\\psi_{+\\frac{1}{2}}(t)\\psi_{-\\frac{1}{2}}(t)^*)\\\\
                \\frac{1}{2}(|\\psi_{+\\frac{1}{2}}(t)|^2 - |\\psi_{-\\frac{1}{2}}(t)|^2)
            \\end{pmatrix}
        \\end{align*}

    For spin one:

    .. math::
        \\begin{align*}
            \\langle F\\rangle(t) = \\begin{pmatrix}
                \\Re(\\sqrt{2}\\psi_{0}(t)^*(\\psi_{+1}(t) + \\psi_{-1}(t))\\\\
                -\\Im(\\sqrt{2}\\psi_{0}(t)^*(\\psi_{+1}(t) - \\psi_{-1}(t))\\\\
                |\\psi_{+1}(t)|^2 - |\\psi_{-1}(t)|^2
            \\end{pmatrix}
        \\end{align*}

    Parameters
    ----------
    state : :class:`numpy.ndarray` of :class:`numpy.cdouble` (time_index, state_index)
        The state (wavefunction) of the spin system in the lab frame, for each time sampled. See :math:`\\psi(t)` in :ref:`overview_of_simulation_method`.
    spin : :class:`numpy.ndarray` of :class:`numpy.double` (time_index, spatial_index)
        The expected value for hyperfine spin of the spin system in the lab frame, for each time sampled. Units of :math:`\\hbar`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    """
    time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if time_index < spin.shape[0]:
        if state.shape[1] == 2:
            spin[time_index, 0] = (state[time_index, 0]*utilities.scalar.conj(state[time_index, 1])).real
            spin[time_index, 1] = (1j*state[time_index, 0]*utilities.scalar.conj(state[time_index, 1])).real
            spin[time_index, 2] = 0.5*(state[time_index, 0].real**2 + state[time_index, 0].imag**2 - state[time_index, 1].real**2 - state[time_index, 1].imag**2)
        else:
            spin[time_index, 0] = (2*utilities.scalar.conj(state[time_index, 1])*(state[time_index, 0] + state[time_index, 2])/sqrt2).real
            spin[time_index, 1] = (2j*utilities.scalar.conj(state[time_index, 1])*(state[time_index, 0] - state[time_index, 2])/sqrt2).real
            spin[time_index, 2] = state[time_index, 0].real**2 + state[time_index, 0].imag**2 - state[time_index, 2].real**2 - state[time_index, 2].imag**2