"""

"""

# from . import utilities

from enum import Enum
import numpy as np
import numba as nb
from numba import cuda
from numba import roc
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
    label : `string`
        What to write in the HDF5 archive file.
    """

    def __init__(self, value, dimension, label):
        super().__init__()
        self._value_ = value
        self.dimension = dimension
        self.label = label

    HALF = (1/2, 2, "half")
    """
    For two level systems.
    """

    ONE = (1, 3, "one")
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

class Device(Enum):
    def __init__(self, value, index):
        super().__init__()
        self._value_ = value
        self.index = index

        if value == "python":
            def jit_host(template, max_registers):
                def jit_host(func):
                    return func
                return jit_host
            self.jit_host = jit_host

            def jit_device(func):
                return func
            self.jit_device = jit_device

            def jit_device_template(template):
                def jit_device_template(func):
                    return func
                return jit_device_template
            self.jit_device_template = jit_device_template

        elif value == "cpu_single":
            def jit_host(template, max_registers):
                def jit_host(func):
                    return nb.njit(template)(func)
                return jit_host
            self.jit_host = jit_host

            def jit_device(func):
                return nb.njit()(func)
            self.jit_device = jit_device

            def jit_device_template(template):
                def jit_device_template(func):
                    return nb.njit(template)(func)
                return jit_device_template
            self.jit_device_template = jit_device_template

        elif value == "cpu":
            def jit_host(template, max_registers):
                def jit_host(func):
                    return nb.njit(template, parallel = True)(func)
                return jit_host
            self.jit_host = jit_host

            def jit_device(func):
                return nb.njit()(func)
            self.jit_device = jit_device

            def jit_device_template(template):
                def jit_device_template(func):
                    return nb.njit(template)(func)
                return jit_device_template
            self.jit_device_template = jit_device_template

        elif value == "cuda":
            def jit_host(template, max_registers):
                def jit_host(func):
                    return cuda.jit(template, debug = False,  max_registers = max_registers)(func)
                return jit_host
            self.jit_host = jit_host

            def jit_device(func):
                return cuda.jit(device = True, inline = True)(func)
            self.jit_device = jit_device

            def jit_device_template(template):
                def jit_device_template(func):
                    return cuda.jit(template, device = True, inline = True)(func)
                return jit_device_template
            self.jit_device_template = jit_device_template

        elif value == "roc":
            def jit_host(template, max_registers):
                def jit_host(func):
                    return roc.jit(template)(func)
                return jit_host
            self.jit_host = jit_host

            def jit_device(func):
                return roc.jit(device = True)(func)
            self.jit_device = jit_device

            def jit_device_template(template):
                def jit_device_template(func):
                    return roc.jit(template, device = True)(func)
                return jit_device_template
            self.jit_device_template = jit_device_template

    PYTHON = ("python", 0)
    CPU_SINGLE = ("cpu_single", 0)
    CPU = ("cpu", 0)
    CUDA = ("cuda", 1)
    ROC = ("roc", 2)

class Simulator:
    def __init__(self, get_source, spin_quantum_number, device = Device.CUDA, exponentiation_method = None, use_rotating_frame = True, integration_method = IntegrationMethod.MAGNUS_CF_4, trotter_cutoff = 28, max_registers = 63, threads_per_block = 64):
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
        self.spin_quantum_number = spin_quantum_number
        self.device = device

        self.get_time_evolution_raw = None
        try:
            self.compile_time_evolver(get_source, spin_quantum_number, device, use_rotating_frame, integration_method, exponentiation_method, trotter_cutoff, max_registers, threads_per_block)
        except:
            print("\033[31mspinsim error: numba could not jit get_source function into a device function.\033[0m\n")
            raise

    # def get_time_evolution(self, source_modifier, time_coarse, time_end_points, time_step_fine, time_step_coarse, time_evolution_coarse):
    #     blocks_per_grid = (time_coarse.size + (self.threads_per_block - 1)) // self.threads_per_block
    #     try:
    #         self.get_time_evolution_raw[blocks_per_grid, self.threads_per_block](source_modifier, time_coarse, time_end_points, time_step_fine, time_step_coarse, time_evolution_coarse)
    #     except:
    #         print("\033[31mspinsim error: numba.cuda could not jit get_source function into a cuda device function.\033[0m\n")
    #         raise

    def compile_time_evolver(self, get_source, spin_quantum_number, device, use_rotating_frame = True, integration_method = IntegrationMethod.MAGNUS_CF_4, exponentiation_method = None, trotter_cutoff = 28, max_registers = 63, threads_per_block = 64):
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
        utilities = Utilities(spin_quantum_number, device, threads_per_block)
        conj = utilities.conj
        complex_abs = utilities.complex_abs
        norm2 = utilities.norm2
        inner = utilities.inner
        set_to = utilities.set_to
        set_to_one = utilities.set_to_one
        set_to_zero = utilities.set_to_zero
        matrix_multiply = utilities.matrix_multiply
        adjoint = utilities.adjoint
        matrix_exponential_analytic = utilities.matrix_exponential_analytic
        matrix_exponential_lie_trotter = utilities.matrix_exponential_lie_trotter
        matrix_exponential_taylor = utilities.matrix_exponential_taylor

        jit_host = device.jit_host
        jit_device = device.jit_device
        jit_device_template = device.jit_device_template
        device_index = device.index

        dimension = spin_quantum_number.dimension
        lie_dimension = dimension + 1
        # utility_set = spin_quantum_number.utility_set

        if not exponentiation_method:
            if spin_quantum_number == SpinQuantumNumber.ONE:
                exponentiation_method = ExponentiationMethod.LIE_TROTTER
            elif spin_quantum_number == SpinQuantumNumber.HALF:
                exponentiation_method = ExponentiationMethod.ANALYTIC

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
        @jit_device_template("(float64[:], complex128[:, :], complex128[:, :])")
        def append_exponentiation(source_sample, time_evolution_fine, time_evolution_coarse):
            if device_index == 0:
                time_evolution_old = np.empty((dimension, dimension), dtype = np.complex128)
            elif device_index == 1:
                time_evolution_old = cuda.local.array((dimension, dimension), dtype = np.complex128)
            elif device_index == 2:
                time_evolution_old_group = roc.shared.array((threads_per_block, dimension, dimension), dtype = np.complex128)
                time_evolution_old = time_evolution_old_group[roc.get_local_id(1), :, :]

            # Calculate the exponential
            if exponentiation_method_index == 0:
                matrix_exponential_analytic(source_sample, time_evolution_fine)
            elif exponentiation_method_index == 1:
                matrix_exponential_lie_trotter(source_sample, time_evolution_fine, trotter_cutoff)
            # elif exponentiation_method_index == 2:
            #     utility_set.matrix_exponential_taylor(source_sample, time_evolution_fine)

            # Premultiply to the exitsing time evolution operator
            set_to(time_evolution_coarse, time_evolution_old)
            matrix_multiply(time_evolution_fine, time_evolution_old, time_evolution_coarse)

        if use_rotating_frame:
            if dimension == 3:
                @jit_device_template("(float64[:], float64, complex128)")
                def transform_frame_spin_one_rotating(source_sample, rotating_wave, rotating_wave_winding):
                    X = (source_sample[0] + 1j*source_sample[1])/rotating_wave_winding
                    source_sample[0] = X.real
                    source_sample[1] = X.imag
                    source_sample[2] = source_sample[2] - rotating_wave
                transform_frame = transform_frame_spin_one_rotating
            else:
                @jit_device_template("(float64[:], float64, complex128)")
                def transform_frame_spin_half_rotating(source_sample, rotating_wave, rotating_wave_winding):
                    X = (source_sample[0] + 1j*source_sample[1])/(rotating_wave_winding**2)
                    source_sample[0] = X.real
                    source_sample[1] = X.imag
                    source_sample[2] = source_sample[2] - 2*rotating_wave
                transform_frame = transform_frame_spin_half_rotating
        else:
            @jit_device_template("(float64[:], float64, complex128)")
            def transform_frame_lab(source_sample, rotating_wave, rotating_wave_winding):
                return
            transform_frame = transform_frame_lab

        get_source_jit = jit_device(get_source)

        if integration_method == IntegrationMethod.MAGNUS_CF_4:
            @jit_device_template("(float64, float64, float64, float64, float64[:, :], float64, complex128[:])")
            def get_source_integration_magnus_cf4(source_modifier, time_fine, time_coarse, time_step_fine, source_sample, rotating_wave, rotating_wave_winding):
                time_sample = ((time_fine + 0.5*time_step_fine*(1 - 1/sqrt3)) - time_coarse)
                rotating_wave_winding[0] = math.cos(math.tau*rotating_wave*time_sample) + 1j*math.sin(math.tau*rotating_wave*time_sample)
                time_sample += time_coarse
                get_source_jit(time_sample, source_modifier, source_sample[0, :])

                time_sample = ((time_fine + 0.5*time_step_fine*(1 + 1/sqrt3)) - time_coarse)
                rotating_wave_winding[1] = math.cos(math.tau*rotating_wave*time_sample) + 1j*math.sin(math.tau*rotating_wave*time_sample)
                time_sample += time_coarse
                get_source_jit(time_sample, source_modifier, source_sample[1, :])

            @jit_device_template("(complex128[:, :], complex128[:, :], float64[:, :], float64, float64, complex128[:])")
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
            @jit_device_template("(float64, float64, float64, float64, float64[:, :], float64, complex128[:])")
            def get_source_integration_half_step(source_modifier, time_fine, time_coarse, time_step_fine, source_sample, rotating_wave, rotating_wave_winding):
                time_sample = time_fine - time_coarse
                rotating_wave_winding[0] = math.cos(math.tau*rotating_wave*time_sample) + 1j*math.sin(math.tau*rotating_wave*time_sample)
                time_sample += time_coarse
                get_source_jit(time_sample, source_modifier, source_sample[0, :])

                time_sample = time_fine + time_step_fine - time_coarse
                rotating_wave_winding[1] = math.cos(math.tau*rotating_wave*time_sample) + 1j*math.sin(math.tau*rotating_wave*time_sample)
                time_sample += time_coarse
                get_source_jit(time_sample, source_modifier, source_sample[1, :])

            @jit_device_template("(complex128[:, :], complex128[:, :], float64[:, :], float64, float64, complex128[:])")
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
            @jit_device_template("(float64, float64, float64, float64, float64[:, :], float64, complex128[:])")
            def get_source_integration_midpoint(source_modifier, time_fine, time_coarse, time_step_fine, source_sample, rotating_wave, rotating_wave_winding):
                time_sample = time_fine + 0.5*time_step_fine - time_coarse
                rotating_wave_winding[0] = math.cos(math.tau*rotating_wave*time_sample) + 1j*math.sin(math.tau*rotating_wave*time_sample)
                time_sample += time_coarse
                get_source_jit(time_sample, source_modifier, source_sample[0, :])

            @jit_device_template("(complex128[:, :], complex128[:, :], float64[:, :], float64, float64, complex128[:])")
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

        @jit_device_template("(int64, float64[:], float64, float64, float64[:], complex128[:, :, :], float64)")
        def get_time_evolution_loop(time_index, time_coarse, time_step_coarse, time_step_fine, time_end_points, time_evolution_coarse, source_modifier):
            # Declare variables
            if device_index == 0:
                time_evolution_fine = np.empty((dimension, dimension), dtype = np.complex128)

                source_sample = np.empty((sample_index_max, lie_dimension), dtype = np.float64)
                rotating_wave_winding = np.empty(sample_index_end, dtype = np.complex128)
            elif device_index == 1:
                time_evolution_fine = cuda.local.array((dimension, dimension), dtype = np.complex128)

                source_sample = cuda.local.array((sample_index_max, lie_dimension), dtype = np.float64)
                rotating_wave_winding = cuda.local.array(sample_index_end, dtype = np.complex128)
            elif device_index == 2:
                time_evolution_fine_group = roc.shared.array((threads_per_block, dimension, dimension), dtype = np.complex128)
                time_evolution_fine = time_evolution_fine_group[roc.get_local_id(1), :, :]

                source_sample_group = roc.shared.array((threads_per_block, sample_index_max, lie_dimension), dtype = np.float64)
                source_sample = source_sample_group[roc.get_local_id(1), :, :]
                rotating_wave_winding_group = roc.shared.array((threads_per_block, sample_index_end), dtype = np.complex128)
                rotating_wave_winding = rotating_wave_winding_group[roc.get_local_id(1), :]
            
            time_coarse[time_index] = time_end_points[0] + time_step_coarse*time_index
            time_fine = time_coarse[time_index]

            # Initialise time evolution operator to 1
            set_to_one(time_evolution_coarse[time_index, :])
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

        @jit_host("(float64, float64[:], float64[:], float64, float64, complex128[:, :, :])", max_registers)
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

            if device_index == 0:
                for time_index in nb.prange(time_coarse.size):
                    get_time_evolution_loop(time_index, time_coarse, time_step_coarse, time_step_fine, time_end_points, time_evolution_coarse, source_modifier)
            elif device_index == 1:
                # Run calculation for each coarse timestep in parallel
                time_index = cuda.grid(1)
                if time_index < time_coarse.size:
                    get_time_evolution_loop(time_index, time_coarse, time_step_coarse, time_step_fine, time_end_points, time_evolution_coarse, source_modifier)
            elif device_index == 2:
                # Run calculation for each coarse timestep in parallel
                time_index = roc.get_global_id(1)
                if time_index < time_coarse.size:
                    get_time_evolution_loop(time_index, time_coarse, time_step_coarse, time_step_fine, time_end_points, time_evolution_coarse, source_modifier)
            return
        # get_time_evolution = cuda.jit("(float64, float64[:], float64[:], float64, float64, complex128[:, :, :])", debug = False,  max_registers = max_registers)(get_time_evolution)
        # get_time_evolution = jit_host("(float64, float64[:], float64[:], float64, float64, complex128[:, :, :])", max_registers)(get_time_evolution)
        @jit_host("(complex128[:, :], float64[:, :])", max_registers = max_registers)
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
            if device_index == 0:
                for time_index in nb.prange(spin.shape[0]):
                    if dimension == 2:
                        spin[time_index, 0] = (state[time_index, 0]*conj(state[time_index, 1])).real
                        spin[time_index, 1] = (1j*state[time_index, 0]*conj(state[time_index, 1])).real
                        spin[time_index, 2] = 0.5*(state[time_index, 0].real**2 + state[time_index, 0].imag**2 - state[time_index, 1].real**2 - state[time_index, 1].imag**2)
                    else:
                        spin[time_index, 0] = (2*conj(state[time_index, 1])*(state[time_index, 0] + state[time_index, 2])/sqrt2).real
                        spin[time_index, 1] = (2j*conj(state[time_index, 1])*(state[time_index, 0] - state[time_index, 2])/sqrt2).real
                        spin[time_index, 2] = state[time_index, 0].real**2 + state[time_index, 0].imag**2 - state[time_index, 2].real**2 - state[time_index, 2].imag**2
            elif device_index > 0:
                if device_index == 1:
                    time_index = cuda.grid(1)
                elif device_index == 1:
                    time_index = roc.get_global_id(1)
                if time_index < spin.shape[0]:
                    if dimension == 2:
                        spin[time_index, 0] = (state[time_index, 0]*conj(state[time_index, 1])).real
                        spin[time_index, 1] = (1j*state[time_index, 0]*conj(state[time_index, 1])).real
                        spin[time_index, 2] = 0.5*(state[time_index, 0].real**2 + state[time_index, 0].imag**2 - state[time_index, 1].real**2 - state[time_index, 1].imag**2)
                    else:
                        spin[time_index, 0] = (2*conj(state[time_index, 1])*(state[time_index, 0] + state[time_index, 2])/sqrt2).real
                        spin[time_index, 1] = (2j*conj(state[time_index, 1])*(state[time_index, 0] - state[time_index, 2])/sqrt2).real
                        spin[time_index, 2] = state[time_index, 0].real**2 + state[time_index, 0].imag**2 - state[time_index, 2].real**2 - state[time_index, 2].imag**2
            return

        self.get_time_evolution_raw = get_time_evolution
        self.get_spin_raw = get_spin

    def get_state(self, source_modifier, time_start, time_end, time_step_fine, time_step_coarse, state_init):
        time_end_points = np.asarray([time_start, time_end], np.float64)

        time_index_max = int((time_end_points[1] - time_end_points[0])/time_step_coarse)
        if self.device.index == 0:
            time = np.empty(time_index_max, np.float64)
            self.time_evolution_coarse = np.empty((time_index_max, self.spin_quantum_number.dimension, self.spin_quantum_number.dimension), np.complex128)

            self.get_time_evolution_raw(source_modifier, time, time_end_points, time_step_fine, time_step_coarse, self.time_evolution_coarse)

            state = np.empty((time_index_max, self.spin_quantum_number.dimension), np.complex128)
            get_state(state_init, state, self.time_evolution_coarse)

        elif self.device == Device.CUDA:
            time = cuda.device_array(time_index_max, np.float64)
            self.time_evolution_coarse = cuda.device_array((time_index_max, self.spin_quantum_number.dimension, self.spin_quantum_number.dimension), np.complex128)

            blocks_per_grid = (time.size + (self.threads_per_block - 1)) // self.threads_per_block
            try:
                self.get_time_evolution_raw[blocks_per_grid, self.threads_per_block](source_modifier, time, time_end_points, time_step_fine, time_step_coarse, self.time_evolution_coarse)
            except:
                print("\033[31mspinsim error: numba.cuda could not jit get_source function into a cuda device function.\033[0m\n")
                raise

            self.time_evolution_coarse = self.time_evolution_coarse.copy_to_host()
            time = time.copy_to_host()
            state = np.empty((time_index_max, self.spin_quantum_number.dimension), np.complex128)
            get_state(state_init, state, self.time_evolution_coarse)
        
        elif self.device == Device.ROC:
            time = roc.device_array(time_index_max, np.float64)
            self.time_evolution_coarse = roc.device_array((time_index_max, self.spin_quantum_number.dimension, self.spin_quantum_number.dimension), np.complex128)

            blocks_per_grid = (time.size + (self.threads_per_block - 1)) // self.threads_per_block
            try:
                self.get_time_evolution_raw[blocks_per_grid, self.threads_per_block](source_modifier, time, time_end_points, time_step_fine, time_step_coarse, self.time_evolution_coarse)
            except:
                print("\033[31mspinsim error: numba.roc could not jit get_source function into a roc device function.\033[0m\n")
                raise

            self.time_evolution_coarse = self.time_evolution_coarse.copy_to_host()
            time = time.copy_to_host()
            state = np.empty((time_index_max, self.spin_quantum_number.dimension), np.complex128)
            get_state(state_init, state, self.time_evolution_coarse)

        return state, time

    # def get_state(self, state_init, state, time_evolution):
    #     """
    #     Use the stepwise time evolution operators in succession to find the quantum state timeseries of the 3 level atom.

    #     Parameters
    #     ----------
    #     state_init : :class:`numpy.ndarray` of :class:`numpy.cdouble`
    #         The state (spin wavefunction) of the system at the start of the simulation.
    #     state : :class:`numpy.ndarray` of :class:`numpy.cdouble` (time_index, state_index)
    #         The state (wavefunction) of the spin system in the lab frame, for each time sampled. See :math:`\\psi(t)` in :ref:`overview_of_simulation_method`. This is an output.
    #     time_evolution : :class:`numpy.ndarray` of :class:`numpy.cdouble` (time_index, bra_state_index, ket_state_index)
    #         Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overview_of_simulation_method`.
    #     """

    #     get_state(state_init, state, time_evolution)

    def get_spin(self, state):
        if self.device.index == 0:
            spin = np.empty((state.shape[0], 3), np.float64)
            self.get_spin_raw(state, spin)
        elif self.device == Device.CUDA:
            spin = cuda.device_array((state.shape[0], 3), np.float64)
            blocks_per_grid = (state.shape[0] + (self.threads_per_block - 1)) // self.threads_per_block
            self.get_spin_raw[blocks_per_grid, self.threads_per_block](cuda.to_device(state), spin)
            spin = spin.copy_to_host()
        elif self.device == Device.ROC:
            spin = roc.device_array((state.shape[0], 3), np.float64)
            blocks_per_grid = (state.shape[0] + (self.threads_per_block - 1)) // self.threads_per_block
            self.get_spin_raw[blocks_per_grid, self.threads_per_block](roc.to_device(state), spin)
            spin = spin.copy_to_host()
        return spin


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

sqrt2 = math.sqrt(2)
sqrt3 = math.sqrt(3)
machine_epsilon = np.finfo(np.float64).eps*1000

class Utilities:
    def __init__(self, spin_quantum_number, device, threads_per_block):
        jit_device = device.jit_device
        device_index = device.index

        @jit_device
        def conj(z):
            """
            Conjugate of a complex number.

            .. math::
                \\begin{align*}
                (a + ib)^* &= a - ib\\\\
                a, b &\\in \\mathbb{R}
                \\end{align*}

            Parameters
            ----------
            z : :class:`numpy.cdouble`
                The complex number to take the conjugate of.
            
            Returns
            -------
            cz : :class:`numpy.cdouble`
                The conjugate of z.
            """
            return (z.real - 1j*z.imag)

        @jit_device
        def complex_abs(z):
            """
            The absolute value of a complex number.

            .. math::
                \\begin{align*}
                |a + ib| &= \\sqrt{a^2 + b^2}\\\\
                a, b &\\in \\mathbb{R}
                \\end{align*}
            
            Parameters
            ----------
            z : :class:`numpy.cdouble`
                The complex number to take the absolute value of.
            
            Returns
            -------
            az : :class:`numpy.double`
                The absolute value of z.
            """
            return math.sqrt(z.real**2 + z.imag**2)

        if spin_quantum_number == SpinQuantumNumber.HALF:
            @jit_device
            def norm2(z):
                """
                The 2 norm of a complex vector.

                .. math::
                    \|a + ib\|_2 = \\sqrt {\\left(\\sum_i a_i^2 + b_i^2\\right)}

                Parameters
                ----------
                z : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (index)
                    The vector to take the 2 norm of.
                Returns
                -------
                nz : :class:`numpy.double`
                    The 2 norm of z.
                """
                return math.sqrt(z[0].real**2 + z[0].imag**2 + z[1].real**2 + z[1].imag**2)

            @jit_device
            def inner(left, right):
                """
                The inner (maths convention dot) product between two complex vectors. 
                
                .. note::
                    The mathematics definition is used here rather than the physics definition, so the left vector is conjugated. Thus the inner product of two orthogonal vectors is 0.

                .. math::
                    \\begin{align*}
                    l \\cdot r &\\equiv \\langle l, r \\rangle\\\\
                    l \\cdot r &= \\sum_i (l_i)^* r_i
                    \\end{align*}

                Parameters
                ----------
                left : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (index)
                    The vector to left multiply in the inner product.
                right : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (index)
                    The vector to right multiply in the inner product.
                
                Returns
                -------
                d : :class:`numpy.cdouble`
                    The inner product of l and r.
                """
                return conj(left[0])*right[0] + conj(left[1])*right[1]

            @jit_device
            def set_to(operator, result):
                """
                Copy the contents of one matrix into another.

                .. math::
                    (A)_{i, j} = (B)_{i, j}

                Parameters
                ----------
                operator : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to copy from.
                result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to copy to.
                """
                for x_index in range(2):
                    for y_index in range(2):
                        result[y_index, x_index] = operator[y_index, x_index]

            @jit_device
            def set_to_one(operator):
                """
                Make a matrix the multiplicative identity, ie, :math:`1`.

                .. math::
                    \\begin{align*}
                    (A)_{i, j} &= \\delta_{i, j}\\\\
                    &= \\begin{cases}
                        1,&i = j\\\\
                        0,&i\\neq j
                    \\end{cases}
                    \\end{align*}

                Parameters
                ----------
                operator : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to set to :math:`1`.
                """
                operator[0, 0] = 1
                operator[1, 0] = 0

                operator[0, 1] = 0
                operator[1, 1] = 1

            @jit_device
            def set_to_zero(operator):
                """
                Make a matrix the additive identity, ie, :math:`0`.

                .. math::
                    \\begin{align*}
                    (A)_{i, j} = 0
                    \\end{align*}

                Parameters
                ----------
                operator : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to set to :math:`0`.
                """
                operator[0, 0] = 0
                operator[1, 0] = 0

                operator[0, 1] = 0
                operator[1, 1] = 0

            @jit_device
            def matrix_multiply(left, right, result):
                """
                Multiply matrices left and right together, to be returned in result.

                .. math::
                    \\begin{align*}
                    (LR)_{i,k} = \\sum_j (L)_{i,j} (R)_{j,k}
                    \\end{align*}

                Parameters
                ----------
                left : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to left multiply by.
                right : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to right multiply by.
                result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    A matrix to be filled with the result of the product.
                """
                result[0, 0] = left[0, 0]*right[0, 0] + left[0, 1]*right[1, 0]
                result[1, 0] = left[1, 0]*right[0, 0] + left[1, 1]*right[1, 0]

                result[0, 1] = left[0, 0]*right[0, 1] + left[0, 1]*right[1, 1]
                result[1, 1] = left[1, 0]*right[0, 1] + left[1, 1]*right[1, 1]

            @jit_device
            def adjoint(operator, result):
                """
                Takes the hermitian adjoint of a matrix.

                .. math::
                    \\begin{align*}
                    A^\\dagger &\\equiv A^H\\\\
                    (A^\\dagger)_{y,x} &= ((A)_{x,y})^*
                    \\end{align*}
                
                Matrix can be in :math:`\\mathbb{C}^{2\\times2}` or :math:`\\mathbb{C}^{3\\times3}`.

                Parameters
                ----------
                operator : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The operator to take the adjoint of.
                result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    An array to write the resultant adjoint to.
                """
                result[0, 0] = conj(operator[0, 0])
                result[1, 0] = conj(operator[0, 1])

                result[0, 1] = conj(operator[1, 0])
                result[1, 1] = conj(operator[1, 1])

            @jit_device
            def matrix_exponential_analytic(source_sample, result):
                """
                Calculates a :math:`su(2)` matrix exponential based on its analytic form.

                Assumes the exponent is an imaginary  linear combination of :math:`su(2)`, being,

                .. math::
                    \\begin{align*}
                        A &= -i(x F_x + y F_y + z F_z),
                    \\end{align*}
                
                with

                .. math::
                    \\begin{align*}
                        F_x &= \\frac{1}{2}\\begin{pmatrix}
                            0 & 1 \\\\
                            1 & 0
                        \\end{pmatrix},&
                        F_y &= \\frac{1}{2}\\begin{pmatrix}
                            0 & -i \\\\
                            i &  0
                        \\end{pmatrix},&
                        F_z &= \\frac{1}{2}\\begin{pmatrix}
                            1 &  0  \\\\
                            0 & -1 
                        \\end{pmatrix}
                    \\end{align*}

                Then the exponential can be calculated as

                .. math::
                    \\begin{align*}
                        \\exp(A) &= \\exp(-ix F_x - iy F_y - iz F_z)\\\\
                        &= \\begin{pmatrix}
                            \\cos(\\frac{r}{2}) - i\\frac{z}{r}\\sin(\\frac{r}{2}) & -\\frac{y + ix}{r}\\sin(\\frac{r}{2})\\\\
                            \\frac{y - ix}{r}\\sin(\\frac{r}{2}) & \\cos(\\frac{r}{2}) + i\\frac{z}{r}\\sin(\\frac{r}{2})
                        \\end{pmatrix}
                    \\end{align*}

                with :math:`r = \\sqrt{x^2 + y^2 + z^2}`.

                Parameters
                ----------
                exponent : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to take the exponential of.
                    
                result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix which the result of the exponentiation is to be written to.
                """
                x = source_sample[0]
                y = source_sample[1]
                z = source_sample[2]

                r = math.sqrt(x**2 + y**2 + z**2)

                if r > 0:
                    x /= r
                    y /= r
                    z /= r

                    c = math.cos(r/2)
                    s = math.sin(r/2)

                    result[0, 0] = c - 1j*z*s
                    result[1, 0] = (y - 1j*x)*s
                    result[0, 1] = -(y + 1j*x)*s
                    result[1, 1] = c + 1j*z*s
                else:
                    result[0, 0] = 1
                    result[1, 0] = 0
                    result[0, 1] = 0
                    result[1, 1] = 1

            @jit_device
            def matrix_exponential_lie_trotter(source_sample, result, trotter_cutoff):
                """
                Calculates a matrix exponential based on the Lie Product Formula,

                .. math::
                    \\exp(A + B) = \\lim_{c \\to \\infty} \\left(\\exp\\left(\\frac{1}{c}A\\right) \\exp\\left(\\frac{1}{c}B\\right)\\right)^c.

                Assumes the exponent is an imaginary  linear combination of a subspace of :math:`su(3)`, being,

                .. math::
                    \\begin{align*}
                        A &= -i(x F_x + y F_y + z F_z + q F_q),
                    \\end{align*}
                
                with

                .. math::
                    \\begin{align*}
                        F_x &= \\frac{1}{\\sqrt{2}}\\begin{pmatrix}
                            0 & 1 & 0 \\\\
                            1 & 0 & 1 \\\\
                            0 & 1 & 0
                        \\end{pmatrix},&
                        F_y &= \\frac{1}{\\sqrt{2}}\\begin{pmatrix}
                            0 & -i &  0 \\\\
                            i &  0 & -i \\\\
                            0 &  i &  0
                        \\end{pmatrix},\\\\
                        F_z &= \\begin{pmatrix}
                            1 & 0 &  0 \\\\
                            0 & 0 &  0 \\\\
                            0 & 0 & -1
                        \\end{pmatrix},&
                        F_q &= \\frac{1}{3}\\begin{pmatrix}
                            1 &  0 & 0 \\\\
                            0 & -2 & 0 \\\\
                            0 &  0 & 1
                        \\end{pmatrix}
                    \\end{align*}

                Then the exponential can be approximated as, for large :math:`\\tau`,

                .. math::
                    \\begin{align*}
                        \\exp(A) &= \\exp(-ix F_x - iy F_y - iz F_z - iq F_q)\\\\
                        &= \\exp(2^{-\\tau}(-ix F_x - iy F_y - iz F_z - iq F_q))^{2^\\tau}\\\\
                        &\\approx (\\exp(-i(2^{-\\tau} x) F_x) \\exp(-i(2^{-\\tau} y) F_y) \\exp(-i(2^{-\\tau} z F_z + (2^{-\\tau} q) F_q)))^{2^\\tau}\\\\
                        &= \\begin{pmatrix}
                            \\frac{e^{-i\\left(Z + \\frac{Q}{3}\\right)}(c_X + c_Y - i s_Xs_Y)}{2} & \\frac{e^{i\\frac{2Q}{3}} (-s_Y -i c_Y s_X)}{\\sqrt{2}} & \\frac{e^{-i\\left(-Z + \\frac{Q}{3}\\right)}(c_X - c_Y + i s_Xs_Y)}{2} \\\\
                            \\frac{e^{-i\\left(Z + \\frac{Q}{3}\\right)} (-i s_X + c_X s_Y)}{\\sqrt{2}} & e^{i\\frac{2Q}{3}} c_X c_Y & \\frac{e^{-i(Z - \\frac{Q}{3})} (-i s_X - c_X s_Y)}{\\sqrt{2}} \\\\
                            \\frac{e^{-i\\left(Z + \\frac{Q}{3}\\right)}(c_X - c_Y - i s_Xs_Y)}{2} & \\frac{e^{i\\frac{2Q}{3}} (s_Y -i c_Y s_X)}{\\sqrt{2}} & \\frac{e^{-i\\left(-Z + \\frac{Q}{3}\\right)}(c_X + c_Y + i s_Xs_Y)}{2}
                        \\end{pmatrix}^{2^\\tau}\\\\
                        &= T^{2^\\tau},
                    \\end{align*}

                with

                .. math::
                    \\begin{align*}
                        X &= 2^{-\\tau}x,\\\\
                        Y &= 2^{-\\tau}y,\\\\
                        Z &= 2^{-\\tau}z,\\\\
                        Q &= 2^{-\\tau}q,\\\\
                        c_{\\theta} &= \\cos(\\theta),\\\\
                        s_{\\theta} &= \\sin(\\theta).
                    \\end{align*}
                
                Once :math:`T` is calculated, it is then recursively squared :math:`\\tau` times to obtain :math:`\\exp(A)`.

                Parameters
                ----------
                exponent : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to take the exponential of.

                    .. warning::
                        Will overwrite the original contents of this as part of the algorithm.
                    
                result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix which the result of the exponentiation is to be written to.
                trotter_cutoff : `int`
                    The number of squares to make to the approximate matrix (:math:`\\tau` above).
                """
                hyper_cube_amount = math.ceil(trotter_cutoff/2)
                if hyper_cube_amount < 0:
                    hyper_cube_amount = 0
                precision = 4**hyper_cube_amount
                
                x = source_sample[0]/(2*precision)
                y = source_sample[1]/(2*precision)
                z = source_sample[2]/(2*precision)

                cx = math.cos(x)
                sx = math.sin(x)
                cy = math.cos(y)
                sy = math.sin(y)

                cisz = math.cos(z) + 1j*math.sin(z)

                result[0, 0] = (cx*cy - 1j*sx*sy)/cisz
                result[1, 0] = (cx*sy -1j*sx*cy)/cisz

                result[0, 1] = -(cx*sy + 1j*sx*cy)*cisz
                result[1, 1] = (cx*cy + 1j*sx*sy)*cisz

                if device_index == 0:
                    temporary = np.empty((2, 2), dtype = np.complex128)
                elif device_index == 1:
                    temporary = cuda.local.array((2, 2), dtype = np.complex128)
                elif device_index == 2:
                    temporary_group = roc.shared.array((threads_per_block, 2, 2), dtype = np.complex128)
                    temporary = temporary_group[roc.get_local_id(1), :, :]
                for power_index in range(hyper_cube_amount):
                    matrix_multiply(result, result, temporary)
                    matrix_multiply(temporary, temporary, result)

            @jit_device
            def matrix_exponential_taylor(exponent, result, cutoff):
                """
                Calculate a matrix exponential using a Taylor series. The matrix being exponentiated is complex, and of any dimension.

                The exponential is approximated as

                .. math::
                    \\begin{align*}
                    \\exp(A) &= \\sum_{k = 0}^\\infty \\frac{1}{k!} A^k\\\\
                    &\\approx \\sum_{k = 0}^{c - 1} \\frac{1}{k!} A^k\\\\
                    &= \\sum_{k = 0}^{c - 1} T_k, \\textrm{ with}\\\\
                    T_0 &= 1,\\\\
                    T_k &= \\frac{1}{k} T_{k - 1} A.
                    \\end{align*}

                Parameters
                ----------
                exponent : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to take the exponential of.
                result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix which the result of the exponentiation is to be written to.
                cutoff : `int`
                    The number of terms in the Taylor expansion (:math:`c` above).
                """
                if device_index == 0:
                    T = np.empty((2, 2), dtype = np.complex128)
                    T_old = np.empty((2, 2), dtype = np.complex128)
                elif device_index == 1:
                    T = cuda.local.array((2, 2), dtype = np.complex128)
                    T_old = cuda.local.array((2, 2), dtype = np.complex128)
                elif device_index == 2:
                    T_group = roc.shared.array((threads_per_block, 2, 2), dtype = np.complex128)
                    T = T_group[roc.get_local_id(1), :, :]
                    T_old_group = roc.shared.array((threads_per_block, 2, 2), dtype = np.complex128)
                    T_old = T_old_group[roc.get_local_id(1), :, :]
                set_to_one(T)
                set_to_one(result)

                # exp(A) = 1 + A + A^2/2 + ...
                for taylor_index in range(cutoff):
                    # T_old = T
                    for x_index in nb.prange(2):
                        for y_index in nb.prange(2):
                            T_old[y_index, x_index] = T[y_index, x_index]
                    # T = T_old*A/n
                    for x_index in nb.prange(2):
                        for y_index in nb.prange(2):
                            T[y_index, x_index] = 0
                            for z_index in range(2):
                                T[y_index, x_index] += (T_old[y_index, z_index]*exponent[z_index, x_index])/(taylor_index + 1)
                    # E = E + T
                    for x_index in nb.prange(2):
                        for y_index in nb.prange(2):
                            result[y_index, x_index] += T[y_index, x_index]

        else:
            @jit_device
            def norm2(z):
                """
                The 2 norm of a complex vector.

                .. math::
                    \|a + ib\|_2 = \\sqrt {\\left(\\sum_i a_i^2 + b_i^2\\right)}

                Parameters
                ----------
                z : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (index)
                    The vector to take the 2 norm of.
                Returns
                -------
                nz : :class:`numpy.double`
                    The 2 norm of z.
                """
                # Original spin 1:
                return math.sqrt(z[0].real**2 + z[0].imag**2 + z[1].real**2 + z[1].imag**2 + z[2].real**2 + z[2].imag**2)

            @jit_device
            def cross(left, right, result):
                """
                The cross product of two vectors in :math:`\\mathbb{C}^3`.
                
                .. note::
                    The mathematics definition is used here rather than the physics definition. This is the conjugate of the real cross product, since this produces a vector orthogonal to the two inputs.

                .. math::
                    \\begin{align*}
                    (l \\times r)_1 &= (l_2 r_3 - l_3 r_2)^*,\\\\
                    (l \\times r)_2 &= (l_3 r_1 - l_1 r_3)^*,\\\\
                    (l \\times r)_3 &= (l_1 r_2 - l_2 r_1)^*,\\\\
                    l, r &\\in \\mathbb{C}^3
                    \\end{align*}

                Parameters
                ----------
                left : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (index)
                    The vector to left multiply in the cross product.
                right : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (index)
                    The vector to right multiply in the cross product.
                result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (index)
                    An array for the resultant vector to be written to.
                """
                result[0] = conj(left[1]*right[2] - left[2]*right[1])
                result[1] = conj(left[2]*right[0] - left[0]*right[2])
                result[2] = conj(left[0]*right[1] - left[1]*right[0])

            @jit_device
            def inner(left, right):
                """
                The inner (maths convention dot) product between two complex vectors. 
                
                .. note::
                    The mathematics definition is used here rather than the physics definition, so the left vector is conjugated. Thus the inner product of two orthogonal vectors is 0.

                .. math::
                    \\begin{align*}
                    l \\cdot r &\\equiv \\langle l, r \\rangle\\\\
                    l \\cdot r &= \\sum_i (l_i)^* r_i
                    \\end{align*}

                Parameters
                ----------
                left : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (index)
                    The vector to left multiply in the inner product.
                right : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (index)
                    The vector to right multiply in the inner product.
                
                Returns
                -------
                d : :class:`numpy.cdouble`
                    The inner product of l and r.
                """
                return conj(left[0])*right[0] + conj(left[1])*right[1] + conj(left[2])*right[2]
            
            @jit_device
            def set_to(operator, result):
                """
                Copy the contents of one matrix into another.

                .. math::
                    (A)_{i, j} = (B)_{i, j}

                Parameters
                ----------
                operator : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to copy from.
                result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to copy to.
                """
                result[0, 0] = operator[0, 0]
                result[1, 0] = operator[1, 0]
                result[2, 0] = operator[2, 0]

                result[0, 1] = operator[0, 1]
                result[1, 1] = operator[1, 1]
                result[2, 1] = operator[2, 1]

                result[0, 2] = operator[0, 2]
                result[1, 2] = operator[1, 2]
                result[2, 2] = operator[2, 2]

            @jit_device
            def set_to_one(operator):
                """
                Make a matrix the multiplicative identity, ie, :math:`1`.

                .. math::
                    \\begin{align*}
                    (A)_{i, j} &= \\delta_{i, j}\\\\
                    &= \\begin{cases}
                        1,&i = j\\\\
                        0,&i\\neq j
                    \\end{cases}
                    \\end{align*}

                Parameters
                ----------
                operator : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to set to :math:`1`.
                """
                operator[0, 0] = 1
                operator[1, 0] = 0
                operator[2, 0] = 0

                operator[0, 1] = 0
                operator[1, 1] = 1
                operator[2, 1] = 0

                operator[0, 2] = 0
                operator[1, 2] = 0
                operator[2, 2] = 1
            
            @jit_device
            def set_to_zero(operator):
                """
                Make a matrix the additive identity, ie, :math:`0`.

                .. math::
                    \\begin{align*}
                    (A)_{i, j} = 0
                    \\end{align*}

                Parameters
                ----------
                operator : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to set to :math:`0`.
                """
                operator[0, 0] = 0
                operator[1, 0] = 0
                operator[2, 0] = 0

                operator[0, 1] = 0
                operator[1, 1] = 0
                operator[2, 1] = 0

                operator[0, 2] = 0
                operator[1, 2] = 0
                operator[2, 2] = 0
            
            @jit_device
            def matrix_multiply(left, right, result):
                """
                Multiply matrices left and right together, to be returned in result.

                .. math::
                    \\begin{align*}
                    (LR)_{i,k} = \\sum_j (L)_{i,j} (R)_{j,k}
                    \\end{align*}

                Parameters
                ----------
                left : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to left multiply by.
                right : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to right multiply by.
                result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    A matrix to be filled with the result of the product.
                """
                result[0, 0] = left[0, 0]*right[0, 0] + left[0, 1]*right[1, 0] + left[0, 2]*right[2, 0]
                result[1, 0] = left[1, 0]*right[0, 0] + left[1, 1]*right[1, 0] + left[1, 2]*right[2, 0]
                result[2, 0] = left[2, 0]*right[0, 0] + left[2, 1]*right[1, 0] + left[2, 2]*right[2, 0]

                result[0, 1] = left[0, 0]*right[0, 1] + left[0, 1]*right[1, 1] + left[0, 2]*right[2, 1]
                result[1, 1] = left[1, 0]*right[0, 1] + left[1, 1]*right[1, 1] + left[1, 2]*right[2, 1]
                result[2, 1] = left[2, 0]*right[0, 1] + left[2, 1]*right[1, 1] + left[2, 2]*right[2, 1]

                result[0, 2] = left[0, 0]*right[0, 2] + left[0, 1]*right[1, 2] + left[0, 2]*right[2, 2]
                result[1, 2] = left[1, 0]*right[0, 2] + left[1, 1]*right[1, 2] + left[1, 2]*right[2, 2]
                result[2, 2] = left[2, 0]*right[0, 2] + left[2, 1]*right[1, 2] + left[2, 2]*right[2, 2]
            
            @jit_device
            def adjoint(operator, result):
                """
                Takes the hermitian adjoint of a matrix.

                .. math::
                    \\begin{align*}
                    A^\\dagger &\\equiv A^H\\\\
                    (A^\\dagger)_{y,x} &= ((A)_{x,y})^*
                    \\end{align*}
                
                Matrix can be in :math:`\\mathbb{C}^{2\\times2}` or :math:`\\mathbb{C}^{3\\times3}`.

                Parameters
                ----------
                operator : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The operator to take the adjoint of.
                result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    An array to write the resultant adjoint to.
                """
                result[0, 0] = conj(operator[0, 0])
                result[1, 0] = conj(operator[0, 1])
                result[2, 0] = conj(operator[0, 2])

                result[0, 1] = conj(operator[1, 0])
                result[1, 1] = conj(operator[1, 1])
                result[2, 1] = conj(operator[1, 2])

                result[0, 2] = conj(operator[2, 0])
                result[1, 2] = conj(operator[2, 1])
                result[2, 2] = conj(operator[2, 2])

            @jit_device
            def matrix_exponential_analytic(source_sample, result, trotter_cutoff):
                pass

            @jit_device
            def matrix_exponential_lie_trotter(source_sample, result, trotter_cutoff):
                """
                Calculates a matrix exponential based on the Lie Product Formula,

                .. math::
                    \\exp(A + B) = \\lim_{c \\to \\infty} \\left(\\exp\\left(\\frac{1}{c}A\\right) \\exp\\left(\\frac{1}{c}B\\right)\\right)^c.

                Assumes the exponent is an imaginary  linear combination of a subspace of :math:`su(3)`, being,

                .. math::
                    \\begin{align*}
                        A &= -i(x F_x + y F_y + z F_z + q F_q),
                    \\end{align*}
                
                with

                .. math::
                    \\begin{align*}
                        F_x &= \\frac{1}{\\sqrt{2}}\\begin{pmatrix}
                            0 & 1 & 0 \\\\
                            1 & 0 & 1 \\\\
                            0 & 1 & 0
                        \\end{pmatrix},&
                        F_y &= \\frac{1}{\\sqrt{2}}\\begin{pmatrix}
                            0 & -i &  0 \\\\
                            i &  0 & -i \\\\
                            0 &  i &  0
                        \\end{pmatrix},\\\\
                        F_z &= \\begin{pmatrix}
                            1 & 0 &  0 \\\\
                            0 & 0 &  0 \\\\
                            0 & 0 & -1
                        \\end{pmatrix},&
                        F_q &= \\frac{1}{3}\\begin{pmatrix}
                            1 &  0 & 0 \\\\
                            0 & -2 & 0 \\\\
                            0 &  0 & 1
                        \\end{pmatrix}
                    \\end{align*}

                Then the exponential can be approximated as, for large :math:`\\tau`,

                .. math::
                    \\begin{align*}
                        \\exp(A) &= \\exp(-ix F_x - iy F_y - iz F_z - iq F_q)\\\\
                        &= \\exp(2^{-\\tau}(-ix F_x - iy F_y - iz F_z - iq F_q))^{2^\\tau}\\\\
                        &\\approx (\\exp(-i(2^{-\\tau} x) F_x) \\exp(-i(2^{-\\tau} y) F_y) \\exp(-i(2^{-\\tau} z F_z + (2^{-\\tau} q) F_q)))^{2^\\tau}\\\\
                        &= \\begin{pmatrix}
                            \\frac{e^{-i\\left(Z + \\frac{Q}{3}\\right)}(c_X + c_Y - i s_Xs_Y)}{2} & \\frac{e^{i\\frac{2Q}{3}} (-s_Y -i c_Y s_X)}{\\sqrt{2}} & \\frac{e^{-i\\left(-Z + \\frac{Q}{3}\\right)}(c_X - c_Y + i s_Xs_Y)}{2} \\\\
                            \\frac{e^{-i\\left(Z + \\frac{Q}{3}\\right)} (-i s_X + c_X s_Y)}{\\sqrt{2}} & e^{i\\frac{2Q}{3}} c_X c_Y & \\frac{e^{-i(Z - \\frac{Q}{3})} (-i s_X - c_X s_Y)}{\\sqrt{2}} \\\\
                            \\frac{e^{-i\\left(Z + \\frac{Q}{3}\\right)}(c_X - c_Y - i s_Xs_Y)}{2} & \\frac{e^{i\\frac{2Q}{3}} (s_Y -i c_Y s_X)}{\\sqrt{2}} & \\frac{e^{-i\\left(-Z + \\frac{Q}{3}\\right)}(c_X + c_Y + i s_Xs_Y)}{2}
                        \\end{pmatrix}^{2^\\tau}\\\\
                        &= T^{2^\\tau},
                    \\end{align*}

                with

                .. math::
                    \\begin{align*}
                        X &= 2^{-\\tau}x,\\\\
                        Y &= 2^{-\\tau}y,\\\\
                        Z &= 2^{-\\tau}z,\\\\
                        Q &= 2^{-\\tau}q,\\\\
                        c_{\\theta} &= \\cos(\\theta),\\\\
                        s_{\\theta} &= \\sin(\\theta).
                    \\end{align*}
                
                Once :math:`T` is calculated, it is then recursively squared :math:`\\tau` times to obtain :math:`\\exp(A)`.

                Parameters
                ----------
                exponent : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to take the exponential of.

                    .. warning::
                        Will overwrite the original contents of this as part of the algorithm.
                    
                result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix which the result of the exponentiation is to be written to.
                trotter_cutoff : `int`
                    The number of squares to make to the approximate matrix (:math:`\\tau` above).
                """
                hyper_cube_amount = math.ceil(trotter_cutoff/2)
                if hyper_cube_amount < 0:
                    hyper_cube_amount = 0
                precision = 4**hyper_cube_amount
                
                x = source_sample[0]/precision
                y = source_sample[1]/precision
                z = source_sample[2]/precision
                q = source_sample[3]/precision

                cx = math.cos(x)
                sx = math.sin(x)
                cy = math.cos(y)
                sy = math.sin(y)

                cisz = math.cos(z + q/3) - 1j*math.sin(z + q/3)
                result[0, 0] = 0.5*cisz*(cx + cy - 1j*sx*sy)
                result[1, 0] = cisz*(-1j*sx + cx*sy)/sqrt2
                result[2, 0] = 0.5*cisz*(cx - cy - 1j*sx*sy)

                cisz = math.cos(2*q/3) + 1j*math.sin(2*q/3)
                result[0, 1] = cisz*(-sy - 1j*cy*sx)/sqrt2
                result[1, 1] = cisz*cx*cy
                result[2, 1] = cisz*(sy - 1j*cy*sx)/sqrt2

                cisz = math.cos(z - q/3) + 1j*math.sin(z - q/3)
                result[0, 2] = 0.5*cisz*(cx - cy + 1j*sx*sy)
                result[1, 2] = cisz*(-1j*sx - cx*sy)/sqrt2
                result[2, 2] = 0.5*cisz*(cx + cy + 1j*sx*sy)

                if device_index == 0:
                    temporary = np.empty((3, 3), dtype = np.complex128)
                elif device_index == 1:
                    temporary = cuda.local.array((3, 3), dtype = np.complex128)
                elif device_index == 2:
                    temporary_group = roc.shared.array((threads_per_block, 3, 3), dtype = np.complex128)
                    temporary = temporary_group[roc.get_local_id(1), :, :]
                for power_index in range(hyper_cube_amount):
                    matrix_multiply(result, result, temporary)
                    matrix_multiply(temporary, temporary, result)

            @jit_device
            def matrix_exponential_taylor(exponent, result, cutoff):
                """
                Calculate a matrix exponential using a Taylor series. The matrix being exponentiated is complex, and of any dimension.

                The exponential is approximated as

                .. math::
                    \\begin{align*}
                    \\exp(A) &= \\sum_{k = 0}^\\infty \\frac{1}{k!} A^k\\\\
                    &\\approx \\sum_{k = 0}^{c - 1} \\frac{1}{k!} A^k\\\\
                    &= \\sum_{k = 0}^{c - 1} T_k, \\textrm{ with}\\\\
                    T_0 &= 1,\\\\
                    T_k &= \\frac{1}{k} T_{k - 1} A.
                    \\end{align*}

                Parameters
                ----------
                exponent : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix to take the exponential of.
                result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
                    The matrix which the result of the exponentiation is to be written to.
                cutoff : `int`
                    The number of terms in the Taylor expansion (:math:`c` above).
                """
                if device_index == 0:
                    T = np.empty((3, 3), dtype = np.complex128)
                    T_old = np.empty((3, 3), dtype = np.complex128)
                elif device_index == 1:
                    T = cuda.local.array((3, 3), dtype = np.complex128)
                    T_old = cuda.local.array((3, 3), dtype = np.complex128)
                elif device_index == 2:
                    T_group = roc.shared.array((threads_per_block, 3, 3), dtype = np.complex128)
                    T = T_group[roc.get_local_id(1), :, :]
                    T_old_group = roc.shared.array((threads_per_block, 3, 3), dtype = np.complex128)
                    T_old = T_old_group[roc.get_local_id(1), :, :]
                set_to_one(T)
                set_to_one(result)

                # exp(A) = 1 + A + A^2/2 + ...
                for taylor_index in range(cutoff):
                    # T_old = T
                    for x_index in nb.prange(exponent.shape[0]):
                        for y_index in nb.prange(exponent.shape[0]):
                            T_old[y_index, x_index] = T[y_index, x_index]
                    # T = T_old*A/n
                    for x_index in nb.prange(exponent.shape[0]):
                        for y_index in nb.prange(exponent.shape[0]):
                            T[y_index, x_index] = 0
                            for z_index in range(exponent.shape[0]):
                                T[y_index, x_index] += (T_old[y_index, z_index]*exponent[z_index, x_index])/(taylor_index + 1)
                    # E = E + T
                    for x_index in nb.prange(exponent.shape[0]):
                        for y_index in nb.prange(exponent.shape[0]):
                            result[y_index, x_index] += T[y_index, x_index]
        self.conj = conj
        self.complex_abs = complex_abs
        self.norm2 = norm2
        self.inner = inner
        self.set_to = set_to
        self.set_to_one = set_to_one
        self.set_to_zero = set_to_zero
        self.matrix_multiply = matrix_multiply
        self.adjoint = adjoint
        self.matrix_exponential_analytic = matrix_exponential_analytic
        self.matrix_exponential_lie_trotter = matrix_exponential_lie_trotter
        self.matrix_exponential_taylor = matrix_exponential_taylor