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
    Options for the spin quantum number of a system.

    Parameters
    ----------
    value : :obj:`float`
        The numerical value of the spin quantum number.
    dimension : :obj:`int`
        Dimension of the hilbert space the states with this spin belong to.
    label : :obj:`str`
        A text label that can be used for archiving.
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
    value : :obj:`str`
        A text label that can be used for archiving.
    """

    MAGNUS_CF4 = "magnus_cf4"
    """
    Commutator free, fourth order Magnus based integrator.
    """

    MIDPOINT_SAMPLE = "midpoint_sample"
    """
    Euler integration method.
    """

    HALJ_STEP = "half_step"
    """
    Integration method from AtomicPy. Makes two Euler integration steps, one sampling the field from the start of the time step, one sampling the field from the end of the time step. The equivalent of the trapezoidal method.
    """

class ExponentiationMethod(Enum):
    """
    The implementation to use for matrix exponentiation within the integrator.

    Parameters
    ----------
    value : :obj:`str`
        A text label that can be used for archiving.
    index : :obj:`int`
        A reference number, used when compiling the integrator, where higher level objects like enums cannot be interpreted.
    """
    def __init__(self, value, index):
        super().__init__()
        self._value_ = value
        self.index = index

    ANALYTIC = ("analytic", 0)
    """
    Analytic expression of the matrix exponential. For spin half :obj:`SpinQuantumNumber.HALF` systems only.
    """

    LIE_TROTTER = ("lie_trotter", 1)
    """
    Approximation using the Lie Trotter theorem.
    """

class Device(Enum):
    """
    The target device that the integrator is being compiled for.

    .. _Supported Python features: http://numba.pydata.org/numba-doc/latest/reference/pysupported.html
    .. _Supported Numpy features: http://numba.pydata.org/numba-doc/latest/reference/numpysupported.html
    .. _Supported CUDA Python features: http://numba.pydata.org/numba-doc/latest/cuda/cudapysupported.html
    """
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
    """
    Use pure python interpreted code for the integrator, ie, don't compile the integrator.
    """

    CPU_SINGLE = ("cpu_single", 0)
    """
    Use the :func:`numba.jit()` LLVM compiler to compile the integrator to run on a single CPU core.

    .. note ::

        To use this device option, the user defined field function must be :func:`numba.jit()` compilable. See `Supported Python features`_ for compilable python features, and `Supported Numpy features`_ for compilable numpy features.
    """

    CPU = ("cpu", 0)
    """
    Use the :func:`numba.jit()` LLVM compiler to compile the integrator to run on all CPU cores, in parallel.

    .. note ::

        To use this device option, the user defined field function must be :func:`numba.jit()` compilable. See `Supported Python features`_ for compilable python features, and `Supported Numpy features`_ for compilable numpy features.
    """

    CUDA = ("cuda", 1)
    """
    Use the :func:`numba.cuda.jit()` LLVM compiler to compile the integrator to run on an Nvidia cuda compatible GPU, in parallel.

    .. note ::

        To use this device option, the user defined field function must be :func:`numba.cuda.jit()` compilable. See `Supported CUDA Python features`_ for compilable python features.
    """

    ROC = ("roc", 2)
    """
    Use the :func:`numba.roc.jit()` LLVM compiler to compile the integrator to run on an AMD ROCm compatible GPU, in parallel.

    .. warning ::

        Work in progress, not currently functional!

    """

class Simulator:
    """
    Attributes
    ----------
    spin_quantum_number : :obj:`SpinQuantumNumber`
            The option to select whether the simulator will integrate a spin half :obj:`SpinQuantumNumber.HALF`, or spin one :obj:`SpinQuantumNumber.ONE` quantum system.
    threads_per_block : :obj:`int`
        The size of each thread block (workgroup), in terms of the number of threads (workitems) they each contain, when running on the GPU target devices :obj:`Device.CUDA` (:obj:`Device.ROC`). Defaults to 64. Modifying might be able to increase execution time for different GPU models.
    device : :obj:`Device`
        The option to select which device will be targeted for integration. That is, whether the integrator is compiled for a CPU or GPU. Defaults to :obj:`Device.CUDA` if the system it is being run on is Nvidia Cuda compatible, and defaults to :obj:`Device.CPU` otherwise. See :obj:`Device` for all options and more details.
    get_time_evolution_raw : :obj:`callable`
        The internal function for evaluating the time evolution operator in parallel. Compiled for chosen device on object constrution.

        Parameters:

        * **field_modifier** (:obj:`float`) - The input to the `get_field` function supplied by the user. Modifies the field function so the integrator can be used for many experiments, without the need for slow recompilation. For example, if the `field_modifier` is used to define the bias field strength in `get_field`, then one can run many simulations, sweeping through bias values, by calling this method multiple times, each time varying `field_modifier`.   
        * **time_coarse**   (:obj:`numpy.ndarray` of :obj:`numpy.float64` (time_index)) - The times that `state` was evaluated at.
        * **time_end_points** (:obj:`numpy.ndarray` of :obj:`numpy.float64` (start/end)) - The time offset that the experiment is to start at, and the time that the experiment is to finish at. Measured in s.
        * **time_step_fine** (:obj:`float`) - The integration time step. Measured in s.
        * **time_step_coarse** (:obj:`float`) - The sample resolution of the output timeseries for the state. Must be a whole number multiple of `time_step_fine`. Measured in s.
        * **time_evolution_coarse** (:obj:`numpy.ndarray` of :obj:`numpy.float128` (time_index, y_index, x_index)) - The evaluated time evolution operator between each time step. See :ref:`architecture` for some information.

    get_spin_raw: :obj:`callable`
        The internal function for evaluating expected spin from the state in parallel. Compiled for chosen device on object constrution.

        Parameters:

        * **state** (:obj:`numpy.ndarray`) of :obj:`numpy.complex128` (time_index, magnetic_quantum_number)) - The quantum state of the spin system over time, written in terms of the eigenstates of the spin projection operator in the z direction.
        * **spin** (:obj:`numpy.ndarray` of :obj:`numpy.float64` (time_index, spatial_direction)) - The expected spin projection (Bloch vector) over time.
    """
    def __init__(self, get_field, spin_quantum_number, device = None, exponentiation_method = None, use_rotating_frame = True, integration_method = IntegrationMethod.MAGNUS_CF4, trotter_cutoff = 28, threads_per_block = 64, max_registers = 63):
        """
        .. _Achieved Occupancy: https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm

        Parameters
        ----------
        get_field : :obj:`callable`
            A python function that describes the field that the spin system is being put under. It must have three arguments:
            
            * **time_sample** (:obj:`float`) - the time to sample the field at, in units of s.
            * **simulation_index** (:obj:`int`) - a parameter that can be swept over when multiple simulations need to be run. For example, it is used to sweep over dressing frequencies during the simulations that `spinsim` was designed for.
            * **field_sample** (:class:`numpy.ndarray` of :class:`numpy.float64` (spatial_index)) the returned value of the field. This is a four dimensional vector, with the first three entries being x, y, z spatial directions (to model a magnetic field, for example), and the fourth entry being the amplitude of the quadratic shift (only appearing, and required, in spin one systems).

            .. note::
                This function must be compilable for the device that the integrator is being compiled for. See :class:`Device` for more information and links.

        spin_quantum_number : :obj:`SpinQuantumNumber`
            The option to select whether the simulator will integrate a spin half :obj:`SpinQuantumNumber.HALF`, or spin one :obj:`SpinQuantumNumber.ONE` quantum system.
        device : :obj:`Device`
            The option to select which device will be targeted for integration. That is, whether the integrator is compiled for a CPU or GPU. Defaults to :obj:`Device.CUDA` if the system it is being run on is Nvidia Cuda compatible, and defaults to :obj:`Device.CPU` otherwise. See :obj:`Device` for all options and more details.
        exponentiation_method : :obj:`ExponentiationMethod`
            Which method to use for matrix exponentiation in the integration algorithm. Defaults to :obj:`ExponentiationMethod.LIE_TROTTER` when `spin_quantum_number` is set to :obj:`SpinQuantumNumber.ONE`, and defaults to :obj:`ExponentiationMethod.ANALYTIC` when `spin_quantum_number` is set to :obj:`SpinQuantumNumber.HALF`. See :obj:`ExponentiationMethod` for more details.
        use_rotating_frame : :obj:`bool`
            Whether or not to use the rotating frame optimisation. Defaults to :obj:`True`. If set to :obj:`True`, the integrator moves into a frame rotating in the z axis by an amount defined by the field in the z direction. This removes the (possibly large) z component of the field, which increases the accuracy of the output since the integrator will on average take smaller steps.

            .. note ::

                The use of a rotating frame is commonly associated with the use of a rotating wave approximation, a technique used to get approximate analytic solutions of spin system dynamics. This is not done when this option is set to :obj:`True` - no such approximations are made, and the output state in given out of the rotating frame. One can, of course, use :mod:`spinsim` to integrate states in the rotating frame, using the rating wave approximation: just define `get_field()` with field functions that use the rotating wave approximation in the rotating frame.

        integration_method : :obj:`IntegrationMethod`
            Which integration method to use in the integration. Defaults to :obj:`IntegrationMethod.MAGNUS_CF4`. See :obj:`IntegrationMethod` for more details.
        trotter_cutoff : :obj:`int`
            The number of squares made by the matrix exponentiator, if :obj:`ExponentiationMethod.LIE_TROTTER` is chosen.
        threads_per_block : :obj:`int`
            The size of each thread block (workgroup), in terms of the number of threads (workitems) they each contain, when running on the GPU target devices :obj:`Device.CUDA` (:obj:`Device.ROC`). Defaults to 64. Modifying might be able to increase execution time for different GPU models.
        max_registers : :obj:`int`
            The maximum number of registers allocated per thread when using :obj:`Device.CUDA` as the target device, and can be modified to increase the execution speed for a specific GPU model. Defaults to 63 (optimal for GTX1070, the device used for testing. Note that one extra register per thread is always added to the number specified for control, so really this number is 64).
            
            Raising this value allocates more registers (fast memory) to each thread, out of a maximum number for the whole GPU, for each specific GPU model. This means that if more registers are allocated than are available for the GPU model, the GPU must run fewer threads concurrently than it has Cuda cores, meaning some cores are inactive, and the GPU is said to have less occupancy. Lowering the value increases GPU occupancy, meaning more threads run concurrently, at the expense of fewer resgiters being avaliable to each thread, meaning slower memory must be used. Thus, there will be an optimal value of `max_registers` for each model of GPU running :mod:`spinsim`, balancing more threads vs faster running threads, and changing this value could increase performance for your GPU. See `Achieved Occupancy`_ for Nvidia's official explanation.
        """
        if not device:
            if cuda.is_available():
                device = Device.CUDA
            else:
                device = Device.CPU

        self.threads_per_block = threads_per_block
        self.spin_quantum_number = spin_quantum_number
        self.device = device

        self.get_time_evolution_raw = None
        self.get_spin_raw = None
        try:
            self.compile_time_evolver(get_field, spin_quantum_number, device, use_rotating_frame, integration_method, exponentiation_method, trotter_cutoff, threads_per_block, max_registers)
        except:
            print("\033[31mspinsim error: numba could not jit get_field function into a device function.\033[0m\n")
            raise

    def compile_time_evolver(self, get_field, spin_quantum_number, device, use_rotating_frame = True, integration_method = IntegrationMethod.MAGNUS_CF4, exponentiation_method = None, trotter_cutoff = 28, threads_per_block = 64, max_registers = 63):
        """
        Compiles the integrator and spin calculation functions of the simulator.

        Parameters
        ----------
        get_field : :obj:`callable`
            A python function that describes the field that the spin system is being put under. It must have three arguments:
            
            * **time_sample** (:obj:`float`) - the time to sample the field at, in units of s.
            * **simulation_index** (:obj:`int`) - a parameter that can be swept over when multiple simulations need to be run. For example, it is used to sweep over dressing frequencies during the simulations that `spinsim` was designed for.
            * **field_sample** (:class:`numpy.ndarray` of :class:`numpy.float64` (spatial_index)) the returned value of the field. This is a four dimensional vector, with the first three entries being x, y, z spatial directions (to model a magnetic field, for example), and the fourth entry being the amplitude of the quadratic shift (only appearing, and required, in spin one systems).

            .. note::
                This function must be compilable for the device that the integrator is being compiled for. See :class:`Device` for more information and links.

        spin_quantum_number : :obj:`SpinQuantumNumber`
            The option to select whether the simulator will integrate a spin half :obj:`SpinQuantumNumber.HALF`, or spin one :obj:`SpinQuantumNumber.ONE` quantum system.
        device : :obj:`Device`
            The option to select which device will be targeted for integration. That is, whether the integrator is compiled for a CPU or GPU. Defaults to :obj:`Device.CUDA` if the system it is being run on is Nvidia Cuda compatible, and defaults to :obj:`Device.CPU` otherwise. See :obj:`Device` for all options and more details.
        exponentiation_method : :obj:`ExponentiationMethod`
            Which method to use for matrix exponentiation in the integration algorithm. Defaults to :obj:`ExponentiationMethod.LIE_TROTTER` when `spin_quantum_number` is set to :obj:`SpinQuantumNumber.ONE`, and defaults to :obj:`ExponentiationMethod.ANALYTIC` when `spin_quantum_number` is set to :obj:`SpinQuantumNumber.HALF`. See :obj:`ExponentiationMethod` for more details.
        use_rotating_frame : :obj:`bool`
            Whether or not to use the rotating frame optimisation. Defaults to :obj:`True`. If set to :obj:`True`, the integrator moves into a frame rotating in the z axis by an amount defined by the field in the z direction. This removes the (possibly large) z component of the field, which increases the accuracy of the output since the integrator will on average take smaller steps.

            .. note ::

                The use of a rotating frame is commonly associated with the use of a rotating wave approximation, a technique used to get approximate analytic solutions of spin system dynamics. This is not done when this option is set to :obj:`True` - no such approximations are made, and the output state in given out of the rotating frame. One can, of course, use :mod:`spinsim` to integrate states in the rotating frame, using the rating wave approximation: just define `get_field()` with field functions that use the rotating wave approximation in the rotating frame.

        integration_method : :obj:`IntegrationMethod`
            Which integration method to use in the integration. Defaults to :obj:`IntegrationMethod.MAGNUS_CF4`. See :obj:`IntegrationMethod` for more details.
        trotter_cutoff : :obj:`int`
            The number of squares made by the matrix exponentiator, if :obj:`ExponentiationMethod.LIE_TROTTER` is chosen.
        threads_per_block : :obj:`int`
            The size of each thread block (workgroup), in terms of the number of threads (workitems) they each contain, when running on the GPU target devices :obj:`Device.CUDA` (:obj:`Device.ROC`). Defaults to 64. Modifying might be able to increase execution time for different GPU models.
        max_registers : :obj:`int`
            The maximum number of registers allocated per thread when using :obj:`Device.CUDA` as the target device, and can be modified to increase the execution speed for a specific GPU model. Defaults to 63 (optimal for GTX1070, the device used for testing. Note that one extra register per thread is always added to the number specified for control, so really this number is 64).
            
            Raising this value allocates more registers (fast memory) to each thread, out of a maximum number for the whole GPU, for each specific GPU model. This means that if more registers are allocated than are available for the GPU model, the GPU must run fewer threads concurrently than it has Cuda cores, meaning some cores are inactive, and the GPU is said to have less occupancy. Lowering the value increases GPU occupancy, meaning more threads run concurrently, at the expense of fewer resgiters being avaliable to each thread, meaning slower memory must be used. Thus, there will be an optimal value of `max_registers` for each model of GPU running :mod:`spinsim`, balancing more threads vs faster running threads, and changing this value could increase performance for your GPU. See `Achieved Occupancy`_ for Nvidia's official explanation.
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

        if integration_method == IntegrationMethod.MAGNUS_CF4:
            sample_index_max = 3
            sample_index_end = 4
        elif integration_method == IntegrationMethod.HALJ_STEP:
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
        def append_exponentiation(field_sample, time_evolution_fine, time_evolution_coarse):
            if device_index == 0:
                time_evolution_old = np.empty((dimension, dimension), dtype = np.complex128)
            elif device_index == 1:
                time_evolution_old = cuda.local.array((dimension, dimension), dtype = np.complex128)
            elif device_index == 2:
                time_evolution_old_group = roc.shared.array((threads_per_block, dimension, dimension), dtype = np.complex128)
                time_evolution_old = time_evolution_old_group[roc.get_local_id(1), :, :]

            # Calculate the exponential
            if exponentiation_method_index == 0:
                matrix_exponential_analytic(field_sample, time_evolution_fine)
            elif exponentiation_method_index == 1:
                matrix_exponential_lie_trotter(field_sample, time_evolution_fine, trotter_cutoff)

            # Premultiply to the exitsing time evolution operator
            set_to(time_evolution_coarse, time_evolution_old)
            matrix_multiply(time_evolution_fine, time_evolution_old, time_evolution_coarse)

        if use_rotating_frame:
            if dimension == 3:
                @jit_device_template("(float64[:], float64, complex128)")
                def transform_frame_spin_one_rotating(field_sample, rotating_wave, rotating_wave_winding):
                    X = (field_sample[0] + 1j*field_sample[1])/rotating_wave_winding
                    field_sample[0] = X.real
                    field_sample[1] = X.imag
                    field_sample[2] = field_sample[2] - rotating_wave
                transform_frame = transform_frame_spin_one_rotating
            else:
                @jit_device_template("(float64[:], float64, complex128)")
                def transform_frame_spin_half_rotating(field_sample, rotating_wave, rotating_wave_winding):
                    X = (field_sample[0] + 1j*field_sample[1])/(rotating_wave_winding**2)
                    field_sample[0] = X.real
                    field_sample[1] = X.imag
                    field_sample[2] = field_sample[2] - 2*rotating_wave
                transform_frame = transform_frame_spin_half_rotating
        else:
            @jit_device_template("(float64[:], float64, complex128)")
            def transform_frame_lab(field_sample, rotating_wave, rotating_wave_winding):
                return
            transform_frame = transform_frame_lab

        get_field_jit = jit_device(get_field)

        if integration_method == IntegrationMethod.MAGNUS_CF4:
            @jit_device_template("(float64, float64, float64, float64, float64[:, :], float64, complex128[:])")
            def get_field_integration_magnus_cf4(field_modifier, time_fine, time_coarse, time_step_fine, field_sample, rotating_wave, rotating_wave_winding):
                time_sample = ((time_fine + 0.5*time_step_fine*(1 - 1/sqrt3)) - time_coarse)
                rotating_wave_winding[0] = math.cos(math.tau*rotating_wave*time_sample) + 1j*math.sin(math.tau*rotating_wave*time_sample)
                time_sample += time_coarse
                get_field_jit(time_sample, field_modifier, field_sample[0, :])

                time_sample = ((time_fine + 0.5*time_step_fine*(1 + 1/sqrt3)) - time_coarse)
                rotating_wave_winding[1] = math.cos(math.tau*rotating_wave*time_sample) + 1j*math.sin(math.tau*rotating_wave*time_sample)
                time_sample += time_coarse
                get_field_jit(time_sample, field_modifier, field_sample[1, :])

            @jit_device_template("(complex128[:, :], complex128[:, :], float64[:, :], float64, float64, complex128[:])")
            def append_exponentiation_integration_magnus_cf4(time_evolution_fine, time_evolution_coarse, field_sample, time_step_fine, rotating_wave, rotating_wave_winding):
                transform_frame(field_sample[0, :], rotating_wave, rotating_wave_winding[0])
                transform_frame(field_sample[1, :], rotating_wave, rotating_wave_winding[1])

                w0 = (1.5 + sqrt3)/6
                w1 = (1.5 - sqrt3)/6
                
                field_sample[2, 0] = math.tau*time_step_fine*(w0*field_sample[0, 0] + w1*field_sample[1, 0])
                field_sample[2, 1] = math.tau*time_step_fine*(w0*field_sample[0, 1] + w1*field_sample[1, 1])
                field_sample[2, 2] = math.tau*time_step_fine*(w0*field_sample[0, 2] + w1*field_sample[1, 2])
                if dimension > 2:
                    field_sample[2, 3] = math.tau*time_step_fine*(w0*field_sample[0, 3] + w1*field_sample[1, 3])

                append_exponentiation(field_sample[2, :], time_evolution_fine, time_evolution_coarse)

                field_sample[2, 0] = math.tau*time_step_fine*(w1*field_sample[0, 0] + w0*field_sample[1, 0])
                field_sample[2, 1] = math.tau*time_step_fine*(w1*field_sample[0, 1] + w0*field_sample[1, 1])
                field_sample[2, 2] = math.tau*time_step_fine*(w1*field_sample[0, 2] + w0*field_sample[1, 2])
                if dimension > 2:
                    field_sample[2, 3] = math.tau*time_step_fine*(w1*field_sample[0, 3] + w0*field_sample[1, 3])

                append_exponentiation(field_sample[2, :], time_evolution_fine, time_evolution_coarse)

            get_field_integration = get_field_integration_magnus_cf4
            append_exponentiation_integration = append_exponentiation_integration_magnus_cf4

        elif integration_method == IntegrationMethod.HALJ_STEP:
            @jit_device_template("(float64, float64, float64, float64, float64[:, :], float64, complex128[:])")
            def get_field_integration_half_step(field_modifier, time_fine, time_coarse, time_step_fine, field_sample, rotating_wave, rotating_wave_winding):
                time_sample = time_fine - time_coarse
                rotating_wave_winding[0] = math.cos(math.tau*rotating_wave*time_sample) + 1j*math.sin(math.tau*rotating_wave*time_sample)
                time_sample += time_coarse
                get_field_jit(time_sample, field_modifier, field_sample[0, :])

                time_sample = time_fine + time_step_fine - time_coarse
                rotating_wave_winding[1] = math.cos(math.tau*rotating_wave*time_sample) + 1j*math.sin(math.tau*rotating_wave*time_sample)
                time_sample += time_coarse
                get_field_jit(time_sample, field_modifier, field_sample[1, :])

            @jit_device_template("(complex128[:, :], complex128[:, :], float64[:, :], float64, float64, complex128[:])")
            def append_exponentiation_integration_half_step(time_evolution_fine, time_evolution_coarse, field_sample, time_step_fine, rotating_wave, rotating_wave_winding):
                transform_frame(field_sample[0, :], rotating_wave, rotating_wave_winding[0])
                transform_frame(field_sample[1, :], rotating_wave, rotating_wave_winding[1])
                
                field_sample[2, 0] = math.tau*time_step_fine*field_sample[0, 0]/2
                field_sample[2, 1] = math.tau*time_step_fine*field_sample[0, 1]/2
                field_sample[2, 2] = math.tau*time_step_fine*field_sample[0, 2]/2
                if dimension > 2:
                    field_sample[2, 3] = math.tau*time_step_fine*field_sample[0, 3]/2

                append_exponentiation(field_sample[2, :], time_evolution_fine, time_evolution_coarse)

                field_sample[2, 0] = math.tau*time_step_fine*field_sample[1, 0]/2
                field_sample[2, 1] = math.tau*time_step_fine*field_sample[1, 1]/2
                field_sample[2, 2] = math.tau*time_step_fine*field_sample[1, 2]/2
                if dimension > 2:
                    field_sample[2, 3] = math.tau*time_step_fine*field_sample[1, 3]/2

                append_exponentiation(field_sample[2, :], time_evolution_fine, time_evolution_coarse)

            get_field_integration = get_field_integration_half_step
            append_exponentiation_integration = append_exponentiation_integration_half_step

        elif integration_method == IntegrationMethod.MIDPOINT_SAMPLE:
            @jit_device_template("(float64, float64, float64, float64, float64[:, :], float64, complex128[:])")
            def get_field_integration_midpoint(field_modifier, time_fine, time_coarse, time_step_fine, field_sample, rotating_wave, rotating_wave_winding):
                time_sample = time_fine + 0.5*time_step_fine - time_coarse
                rotating_wave_winding[0] = math.cos(math.tau*rotating_wave*time_sample) + 1j*math.sin(math.tau*rotating_wave*time_sample)
                time_sample += time_coarse
                get_field_jit(time_sample, field_modifier, field_sample[0, :])

            @jit_device_template("(complex128[:, :], complex128[:, :], float64[:, :], float64, float64, complex128[:])")
            def append_exponentiation_integration_midpoint(time_evolution_fine, time_evolution_coarse, field_sample, time_step_fine, rotating_wave, rotating_wave_winding):
                transform_frame(field_sample[0, :], rotating_wave, rotating_wave_winding[0])
                
                field_sample[0, 0] = math.tau*time_step_fine*field_sample[0, 0]
                field_sample[0, 1] = math.tau*time_step_fine*field_sample[0, 1]
                field_sample[0, 2] = math.tau*time_step_fine*field_sample[0, 2]
                if dimension > 2:
                    field_sample[0, 3] = math.tau*time_step_fine*field_sample[0, 3]

                append_exponentiation(field_sample[0, :], time_evolution_fine, time_evolution_coarse)

            get_field_integration = get_field_integration_midpoint
            append_exponentiation_integration = append_exponentiation_integration_midpoint

        @jit_device_template("(int64, float64[:], float64, float64, float64[:], complex128[:, :, :], float64)")
        def get_time_evolution_loop(time_index, time_coarse, time_step_coarse, time_step_fine, time_end_points, time_evolution_coarse, field_modifier):
            # Declare variables
            if device_index == 0:
                time_evolution_fine = np.empty((dimension, dimension), dtype = np.complex128)

                field_sample = np.empty((sample_index_max, lie_dimension), dtype = np.float64)
                rotating_wave_winding = np.empty(sample_index_end, dtype = np.complex128)
            elif device_index == 1:
                time_evolution_fine = cuda.local.array((dimension, dimension), dtype = np.complex128)

                field_sample = cuda.local.array((sample_index_max, lie_dimension), dtype = np.float64)
                rotating_wave_winding = cuda.local.array(sample_index_end, dtype = np.complex128)
            elif device_index == 2:
                time_evolution_fine_group = roc.shared.array((threads_per_block, dimension, dimension), dtype = np.complex128)
                time_evolution_fine = time_evolution_fine_group[roc.get_local_id(1), :, :]

                field_sample_group = roc.shared.array((threads_per_block, sample_index_max, lie_dimension), dtype = np.float64)
                field_sample = field_sample_group[roc.get_local_id(1), :, :]
                rotating_wave_winding_group = roc.shared.array((threads_per_block, sample_index_end), dtype = np.complex128)
                rotating_wave_winding = rotating_wave_winding_group[roc.get_local_id(1), :]
            
            time_coarse[time_index] = time_end_points[0] + time_step_coarse*time_index
            time_fine = time_coarse[time_index]

            # Initialise time evolution operator to 1
            set_to_one(time_evolution_coarse[time_index, :])
            field_sample[0, 2] = 0
            if use_rotating_frame:
                time_sample = time_coarse[time_index] + time_step_coarse/2
                get_field_jit(time_sample, field_modifier, field_sample[0, :])
            rotating_wave = field_sample[0, 2]
            if dimension == 2:
                rotating_wave /= 2

            # For every fine step
            for time_fine_index in range(math.floor(time_step_coarse/time_step_fine + 0.5)):
                get_field_integration(field_modifier, time_fine, time_coarse[time_index], time_step_fine, field_sample, rotating_wave, rotating_wave_winding)
                append_exponentiation_integration(time_evolution_fine, time_evolution_coarse[time_index, :], field_sample, time_step_fine, rotating_wave, rotating_wave_winding)

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
        def get_time_evolution(field_modifier, time_coarse, time_end_points, time_step_fine, time_step_coarse, time_evolution_coarse):
            """
            Find the stepwise time evolution opperator.

            Parameters
            ----------
            field_modifier : :obj:`float`

            time_coarse : :class:`numpy.ndarray` of :class:`numpy.float64` (time_index)
                A coarse grained list of time samples that the time evolution operator is found for. In units of s. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numpy.ndarray` using :func:`numba.cuda.device_array_like()`.
            time_end_points : :class:`numpy.ndarray` of :class:`numpy.float64` (start time (0) or end time (1))
                The time values for when the experiment is to start and finishes. In units of s.
            time_step_fine : :obj:`float`
                The time step used within the integration algorithm. In units of s.
            time_step_coarse : :obj:`float`
                The time difference between each element of `time_coarse`. In units of s. Determines the sample rate of the outputs `time_coarse` and `time_evolution_coarse`.
            time_evolution_coarse : :class:`numpy.ndarray` of :class:`numpy.complex128` (time_index, bra_state_index, ket_state_index)
                Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overview_of_simulation_method`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numpy.ndarray` using :func:`numba.cuda.device_array_like()`.
            """

            if device_index == 0:
                for time_index in nb.prange(time_coarse.size):
                    get_time_evolution_loop(time_index, time_coarse, time_step_coarse, time_step_fine, time_end_points, time_evolution_coarse, field_modifier)
            elif device_index == 1:
                # Run calculation for each coarse timestep in parallel
                time_index = cuda.grid(1)
                if time_index < time_coarse.size:
                    get_time_evolution_loop(time_index, time_coarse, time_step_coarse, time_step_fine, time_end_points, time_evolution_coarse, field_modifier)
            elif device_index == 2:
                # Run calculation for each coarse timestep in parallel
                time_index = roc.get_global_id(1)
                if time_index < time_coarse.size:
                    get_time_evolution_loop(time_index, time_coarse, time_step_coarse, time_step_fine, time_end_points, time_evolution_coarse, field_modifier)
            return

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
            state : :class:`numpy.ndarray` of :class:`numpy.complex128` (time_index, state_index)
                The state (wavefunction) of the spin system in the lab frame, for each time sampled. See :math:`\\psi(t)` in :ref:`overview_of_simulation_method`.
            spin : :class:`numpy.ndarray` of :class:`numpy.float64` (time_index, spatial_index)
                The expected value for hyperfine spin of the spin system in the lab frame, for each time sampled. Units of :math:`\\hbar`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numpy.ndarray` using :func:`numba.cuda.device_array_like()`.
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

    def get_state(self, field_modifier, time_start, time_end, time_step_fine, time_step_coarse, state_init):
        """
        Integrates the time dependent Schroedinger equation and returns the quantum state of the spin system over time.

        Parameters
        ----------
        field_modifier : :obj:`float`
            The input to the `get_field` function supplied by the user. Modifies the field function so the integrator can be used for many experiments, without the need for slow recompilation. For example, if the `field_modifier` is used to define the bias field strength in `get_field`, then one can run many simulations, sweeping through bias values, by calling this method multiple times, each time varying `field_modifier`.
        time_start : :obj:`float`
            The time offset that the experiment is to start at. Measured in s.
        time_end : :obj:`float`
            The time that the experiment is to finish at. Measured in s. The duration of the experiment is `time_end - time_start`.
        time_step_fine : :obj:`float`
            The integration time step. Measured in s.
        time_step_coarse : :obj:`float`
            The sample resolution of the output timeseries for the state. Must be a whole number multiple of `time_step_fine`. Measured in s.
        state_init : :obj:`numpy.ndarray` of :obj:`numpy.complex128` (magnetic_quantum_number)
            The initial quantum state of the spin system, written in terms of the eigenstates of the spin projection operator in the z direction.

        Returns
        -------
        state : :obj:`numpy.ndarray` of :obj:`numpy.complex128` (time_index, magnetic_quantum_number)
            The evaluated quantum state of the spin system over time, written in terms of the eigenstates of the spin projection operator in the z direction.
        time : :obj:`numpy.ndarray` of :obj:`numpy.float64` (time_index)
            The times that `state` was evaluated at.
        """
        time_end_points = np.asarray([time_start, time_end], np.float64)
        state_init = np.asarray(state_init, np.complex128)

        time_index_max = int((time_end_points[1] - time_end_points[0])/time_step_coarse)
        if self.device.index == 0:
            time = np.empty(time_index_max, np.float64)
            self.time_evolution_coarse = np.empty((time_index_max, self.spin_quantum_number.dimension, self.spin_quantum_number.dimension), np.complex128)

            self.get_time_evolution_raw(field_modifier, time, time_end_points, time_step_fine, time_step_coarse, self.time_evolution_coarse)

        elif self.device == Device.CUDA:
            time = cuda.device_array(time_index_max, np.float64)
            self.time_evolution_coarse = cuda.device_array((time_index_max, self.spin_quantum_number.dimension, self.spin_quantum_number.dimension), np.complex128)

            blocks_per_grid = (time.size + (self.threads_per_block - 1)) // self.threads_per_block
            try:
                self.get_time_evolution_raw[blocks_per_grid, self.threads_per_block](field_modifier, time, time_end_points, time_step_fine, time_step_coarse, self.time_evolution_coarse)
            except:
                print("\033[31mspinsim error: numba.cuda could not jit get_field function into a cuda device function.\033[0m\n")
                raise

            self.time_evolution_coarse = self.time_evolution_coarse.copy_to_host()
            time = time.copy_to_host()
        
        elif self.device == Device.ROC:
            time = roc.device_array(time_index_max, np.float64)
            self.time_evolution_coarse = roc.device_array((time_index_max, self.spin_quantum_number.dimension, self.spin_quantum_number.dimension), np.complex128)

            blocks_per_grid = (time.size + (self.threads_per_block - 1)) // self.threads_per_block
            try:
                self.get_time_evolution_raw[blocks_per_grid, self.threads_per_block](field_modifier, time, time_end_points, time_step_fine, time_step_coarse, self.time_evolution_coarse)
            except:
                print("\033[31mspinsim error: numba.roc could not jit get_field function into a roc device function.\033[0m\n")
                raise

            self.time_evolution_coarse = self.time_evolution_coarse.copy_to_host()
            time = time.copy_to_host()

        state = np.empty((time_index_max, self.spin_quantum_number.dimension), np.complex128)
        get_state(state_init, state, self.time_evolution_coarse)

        return state, time

    def get_spin(self, state):
        """
        Calculates the expected spin projection (Bloch vector) over time for a given time series of a quantum state.

        Parameters
        ----------
        state : :obj:`numpy.ndarray` of :obj:`numpy.complex128` (time_index, magnetic_quantum_number)
            The quantum state of the spin system over time, written in terms of the eigenstates of the spin projection operator in the z direction.

        Returns
        -------
        spin : :obj:`numpy.ndarray` of :obj:`numpy.float64` (time_index, spatial_direction)
            The expected spin projection (Bloch vector) over time.
        """
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
    state_init : :class:`numpy.ndarray` of :class:`numpy.complex128`
        The state (spin wavefunction) of the system at the start of the simulation.
    state : :class:`numpy.ndarray` of :class:`numpy.complex128` (time_index, state_index)
        The state (wavefunction) of the spin system in the lab frame, for each time sampled. See :math:`\\psi(t)` in :ref:`overview_of_simulation_method`. This is an output.
    time_evolution : :class:`numpy.ndarray` of :class:`numpy.complex128` (time_index, bra_state_index, ket_state_index)
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
    """
    A on object that contains definitions of all of the device functions (functions compiled for use on the target device) used in the integrator. These device functions are compiled for the chosen target device on construction of the object.

    Attributes
    ----------
    conj(z) : :obj:`callable`
        Conjugate of a complex number.

        .. math::
            \\begin{align*}
            (a + ib)^* &= a - ib\\\\
            a, b &\\in \\mathbb{R}
            \\end{align*}

        Parameters

        * **z** (:class:`numpy.complex128`) - The complex number to take the conjugate of.
        
        Returns

        * **cz** (:class:`numpy.complex128`) - The conjugate of z.

    complex_abs(z) : :obj:`callable`
        The absolute value of a complex number.

        .. math::
            \\begin{align*}
            |a + ib| &= \\sqrt{a^2 + b^2}\\\\
            a, b &\\in \\mathbb{R}
            \\end{align*}
        
        Parameters

        * **z** (:class:`numpy.complex128`) - The complex number to take the absolute value of.
        
        Returns

        * **az** (:class:`numpy.float64`) - The absolute value of z.

    norm2(z) : :obj:`callable`
        The 2 norm of a complex vector.

        .. math::
            \|a + ib\|_2 = \\sqrt {\\left(\\sum_i a_i^2 + b_i^2\\right)}

        Parameters
        
        * **z** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (index)) - The vector to take the 2 norm of.

        Returns
        
        * **nz** (:class:`numpy.float64`) - The 2 norm of z.

    inner(left, right) : :obj:`callable`
        The inner (maths convention dot) product between two complex vectors. 
                
        .. note::
            The mathematics definition is used here rather than the physics definition, so the left vector is conjugated. Thus the inner product of two orthogonal vectors is 0.

        .. math::
            \\begin{align*}
            l \\cdot r &\\equiv \\langle l, r \\rangle\\\\
            l \\cdot r &= \\sum_i (l_i)^* r_i
            \\end{align*}

        Parameters
        
        * **left** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (index)) - The vector to left multiply in the inner product.
        * **right** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (index)) - The vector to right multiply in the inner product.
        
        Returns
        
        * **d** (:class:`numpy.complex128`) - The inner product of l and r.
        
    set_to(operator, result) : :obj:`callable`
        Copy the contents of one matrix into another.

        .. math::
            (A)_{i, j} = (B)_{i, j}

        Parameters
        
        * **operator** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The matrix to copy from.
        * **result** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The matrix to copy to.

    set_to_one(operator) : :obj:`callable`
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
        
        * **operator** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The matrix to set to :math:`1`.

    set_to_zero(operator) : :obj:`callable`
        Make a matrix the additive identity, ie, :math:`0`.

        .. math::
            \\begin{align*}
            (A)_{i, j} = 0
            \\end{align*}

        Parameters
        
        * **operator** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The matrix to set to :math:`0`.

    matrix_multiply(left, right, result) : :obj:`callable`
        Multiply matrices left and right together, to be returned in result.

        .. math::
            \\begin{align*}
            (LR)_{i,k} = \\sum_j (L)_{i,j} (R)_{j,k}
            \\end{align*}

        Parameters
        
        * **left** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The matrix to left multiply by.
        * **right** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The matrix to right multiply by.
        * **result** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - A matrix to be filled with the result of the product.

    adjoint(operator) : :obj:`callable`
        Takes the hermitian adjoint of a matrix.

        .. math::
            \\begin{align*}
            A^\\dagger &\\equiv A^H\\\\
            (A^\\dagger)_{y,x} &= ((A)_{x,y})^*
            \\end{align*}
        
        Matrix can be in :math:`\\mathbb{C}^{2\\times2}` or :math:`\\mathbb{C}^{3\\times3}`.

        Parameters
        
        * **operator** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The operator to take the adjoint of.
        * **result** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - An array to write the resultant adjoint to.
            
    matrix_exponential_analytic(field_sample, result) : :obj:`callable`
        Calculates a :math:`\\mathfrak{su}(2)` matrix exponential based on its analytic form.

        .. warning::
            
            Only available for use with spin half systems. Will not work with spin one systems.

        Assumes the exponent is an imaginary  linear combination of :math:`\\mathfrak{su}(2)`, being,

        .. math::
            \\begin{align*}
                A &= -i(x J_x + y J_y + z J_z),
            \\end{align*}
        
        with

        .. math::
            \\begin{align*}
                J_x &= \\frac{1}{2}\\begin{pmatrix}
                    0 & 1 \\\\
                    1 & 0
                \\end{pmatrix},&
                J_y &= \\frac{1}{2}\\begin{pmatrix}
                    0 & -i \\\\
                    i &  0
                \\end{pmatrix},&
                J_z &= \\frac{1}{2}\\begin{pmatrix}
                    1 &  0  \\\\
                    0 & -1 
                \\end{pmatrix}
            \\end{align*}

        Then the exponential can be calculated as

        .. math::
            \\begin{align*}
                \\exp(A) &= \\exp(-ix J_x - iy J_y - iz J_z)\\\\
                &= \\begin{pmatrix}
                    \\cos(\\frac{r}{2}) - i\\frac{z}{r}\\sin(\\frac{r}{2}) & -\\frac{y + ix}{r}\\sin(\\frac{r}{2})\\\\
                    \\frac{y - ix}{r}\\sin(\\frac{r}{2}) & \\cos(\\frac{r}{2}) + i\\frac{z}{r}\\sin(\\frac{r}{2})
                \\end{pmatrix}
            \\end{align*}

        with :math:`r = \\sqrt{x^2 + y^2 + z^2}`.

        Parameters
        
        * **field_sample** (:class:`numpy.ndarray` of :class:`numpy.float64`, (y_index, x_index)) - The values of x, y and z respectively, as described above.
        * **result** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The matrix which the result of the exponentiation is to be written to.

    matrix_exponential_lie_trotter(field_sample, result) : :obj:`callable`
        Calculates a matrix exponential based on the Lie Product Formula,

        .. math::
            \\exp(A + B) = \\lim_{c \\to \\infty} \\left(\\exp\\left(\\frac{1}{c}A\\right) \\exp\\left(\\frac{1}{c}B\\right)\\right)^c.

        **For spin half systems:**

        Assumes the exponent is an imaginary  linear combination of a subspace of :math:`\\mathfrak{su}(2)`, being,

        .. math::
            \\begin{align*}
                A &= -i(x J_x + y J_y + z J_z),
            \\end{align*}
        
        with

        .. math::
            \\begin{align*}
                J_x &= \\frac{1}{2}\\begin{pmatrix}
                    0 & 1 \\\\
                    1 & 0
                \\end{pmatrix},&
                J_y &= \\frac{1}{2}\\begin{pmatrix}
                    0 & -i \\\\
                    i &  0
                \\end{pmatrix},&
                J_z &= \\frac{1}{2}\\begin{pmatrix}
                    1 &  0  \\\\
                    0 & -1 
                \\end{pmatrix}
            \\end{align*}

        Then the exponential can be approximated as, for large :math:`\\tau`,

        .. math::
            \\begin{align*}
                \\exp(A) &= \\exp(-ix J_x - iy J_y - iz J_z)\\\\
                &= \\exp(2^{-\\tau}(-ix J_x - iy J_y - iz J_z))^{2^\\tau}\\\\
                &\\approx (\\exp(-i(2^{-\\tau} x) J_x) \\exp(-i(2^{-\\tau} y) J_y) \\exp(-i(2^{-\\tau} z) J_z)^{2^\\tau}\\\\
                &= \\begin{pmatrix}
                    (c_Xc_Y - is_Xs_Y) e^{-iZ} &
                    -(c_Xs_Y + is_Xc_Y) e^{iZ} \\\\
                    (c_Xs_Y - is_Xc_Y) e^{-iZ} &
                    (c_Xc_Y + is_Xs_Y) e^{iZ}
                \\end{pmatrix}^{2^\\tau}\\\\
                &= T^{2^\\tau},
            \\end{align*}

        with

        .. math::
            \\begin{align*}
                X &= \\frac{1}{2}2^{-\\tau}x,\\\\
                Y &= \\frac{1}{2}2^{-\\tau}y,\\\\
                Z &= \\frac{1}{2}2^{-\\tau}z,\\\\
                c_{\\theta} &= \\cos(\\theta),\\\\
                s_{\\theta} &= \\sin(\\theta).
            \\end{align*}

        **For spin one systems**

        Assumes the exponent is an imaginary  linear combination of a subspace of :math:`\\mathfrak{su}(3)`, being,

        .. math::
            \\begin{align*}
                A &= -i(x J_x + y J_y + z J_z + q J_q),
            \\end{align*}
        
        with

        .. math::
            \\begin{align*}
                J_x &= \\frac{1}{\\sqrt{2}}\\begin{pmatrix}
                    0 & 1 & 0 \\\\
                    1 & 0 & 1 \\\\
                    0 & 1 & 0
                \\end{pmatrix},&
                J_y &= \\frac{1}{\\sqrt{2}}\\begin{pmatrix}
                    0 & -i &  0 \\\\
                    i &  0 & -i \\\\
                    0 &  i &  0
                \\end{pmatrix},\\\\
                J_z &= \\begin{pmatrix}
                    1 & 0 &  0 \\\\
                    0 & 0 &  0 \\\\
                    0 & 0 & -1
                \\end{pmatrix},&
                J_q &= \\frac{1}{3}\\begin{pmatrix}
                    1 &  0 & 0 \\\\
                    0 & -2 & 0 \\\\
                    0 &  0 & 1
                \\end{pmatrix}
            \\end{align*}

        Then the exponential can be approximated as, for large :math:`\\tau`,

        .. math::
            \\begin{align*}
                \\exp(A) &= \\exp(-ix J_x - iy J_y - iz J_z - iq J_q)\\\\
                &= \\exp(2^{-\\tau}(-ix J_x - iy J_y - iz J_z - iq J_q))^{2^\\tau}\\\\
                &\\approx (\\exp(-i(2^{-\\tau} x) J_x) \\exp(-i(2^{-\\tau} y) J_y) \\exp(-i(2^{-\\tau} z J_z + (2^{-\\tau} q) J_q)))^{2^\\tau}\\\\
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
        
        * **field_sample** (:class:`numpy.ndarray` of :class:`numpy.float64`, (y_index, x_index)) - The values of x, y and z (and q for spin one) respectively, as described above.
        * **result** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The matrix which the result of the exponentiation is to be written to.
        * **trotter_cutoff** (:obj:`int`) - The number of squares to make to the approximate matrix (:math:`\\tau` above).

    """
    def __init__(self, spin_quantum_number, device, threads_per_block):
        """
        Parameters
        ----------
        spin_quantum_number : :obj:`SpinQuantumNumber`
            The option to select whether the simulator will integrate a spin half :obj:`SpinQuantumNumber.HALF`, or spin one :obj:`SpinQuantumNumber.ONE` quantum system.
        device : :obj:`Device`
            The option to select which device will be targeted for integration. That is, whether the integrator is compiled for a CPU or GPU. Defaults to :obj:`Device.CUDA` if the system it is being run on is Nvidia Cuda compatible, and defaults to :obj:`Device.CPU` otherwise. See :obj:`Device` for all options and more details.
        threads_per_block : :obj:`int`
            The size of each thread block (workgroup), in terms of the number of threads (workitems) they each contain, when running on the GPU target devices :obj:`Device.CUDA` (:obj:`Device.ROC`). Defaults to 64. Modifying might be able to increase execution time for different GPU models.
        """
        jit_device = device.jit_device
        device_index = device.index

        @jit_device
        def conj(z):
            return (z.real - 1j*z.imag)

        @jit_device
        def complex_abs(z):
            return math.sqrt(z.real**2 + z.imag**2)

        if spin_quantum_number == SpinQuantumNumber.HALF:
            @jit_device
            def norm2(z):
                return math.sqrt(z[0].real**2 + z[0].imag**2 + z[1].real**2 + z[1].imag**2)

            @jit_device
            def inner(left, right):
                return conj(left[0])*right[0] + conj(left[1])*right[1]

            @jit_device
            def set_to(operator, result):
                result[0, 0] = operator[0, 0]
                result[1, 0] = operator[1, 0]

                result[0, 1] = operator[0, 1]
                result[1, 1] = operator[1, 1]

            @jit_device
            def set_to_one(operator):
                operator[0, 0] = 1
                operator[1, 0] = 0

                operator[0, 1] = 0
                operator[1, 1] = 1

            @jit_device
            def set_to_zero(operator):
                operator[0, 0] = 0
                operator[1, 0] = 0

                operator[0, 1] = 0
                operator[1, 1] = 0

            @jit_device
            def matrix_multiply(left, right, result):
                result[0, 0] = left[0, 0]*right[0, 0] + left[0, 1]*right[1, 0]
                result[1, 0] = left[1, 0]*right[0, 0] + left[1, 1]*right[1, 0]

                result[0, 1] = left[0, 0]*right[0, 1] + left[0, 1]*right[1, 1]
                result[1, 1] = left[1, 0]*right[0, 1] + left[1, 1]*right[1, 1]

            @jit_device
            def adjoint(operator, result):
                result[0, 0] = conj(operator[0, 0])
                result[1, 0] = conj(operator[0, 1])

                result[0, 1] = conj(operator[1, 0])
                result[1, 1] = conj(operator[1, 1])

            @jit_device
            def matrix_exponential_analytic(field_sample, result):
                x = field_sample[0]
                y = field_sample[1]
                z = field_sample[2]

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
            def matrix_exponential_lie_trotter(field_sample, result, trotter_cutoff):
                hyper_cube_amount = math.ceil(trotter_cutoff/2)
                if hyper_cube_amount < 0:
                    hyper_cube_amount = 0
                precision = 4**hyper_cube_amount
                
                x = field_sample[0]/(2*precision)
                y = field_sample[1]/(2*precision)
                z = field_sample[2]/(2*precision)

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

        else:
            @jit_device
            def norm2(z):
                return math.sqrt(z[0].real**2 + z[0].imag**2 + z[1].real**2 + z[1].imag**2 + z[2].real**2 + z[2].imag**2)

            @jit_device
            def cross(left, right, result):
                result[0] = conj(left[1]*right[2] - left[2]*right[1])
                result[1] = conj(left[2]*right[0] - left[0]*right[2])
                result[2] = conj(left[0]*right[1] - left[1]*right[0])

            @jit_device
            def inner(left, right):
                return conj(left[0])*right[0] + conj(left[1])*right[1] + conj(left[2])*right[2]
            
            @jit_device
            def set_to(operator, result):
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
            def matrix_exponential_analytic(field_sample, result, trotter_cutoff):
                pass

            @jit_device
            def matrix_exponential_lie_trotter(field_sample, result, trotter_cutoff):
                hyper_cube_amount = math.ceil(trotter_cutoff/2)
                if hyper_cube_amount < 0:
                    hyper_cube_amount = 0
                precision = 4**hyper_cube_amount
                
                x = field_sample[0]/precision
                y = field_sample[1]/precision
                z = field_sample[2]/precision
                q = field_sample[3]/precision

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