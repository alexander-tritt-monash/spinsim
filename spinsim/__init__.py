"""

"""
from enum import Enum
import numpy as np
import numba as nb
from numba import cuda
from numba import roc
import math
import cmath

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
    plus_x, plus_y, plus_z, zero_x, zero_y, zero_z, minus_x, minus_y, minus_z : :obj:`numpy.ndarray` of :obj:`numpy.complex128`
        Eigenstates of the spin operators for quick reference.
    """

    def __init__(self, value:np.float64, dimension:int, label:str):
        super().__init__()
        self._value_ = value
        self.dimension = dimension
        self.label = label
        if self.label == "half":
            self.plus_x = np.array([1, 1], np.complex128)/math.sqrt(2)
            self.minus_x = np.array([-1, 1], np.complex128)/math.sqrt(2)

            self.plus_y = np.array([1, 1j], np.complex128)/math.sqrt(2)
            self.minus_y = np.array([1, -1j], np.complex128)/math.sqrt(2)

            self.plus_z = np.array([1, 0], np.complex128)
            self.minus_z = np.array([0, 1], np.complex128)
        else:
            self.plus_x = np.array([1, math.sqrt(2), 1], np.complex128)/2
            self.zero_x = np.array([-1, 0, 1], np.complex128)/math.sqrt(2)
            self.minus_x = np.array([1, -math.sqrt(2), 1], np.complex128)/2

            self.plus_y = np.array([-1, -1j*math.sqrt(2), 1], np.complex128)/2
            self.zero_y = np.array([1, 0, 1], np.complex128)/math.sqrt(2)
            self.minus_y = np.array([1, -1j*math.sqrt(2), 1], np.complex128)/2

            self.plus_z = np.array([1, 0, 0], np.complex128)
            self.zero_z = np.array([0, 1, 0], np.complex128)
            self.minus_z = np.array([0, 0, 1], np.complex128)

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

    EULER = "euler"
    """
    Euler integration method.
    """

    HEUN = "heun"
    """
    Integration method from AtomicPy.
    Makes two Euler integration steps, one sampling the field from the start of the time step, one sampling the field from the end of the time step.
    The equivalent of the trapezoidal method.
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
    def __init__(self, value:str, index:int):
        super().__init__()
        self._value_ = value
        self.index = index

    ANALYTIC = ("analytic", 0)
    """
    Analytic expression of the matrix exponential.
    For spin-half :obj:`SpinQuantumNumber.HALF` systems only.
    See :obj:`Utilities.matrix_exponential_analytic()` for more information.
    """

    LIE_TROTTER = ("lie_trotter", 1)
    """
    Approximation using the Lie Trotter theorem, using the Pauli matrices and a single quadratic operator.
    See :obj:`Utilities.matrix_exponential_lie_trotter()` for more information.
    """

    LIE_TROTTER_8 = ("lie_trotter_8", 2)
    """
    Approximation using the Lie Trotter theorem, using all basis elements of su(3).
    For spin-one :obj:`SpinQuantumNumber.HALF` systems only.
    See :obj:`Utilities.matrix_exponential_lie_trotter_8()` for more information.
    """

class Device(Enum):
    """
    The target device that the integrator is being compiled for.

    .. _Supported Python features: http://numba.pydata.org/numba-doc/latest/reference/pysupported.html
    .. _Supported Numpy features: http://numba.pydata.org/numba-doc/latest/reference/numpysupported.html
    .. _Supported CUDA Python features: http://numba.pydata.org/numba-doc/latest/cuda/cudapysupported.html
    """
    def __init__(self, value:str, index:int):
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

class Results:
    """
    The results of a an evaluation of the integrator.

    Attributes
    ----------
    time : :obj:`numpy.ndarray` of :obj:`numpy.float64` (time_index)
        The times that `state` was evaluated at.
    time_evolution : :obj:`numpy.ndarray` of :obj:`numpy.float128` (time_index, y_index, x_index)
        The evaluated time evolution operator between each time step.
        See :ref:`architecture` for some information.
    state : :obj:`numpy.ndarray` of :obj:`numpy.complex128` (time_index, magnetic_quantum_number)
        The evaluated quantum state of the spin system over time, written in terms of the eigenstates of the spin projection operator in the z direction.
    spin : :obj:`numpy.ndarray` of :obj:`numpy.float64` (time_index, spatial_direction)
        The expected spin projection (Bloch vector) over time.
        This is calculated just in time using the JITed :obj:`callable` `spin_calculator`.
    spin_calculator : :obj:`callable`
        Calculates the expected spin projection (Bloch vector) over time for a given time series of a quantum state.
        Used to calculate `spin` the first time it is referenced by the user.

        Parameters:
        
        * **state** (:obj:`numpy.ndarray` of :obj:`numpy.complex128` (time_index, magnetic_quantum_number)) - The quantum state of the spin system over time, written in terms of the eigenstates of the spin projection operator in the z direction.

        Returns:
        
        * **spin** (:obj:`numpy.ndarray` of :obj:`numpy.float64` (time_index, spatial_direction)) - The expected spin projection (Bloch vector) over time.
    """
    def __init__(self, time:np.ndarray, time_evolution:np.ndarray, state:np.ndarray, spin_calculator:callable):
        """
        Parameters
        ----------
        time : :obj:`numpy.ndarray` of :obj:`numpy.float64` (time_index)
            The times that `state` was evaluated at.
        time_evolution : :obj:`numpy.ndarray` of :obj:`numpy.float128` (time_index, y_index, x_index)
            The evaluated time evolution operator between each time step.
            See :ref:`architecture` for some information.
        state : :obj:`numpy.ndarray` of :obj:`numpy.complex128` (time_index, magnetic_quantum_number)
            The evaluated quantum state of the spin system over time, written in terms of the eigenstates of the spin projection operator in the z direction.
        spin_calculator : :obj:`callable`
            Calculates the expected spin projection (Bloch vector) over time for a given time series of a quantum state.
            Used to calculate `spin` the first time it is referenced by the user.

            Parameters:
            
            * **state** (:obj:`numpy.ndarray` of :obj:`numpy.complex128` (time_index, magnetic_quantum_number)) - The quantum state of the spin system over time, written in terms of the eigenstates of the spin projection operator in the z direction.

            Returns:
            
            * **spin** (:obj:`numpy.ndarray` of :obj:`numpy.float64` (time_index, spatial_direction)) - The expected spin projection (Bloch vector) over time.
        """
        self.time = time
        self.time_evolution = time_evolution
        self.state = state
        self.spin_calculator = spin_calculator

    def __getattr__(self, attr_name:str) -> np.ndarray:
        if attr_name == "spin":
            spin = self.spin_calculator(self.state)
            setattr(self, attr_name, spin)
            return self.spin
        raise AttributeError("{} has no attribute called {}.".format(self, attr_name))

class Simulator:
    """
    Attributes
    ----------
    spin_quantum_number : :obj:`SpinQuantumNumber`
            The option to select whether the simulator will integrate a spin-half :obj:`SpinQuantumNumber.HALF`, or spin-one :obj:`SpinQuantumNumber.ONE` quantum system.
    threads_per_block : :obj:`int`
        The size of each thread block (workgroup), in terms of the number of threads (workitems) they each contain, when running on the GPU target devices :obj:`Device.CUDA` (:obj:`Device.ROC`).
        Defaults to 64.
        Modifying might be able to increase execution time for different GPU models.
    device : :obj:`Device`
        The option to select which device will be targeted for integration.
        That is, whether the integrator is compiled for a CPU or GPU.
        Defaults to :obj:`Device.CUDA` if the system it is being run on is Nvidia Cuda compatible, and defaults to :obj:`Device.CPU` otherwise.
        See :obj:`Device` for all options and more details.
    number_of_threads : :obj:`int`
        The number of CPU threads to use when running on a CPU device.
    get_time_evolution : :obj:`callable`
        The internal function for evaluating the time evolution operator in parallel. Compiled for chosen device on object constrution.

        Parameters:

        * **sweep_parameters** (:obj:`numpy.ndarray` of :obj:`numpy.float64`) - The input to the :obj:`get_field()` function supplied by the user. Modifies the field function so the integrator can be used for many experiments, without the need for slow recompilation. For example, if the `sweep_parameters` is used to define the bias field strength in :obj:`get_field()`, then one can run many simulations, sweeping through bias values, by calling this method multiple times, each time varying `sweep_parameters`.   
        * **time_coarse**   (:obj:`numpy.ndarray` of :obj:`numpy.float64` (time_index)) - The times that `state` was evaluated at.
        * **time_end_points** (:obj:`numpy.ndarray` of :obj:`numpy.float64` (start/end)) - The time offset that the experiment is to start at, and the time that the experiment is to finish at. Measured in s.
        * **time_step_integration** (:obj:`float`) - The integration time step. Measured in s.
        * **time_step_output** (:obj:`float`) - The sample resolution of the output timeseries for the state. Must be a whole number multiple of `time_step_integration`. Measured in s.
        * **time_evolution_output** (:obj:`numpy.ndarray` of :obj:`numpy.float128` (time_index, y_index, x_index)) - The evaluated time evolution operator between each time step. See :ref:`architecture` for some information.
    spin_calculator : :obj:`callable`
        Calculates the expected spin projection (Bloch vector) over time for a given time series of a quantum state.
        This :obj:`callable` is passed to the :obj:`Results` object returned from :func:`Simulator.evaluate()`, and is executed there just in time if the `spin` property is needed.
        Compiled for chosen device on object constrution.

        Parameters:
        
        * **state** (:obj:`numpy.ndarray` of :obj:`numpy.complex128` (time_index, magnetic_quantum_number)) - The quantum state of the spin system over time, written in terms of the eigenstates of the spin projection operator in the z direction.

        Returns:
        
        * **spin** (:obj:`numpy.ndarray` of :obj:`numpy.float64` (time_index, spatial_direction)) - The expected spin projection (Bloch vector) over time.
    """
    def __init__(self, get_field:callable, spin_quantum_number:SpinQuantumNumber, device:Device = None, exponentiation_method:ExponentiationMethod = None, use_rotating_frame:bool = True, integration_method:IntegrationMethod = IntegrationMethod.MAGNUS_CF4, number_of_squares:int = 24, threads_per_block:int = 64, max_registers:int = None, number_of_threads:int = None):
        """
        .. _Achieved Occupancy: https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm

        Parameters
        ----------
        get_field : :obj:`callable`
            A python function that describes the field that the spin system is being put under. It must have three arguments:
            
            * **time_sample** (:obj:`float`) - the time to sample the field at, in units of s.
            * **simulation_index** (:obj:`int`) - a parameter that can be swept over when multiple simulations need to be run. For example, it is used to sweep over dressing frequencies during the simulations that `spinsim` was designed for.
            * **field_sample** (:class:`numpy.ndarray` of :class:`numpy.float64` (spatial_index)) the returned value of the field. This is a four dimensional vector, with the first three entries being x, y, z spatial directions (to model a magnetic field, for example), and the fourth entry being the amplitude of the quadratic shift (only appearing, and required, in spin-one systems).

            .. note::
                This function must be compilable for the device that the integrator is being compiled for. See :class:`Device` for more information and links.

        spin_quantum_number : :obj:`SpinQuantumNumber`
            The option to select whether the simulator will integrate a spin-half :obj:`SpinQuantumNumber.HALF`, or spin-one :obj:`SpinQuantumNumber.ONE` quantum system.
        device : :obj:`Device`
            The option to select which device will be targeted for integration.
            That is, whether the integrator is compiled for a CPU or GPU.
            Defaults to :obj:`Device.CUDA` if the system it is being run on is Nvidia Cuda compatible, and defaults to :obj:`Device.CPU` otherwise.
            See :obj:`Device` for all options and more details.
        exponentiation_method : :obj:`ExponentiationMethod`
            Which method to use for matrix exponentiation in the integration algorithm.
            Defaults to :obj:`ExponentiationMethod.LIE_TROTTER` when `spin_quantum_number` is set to :obj:`SpinQuantumNumber.ONE`, and defaults to :obj:`ExponentiationMethod.ANALYTIC` when `spin_quantum_number` is set to :obj:`SpinQuantumNumber.HALF`.
            See :obj:`ExponentiationMethod` for more details.
        use_rotating_frame : :obj:`bool`
            Whether or not to use the rotating frame optimisation.
            Defaults to :obj:`True`.
            If set to :obj:`True`, the integrator moves into a frame rotating in the z axis by an amount defined by the field in the z direction.
            This removes the (possibly large) z component of the field, which increases the accuracy of the output since the integrator will on average take smaller steps.

            .. note ::

                The use of a rotating frame is commonly associated with the use of a rotating wave approximation, a technique used to get approximate analytic solutions of spin system dynamics.
                This is not done when this option is set to :obj:`True` - no such approximations are made, and the output state in given out of the rotating frame.
                One can, of course, use :mod:`spinsim` to integrate states in the rotating frame, using the rating wave approximation: just define :obj:`get_field()` with field functions that use the rotating wave approximation in the rotating frame.

        integration_method : :obj:`IntegrationMethod`
            Which integration method to use in the integration.
            Defaults to :obj:`IntegrationMethod.MAGNUS_CF4`.
            See :obj:`IntegrationMethod` for more details.
        number_of_squares : :obj:`int`
            The number of squares made by the matrix exponentiator, if :obj:`ExponentiationMethod.LIE_TROTTER` is chosen.
        threads_per_block : :obj:`int`
            The size of each thread block (workgroup), in terms of the number of threads (workitems) they each contain, when running on the GPU target devices :obj:`Device.CUDA` (:obj:`Device.ROC`).
            Defaults to 64.
            Modifying might be able to increase execution time for different GPU models.
        max_registers : :obj:`int`
            The maximum number of registers allocated per thread when using :obj:`Device.CUDA` as the target device, and can be modified to increase the execution speed for a specific GPU model.
            
            Raising this value allocates more registers (fast memory) to each thread, out of a maximum number for the whole GPU, for each specific GPU model.
            This means that if more registers are allocated than are available for the GPU model, the GPU must run fewer threads concurrently than it has Cuda cores, meaning some cores are inactive, and the GPU is said to have less occupancy.
            Lowering the value increases GPU occupancy, meaning more threads run concurrently, at the expense of fewer resgiters being avaliable to each thread, meaning slower memory must be used.
            Thus, there will be an optimal value of `max_registers` for each model of GPU running :mod:`spinsim`, balancing more threads vs faster running threads, and changing this value could increase performance for your GPU.
            See `Achieved Occupancy`_ for Nvidia's official explanation.
        number_of_threads : :obj:`int`
            The number of CPU threads to use when running on a CPU device.
        """
        if not device:
            if cuda.is_available():
                device = Device.CUDA
            else:
                device = Device.CPU

        self.threads_per_block = threads_per_block
        self.spin_quantum_number = spin_quantum_number
        self.device = device
        self.number_of_threads = number_of_threads

        self.get_time_evolution = None
        try:
            self.compile_time_evolver(get_field, spin_quantum_number, device, use_rotating_frame, integration_method, exponentiation_method, number_of_squares, threads_per_block, max_registers)
        except:
            print("\033[31mspinsim error!!!\nnumba could not jit get_field() function into a device function.\033[0m\n")
            raise

    def compile_time_evolver(self, get_field:callable, spin_quantum_number:SpinQuantumNumber, device:Device, use_rotating_frame:bool = True, integration_method:IntegrationMethod = IntegrationMethod.MAGNUS_CF4, exponentiation_method:ExponentiationMethod = None, number_of_squares:int = 24, threads_per_block:int = 64, max_registers:int = None):
        """
        Compiles the integrator and spin calculation functions of the simulator.

        Parameters
        ----------
        get_field : :obj:`callable`
            A python function that describes the field that the spin system is being put under.
            It must have three arguments:
            
            * **time_sample** (:obj:`float`) - the time to sample the field at, in units of s.
            * **sweep_parameters** (:obj:`numpy.ndarray` of :obj:`numpy.float64`) - an array of parameters that can be swept over when multiple simulations need to be run. For example, it is used to sweep over dressing frequencies during the magnetometry experiments that `spinsim` was designed for.
            * **field_sample** (:class:`numpy.ndarray` of :class:`numpy.float64` (spatial_index)) the returned value of the field. This is a four dimensional vector, with the first three entries being x, y, z spatial directions (to model a magnetic field, for example), and the fourth entry being the amplitude of the quadratic shift (only appearing, and required, in spin-one systems).

            .. note::
                This function must be compilable for the device that the integrator is being compiled for.
                See :class:`Device` for more information and links.

        spin_quantum_number : :obj:`SpinQuantumNumber`
            The option to select whether the simulator will integrate a spin-half :obj:`SpinQuantumNumber.HALF`, or spin-one :obj:`SpinQuantumNumber.ONE` quantum system.
        device : :obj:`Device`
            The option to select which device will be targeted for integration.
            That is, whether the integrator is compiled for a CPU or GPU.
            Defaults to :obj:`Device.CUDA` if the system it is being run on is Nvidia Cuda compatible, and defaults to :obj:`Device.CPU` otherwise.
            See :obj:`Device` for all options and more details.
        exponentiation_method : :obj:`ExponentiationMethod`
            Which method to use for matrix exponentiation in the integration algorithm.
            Defaults to :obj:`ExponentiationMethod.LIE_TROTTER` when `spin_quantum_number` is set to :obj:`SpinQuantumNumber.ONE`, and defaults to :obj:`ExponentiationMethod.ANALYTIC` when `spin_quantum_number` is set to :obj:`SpinQuantumNumber.HALF`.
            See :obj:`ExponentiationMethod` for more details.
        use_rotating_frame : :obj:`bool`
            Whether or not to use the rotating frame optimisation.
            Defaults to :obj:`True`.
            If set to :obj:`True`, the integrator moves into a frame rotating in the z axis by an amount defined by the field in the z direction.
            This removes the (possibly large) z component of the field, which increases the accuracy of the output since the integrator will on average take smaller steps.

            .. note ::

                The use of a rotating frame is commonly associated with the use of a rotating wave approximation, a technique used to get approximate analytic solutions of spin system dynamics.
                This is not done when this option is set to :obj:`True` - no such approximations are made, and the output state in given out of the rotating frame.
                One can, of course, use :mod:`spinsim` to integrate states in the rotating frame, using the rating wave approximation: just define :obj:`get_field()` with field functions that use the rotating wave approximation in the rotating frame.

        integration_method : :obj:`IntegrationMethod`
            Which integration method to use in the integration.
            Defaults to :obj:`IntegrationMethod.MAGNUS_CF4`.
            See :obj:`IntegrationMethod` for more details.
        number_of_squares : :obj:`int`
            The number of squares made by the matrix exponentiator, if :obj:`ExponentiationMethod.LIE_TROTTER` is chosen.
        threads_per_block : :obj:`int`
            The size of each thread block (workgroup), in terms of the number of threads (workitems) they each contain, when running on the GPU target devices :obj:`Device.CUDA` (:obj:`Device.ROC`).
            Defaults to 64.
            Modifying might be able to increase execution time for different GPU models.
        max_registers : :obj:`int`
            The maximum number of registers allocated per thread when using :obj:`Device.CUDA` as the target device, and can be modified to increase the execution speed for a specific GPU model.
            Defaults to 63 (optimal for GTX1070, the device used for testing.
            Note that one extra register per thread is always added to the number specified for control, so really this number is 64).
            
            Raising this value allocates more registers (fast memory) to each thread, out of a maximum number for the whole GPU, for each specific GPU model.
            This means that if more registers are allocated than are available for the GPU model, the GPU must run fewer threads concurrently than it has Cuda cores, meaning some cores are inactive, and the GPU is said to have less occupancy.
            Lowering the value increases GPU occupancy, meaning more threads run concurrently, at the expense of fewer registers being avaliable to each thread, meaning slower memory must be used.
            Thus, there will be an optimal value of `max_registers` for each model of GPU running :mod:`spinsim`, balancing more threads vs faster running threads, and changing this value could increase performance for your GPU.
            See `Achieved Occupancy`_ for Nvidia's official explanation.
        """
        utilities = Utilities(spin_quantum_number, device, threads_per_block, number_of_squares)
        conj = utilities.conj
        set_to = utilities.set_to
        set_to_one = utilities.set_to_one
        matrix_multiply = utilities.matrix_multiply
        matrix_exponential_analytic = utilities.matrix_exponential_analytic
        matrix_exponential_lie_trotter = utilities.matrix_exponential_lie_trotter
        matrix_exponential_lie_trotter_8 = utilities.matrix_exponential_lie_trotter_8

        jit_host = device.jit_host
        jit_device = device.jit_device
        jit_device_template = device.jit_device_template
        device_index = device.index

        if not exponentiation_method:
            if spin_quantum_number == SpinQuantumNumber.ONE:
                exponentiation_method = ExponentiationMethod.LIE_TROTTER
            elif spin_quantum_number == SpinQuantumNumber.HALF:
                exponentiation_method = ExponentiationMethod.ANALYTIC

        if integration_method == IntegrationMethod.MAGNUS_CF4:
            sample_index_max = 3
            sample_index_end = 4
        elif integration_method == IntegrationMethod.HEUN:
            sample_index_max = 3
            sample_index_end = 4
        elif integration_method == IntegrationMethod.EULER:
            sample_index_max = 1
            sample_index_end = 1

        exponentiation_method_index = exponentiation_method.index

        dimension = spin_quantum_number.dimension
        if spin_quantum_number == SpinQuantumNumber.HALF:
            lie_dimension = 3
        elif spin_quantum_number == SpinQuantumNumber.ONE:
            if exponentiation_method == ExponentiationMethod.LIE_TROTTER:
                lie_dimension = 4
            elif exponentiation_method == ExponentiationMethod.LIE_TROTTER_8:
                lie_dimension = 8

        if (exponentiation_method == ExponentiationMethod.ANALYTIC) and (spin_quantum_number != SpinQuantumNumber.HALF):
            print("\033[31mspinsim warning!!!\n_attempting to use an analytic exponentiation method outside of spin-half. Switching to a Lie Trotter method.\033[0m")
            exponentiation_method = ExponentiationMethod.LIE_TROTTER
            exponentiation_method_index = 1
        @jit_device_template("(float64[:], complex128[:, :], complex128[:, :])")
        def append_exponentiation(field_sample, time_evolution_fine, time_evolution_output):
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
                matrix_exponential_lie_trotter(field_sample, time_evolution_fine)
            elif exponentiation_method_index == 2:
                matrix_exponential_lie_trotter_8(field_sample, time_evolution_fine)

            # Premultiply to the exitsing time evolution operator
            set_to(time_evolution_output, time_evolution_old)
            matrix_multiply(time_evolution_fine, time_evolution_old, time_evolution_output)

        if use_rotating_frame:
            if dimension == 3:
                if exponentiation_method_index == 2:
                    @jit_device_template("(float64[:], float64, complex128)")
                    def transform_frame_spin_one_rotating_8(field_sample, rotating_wave, rotating_wave_winding):
                        X = (field_sample[0] + 1j*field_sample[1])*conj(rotating_wave_winding)
                        field_sample[0] = X.real
                        field_sample[1] = X.imag
                        field_sample[2] = field_sample[2] - rotating_wave
                        X = (field_sample[4] + 1j*field_sample[5])*conj(rotating_wave_winding)*conj(rotating_wave_winding)
                        field_sample[4] = X.real
                        field_sample[5] = X.imag
                        X = (field_sample[6] + 1j*field_sample[7])*conj(rotating_wave_winding)
                        field_sample[6] = X.real
                        field_sample[7] = X.imag
                    transform_frame = transform_frame_spin_one_rotating_8
                else:
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
            @jit_device_template("(float64[:], float64, float64, float64, float64[:, :], float64, complex128[:])")
            def get_field_integration_magnus_cf4(sweep_parameters, time_fine, time_coarse, time_step_integration, field_sample, rotating_wave, rotating_wave_winding):
                time_sample = ((time_fine + 0.5*time_step_integration*(1 - 1/sqrt3)) - time_coarse)
                rotating_wave_winding[0] = cmath.exp(1j*rotating_wave*time_sample)
                time_sample += time_coarse
                get_field_jit(time_sample, sweep_parameters, field_sample[0, :])

                time_sample = ((time_fine + 0.5*time_step_integration*(1 + 1/sqrt3)) - time_coarse)
                rotating_wave_winding[1] = cmath.exp(1j*rotating_wave*time_sample)
                time_sample += time_coarse
                get_field_jit(time_sample, sweep_parameters, field_sample[1, :])

            @jit_device_template("(complex128[:, :], complex128[:, :], float64[:, :], float64, float64, complex128[:])")
            def append_exponentiation_integration_magnus_cf4(time_evolution_fine, time_evolution_output, field_sample, time_step_integration, rotating_wave, rotating_wave_winding):
                transform_frame(field_sample[0, :], rotating_wave, rotating_wave_winding[0])
                transform_frame(field_sample[1, :], rotating_wave, rotating_wave_winding[1])

                w0 = (1.5 + sqrt3)/6
                w1 = (1.5 - sqrt3)/6
                
                field_sample[2, 0] = time_step_integration*(w0*field_sample[0, 0] + w1*field_sample[1, 0])
                field_sample[2, 1] = time_step_integration*(w0*field_sample[0, 1] + w1*field_sample[1, 1])
                field_sample[2, 2] = time_step_integration*(w0*field_sample[0, 2] + w1*field_sample[1, 2])
                if dimension > 2:
                    field_sample[2, 3] = time_step_integration*(w0*field_sample[0, 3] + w1*field_sample[1, 3])

                append_exponentiation(field_sample[2, :], time_evolution_fine, time_evolution_output)

                field_sample[2, 0] = time_step_integration*(w1*field_sample[0, 0] + w0*field_sample[1, 0])
                field_sample[2, 1] = time_step_integration*(w1*field_sample[0, 1] + w0*field_sample[1, 1])
                field_sample[2, 2] = time_step_integration*(w1*field_sample[0, 2] + w0*field_sample[1, 2])
                if dimension > 2:
                    field_sample[2, 3] = time_step_integration*(w1*field_sample[0, 3] + w0*field_sample[1, 3])

                append_exponentiation(field_sample[2, :], time_evolution_fine, time_evolution_output)

            get_field_integration = get_field_integration_magnus_cf4
            append_exponentiation_integration = append_exponentiation_integration_magnus_cf4

        elif integration_method == IntegrationMethod.HEUN:
            @jit_device_template("(float64[:], float64, float64, float64, float64[:, :], float64, complex128[:])")
            def get_field_integration_heun(sweep_parameters, time_fine, time_coarse, time_step_integration, field_sample, rotating_wave, rotating_wave_winding):
                time_sample = time_fine - time_coarse
                rotating_wave_winding[0] = cmath.exp(1j*rotating_wave*time_sample)
                time_sample += time_coarse
                get_field_jit(time_sample, sweep_parameters, field_sample[0, :])

                time_sample = time_fine + time_step_integration - time_coarse
                rotating_wave_winding[1] = cmath.exp(1j*rotating_wave*time_sample)
                time_sample += time_coarse
                get_field_jit(time_sample, sweep_parameters, field_sample[1, :])

            @jit_device_template("(complex128[:, :], complex128[:, :], float64[:, :], float64, float64, complex128[:])")
            def append_exponentiation_integration_heun(time_evolution_fine, time_evolution_output, field_sample, time_step_integration, rotating_wave, rotating_wave_winding):
                transform_frame(field_sample[0, :], rotating_wave, rotating_wave_winding[0])
                transform_frame(field_sample[1, :], rotating_wave, rotating_wave_winding[1])
                
                field_sample[2, 0] = time_step_integration*field_sample[0, 0]/2
                field_sample[2, 1] = time_step_integration*field_sample[0, 1]/2
                field_sample[2, 2] = time_step_integration*field_sample[0, 2]/2
                if dimension > 2:
                    field_sample[2, 3] = time_step_integration*field_sample[0, 3]/2

                append_exponentiation(field_sample[2, :], time_evolution_fine, time_evolution_output)

                field_sample[2, 0] = time_step_integration*field_sample[1, 0]/2
                field_sample[2, 1] = time_step_integration*field_sample[1, 1]/2
                field_sample[2, 2] = time_step_integration*field_sample[1, 2]/2
                if dimension > 2:
                    field_sample[2, 3] = time_step_integration*field_sample[1, 3]/2

                append_exponentiation(field_sample[2, :], time_evolution_fine, time_evolution_output)

            get_field_integration = get_field_integration_heun
            append_exponentiation_integration = append_exponentiation_integration_heun

        elif integration_method == IntegrationMethod.EULER:
            @jit_device_template("(float64[:], float64, float64, float64, float64[:, :], float64, complex128[:])")
            def get_field_integration_euler(sweep_parameters, time_fine, time_coarse, time_step_integration, field_sample, rotating_wave, rotating_wave_winding):
                time_sample = time_fine + 0.5*time_step_integration - time_coarse
                rotating_wave_winding[0] = cmath.exp(1j*rotating_wave*time_sample)
                time_sample += time_coarse
                get_field_jit(time_sample, sweep_parameters, field_sample[0, :])

            @jit_device_template("(complex128[:, :], complex128[:, :], float64[:, :], float64, float64, complex128[:])")
            def append_exponentiation_integration_euler(time_evolution_fine, time_evolution_output, field_sample, time_step_integration, rotating_wave, rotating_wave_winding):
                transform_frame(field_sample[0, :], rotating_wave, rotating_wave_winding[0])
                
                field_sample[0, 0] = time_step_integration*field_sample[0, 0]
                field_sample[0, 1] = time_step_integration*field_sample[0, 1]
                field_sample[0, 2] = time_step_integration*field_sample[0, 2]
                if dimension > 2:
                    field_sample[0, 3] = time_step_integration*field_sample[0, 3]

                append_exponentiation(field_sample[0, :], time_evolution_fine, time_evolution_output)

            get_field_integration = get_field_integration_euler
            append_exponentiation_integration = append_exponentiation_integration_euler

        @jit_device_template("(int64, float64[:], float64, float64, float64[:], complex128[:, :, :], float64[:])")
        def get_time_evolution_loop(time_index, time_coarse, time_step_output, time_step_integration, time_end_points, time_evolution_output, sweep_parameters):
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
            
            time_coarse[time_index] = time_end_points[0] + time_step_output*time_index
            time_fine = time_coarse[time_index]

            # Initialise time evolution operator to 1
            set_to_one(time_evolution_output[time_index, :])
            field_sample[0, 2] = 0
            if use_rotating_frame:
                time_sample = time_coarse[time_index] + time_step_output/2
                get_field_jit(time_sample, sweep_parameters, field_sample[0, :])
            rotating_wave = field_sample[0, 2]
            if dimension == 2:
                rotating_wave /= 2

            # For every fine step
            for time_fine_index in range(math.floor(time_step_output/time_step_integration + 0.5)):
                get_field_integration(sweep_parameters, time_fine, time_coarse[time_index], time_step_integration, field_sample, rotating_wave, rotating_wave_winding)
                append_exponentiation_integration(time_evolution_fine, time_evolution_output[time_index, :], field_sample, time_step_integration, rotating_wave, rotating_wave_winding)

                time_fine += time_step_integration

            # Take out of rotating frame
            if use_rotating_frame:
                rotating_wave_winding[0] = cmath.exp(1j*rotating_wave*time_step_output)

                time_evolution_output[time_index, 0, 0] /= rotating_wave_winding[0]
                time_evolution_output[time_index, 0, 1] /= rotating_wave_winding[0]
                if dimension > 2:
                    time_evolution_output[time_index, 0, 2] /= rotating_wave_winding[0]

                    time_evolution_output[time_index, 2, 0] *= rotating_wave_winding[0]
                    time_evolution_output[time_index, 2, 1] *= rotating_wave_winding[0]
                    time_evolution_output[time_index, 2, 2] *= rotating_wave_winding[0]
                else:
                    time_evolution_output[time_index, 1, 0] *= rotating_wave_winding[0]
                    time_evolution_output[time_index, 1, 1] *= rotating_wave_winding[0]

        @jit_host("(float64[:], float64[:], float64[:], float64, float64, complex128[:, :, :])", max_registers)
        def get_time_evolution(sweep_parameters, time_coarse, time_end_points, time_step_integration, time_step_output, time_evolution_output):
            """
            Find the stepwise time evolution opperator.

            Parameters
            ----------
            sweep_parameters : :obj:`numpy.ndarray` of :obj:`numpy.float64`

            time_coarse : :class:`numpy.ndarray` of :class:`numpy.float64` (time_index)
                A coarse grained list of time samples that the time evolution operator is found for.
                In units of s.
                This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numpy.ndarray` using :func:`numba.cuda.device_array_like()`.
            time_end_points : :class:`numpy.ndarray` of :class:`numpy.float64` (start time (0) or end time (1))
                The time values for when the experiment is to start and finishes.
                In units of s.
            time_step_integration : :obj:`float`
                The time step used within the integration algorithm.
                In units of s.
            time_step_output : :obj:`float`
                The time difference between each element of `time_coarse`.
                In units of s.
                Determines the sample rate of the outputs `time_coarse` and `time_evolution_output`.
            time_evolution_output : :class:`numpy.ndarray` of :class:`numpy.complex128` (time_index, bra_state_index, ket_state_index)
                Time evolution operator (matrix) between the current and next timesteps, for each time sampled.
                See :math:`U(t)` in :ref:`overview_of_simulation_method`.
                This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numpy.ndarray` using :func:`numba.cuda.device_array_like()`.
            """

            if device_index == 0:
                for time_index in nb.prange(time_coarse.size):
                    get_time_evolution_loop(time_index, time_coarse, time_step_output, time_step_integration, time_end_points, time_evolution_output, sweep_parameters)
            elif device_index == 1:
                # Run calculation for each coarse timestep in parallel
                time_index = cuda.grid(1)
                if time_index < time_coarse.size:
                    get_time_evolution_loop(time_index, time_coarse, time_step_output, time_step_integration, time_end_points, time_evolution_output, sweep_parameters)
            elif device_index == 2:
                # Run calculation for each coarse timestep in parallel
                time_index = roc.get_global_id(1)
                if time_index < time_coarse.size:
                    get_time_evolution_loop(time_index, time_coarse, time_step_output, time_step_integration, time_end_points, time_evolution_output, sweep_parameters)
            return

        @jit_host("(complex128[:, :], float64[:, :])", max_registers = max_registers)
        def get_spin(state, spin):
            """
            Calculate each expected spin value in parallel.

            For spin-half:

            .. math::
                \\begin{align*}
                    \\langle F\\rangle(t) = \\begin{pmatrix}
                        \\Re(\\psi_{+\\frac{1}{2}}(t)\\psi_{-\\frac{1}{2}}(t)^*)\\\\
                        -\\Im(\\psi_{+\\frac{1}{2}}(t)\\psi_{-\\frac{1}{2}}(t)^*)\\\\
                        \\frac{1}{2}(|\\psi_{+\\frac{1}{2}}(t)|^2 - |\\psi_{-\\frac{1}{2}}(t)|^2)
                    \\end{pmatrix}
                \\end{align*}

            For spin-one:

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
                The state (wavefunction) of the spin system in the lab frame, for each time sampled.
                See :math:`\\psi(t)` in :ref:`overview_of_simulation_method`.
            spin : :class:`numpy.ndarray` of :class:`numpy.float64` (time_index, spatial_index)
                The expected value for hyperfine spin of the spin system in the lab frame, for each time sampled.
                Units of :math:`\\hbar`.
                This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numpy.ndarray` using :func:`numba.cuda.device_array_like()`.
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

        def spin_calculator(state):
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
            if device.index == 0:
                spin = np.empty((state.shape[0], 3), np.float64)
                get_spin(state, spin)
            elif device == Device.CUDA:
                spin = cuda.device_array((state.shape[0], 3), np.float64)
                blocks_per_grid = (state.shape[0] + (threads_per_block - 1)) // threads_per_block
                get_spin[blocks_per_grid, threads_per_block](cuda.to_device(state), spin)
                spin = spin.copy_to_host()
            elif device == Device.ROC:
                spin = roc.device_array((state.shape[0], 3), np.float64)
                blocks_per_grid = (state.shape[0] + (threads_per_block - 1)) // threads_per_block
                get_spin[blocks_per_grid, threads_per_block](roc.to_device(state), spin)
                spin = spin.copy_to_host()
            return spin

        self.get_time_evolution = get_time_evolution
        self.spin_calculator = spin_calculator

    def evaluate(self, time_start:np.float64, time_end:np.float64, time_step_integration:np.float64, time_step_output:np.float64, state_init:np.ndarray, sweep_parameters:np.ndarray = [0]) -> Results:
        """
        Integrates the time dependent Schroedinger equation and returns the quantum state of the spin system over time.

        Parameters
        ----------
        sweep_parameters : :obj:`numpy.ndarray` of :obj:`numpy.float64`
            The input to the :obj:`get_field()` function supplied by the user.
            Modifies the field function so the integrator can be used for many experiments, without the need for slow recompilation.
            For example, if the `sweep_parameters` is used to define the bias field strength in :obj:`get_field()`, then one can run many simulations, sweeping through bias values, by calling this method multiple times, each time varying `sweep_parameters`.
        time_start : :obj:`float`
            The time offset that the experiment is to start at.
            Measured in s.
        time_end : :obj:`float`
            The time that the experiment is to finish at.
            Measured in s.
            The duration of the experiment is `time_end - time_start`.
        time_step_integration : :obj:`float`
            The integration time step.
            Measured in s.
        time_step_output : :obj:`float`
            The sample resolution of the output timeseries for the state.
            Must be a whole number multiple of `time_step_integration`.
            Measured in s.
        state_init : :obj:`numpy.ndarray` of :obj:`numpy.complex128` (magnetic_quantum_number)
            The initial quantum state of the spin system, written in terms of the eigenstates of the spin projection operator in the z direction.

        Returns
        -------
        results : :obj:`Results`
            An object containing the results of the simulation.
        """
        if math.fabs(time_step_output/time_step_integration - round(time_step_output/time_step_integration)) > 1e-6:
            time_step_integration_old = time_step_integration
            time_step_integration = time_step_output/round(max(time_step_output/time_step_integration, 1))
            print(f"\033[33mspinsim warning!!!\ntime_step_output ({time_step_output:8.4e}) not an integer multiple of time_step_integration ({time_step_integration_old:8.4e}). Resetting time_step_integration to {time_step_integration:8.4e}.\033[0m\n")

        time_end_points = np.asarray([time_start, time_end], np.float64)
        state_init = np.asarray(state_init, np.complex128)
        sweep_parameters = np.asarray(sweep_parameters, np.float64)

        time_index_max = int((time_end_points[1] - time_end_points[0])/time_step_output)
        if self.device.index == 0:
            if self.device == Device.CPU:
                if self.number_of_threads:
                    old_threads = nb.get_num_threads()
                    nb.set_num_threads(self.number_of_threads)
            time = np.empty(time_index_max, np.float64)
            time_evolution_output = np.empty((time_index_max, self.spin_quantum_number.dimension, self.spin_quantum_number.dimension), np.complex128)

            self.get_time_evolution(sweep_parameters, time, time_end_points, time_step_integration, time_step_output, time_evolution_output)

            if self.device == Device.CPU:
                if self.number_of_threads:
                    nb.set_num_threads(old_threads)

        elif self.device == Device.CUDA:
            try:
                time = cuda.device_array(time_index_max, np.float64)
                time_evolution_output = cuda.device_array((time_index_max, self.spin_quantum_number.dimension, self.spin_quantum_number.dimension), np.complex128)

                sweep_parameters_device = cuda.to_device(sweep_parameters)

                blocks_per_grid = (time.size + (self.threads_per_block - 1)) // self.threads_per_block
                self.get_time_evolution[blocks_per_grid, self.threads_per_block](sweep_parameters_device, time, time_end_points, time_step_integration, time_step_output, time_evolution_output)
            except:
                print("\033[31mspinsim error!!!\nnumba.cuda could not jit get_field() function into a cuda device function.\033[0m\n")
                raise

            time_evolution_output = time_evolution_output.copy_to_host()
            time = time.copy_to_host()
        
        elif self.device == Device.ROC:
            try:
                time = roc.device_array(time_index_max, np.float64)
                time_evolution_output = roc.device_array((time_index_max, self.spin_quantum_number.dimension, self.spin_quantum_number.dimension), np.complex128)

                sweep_parameters_device = roc.to_device(sweep_parameters)

                blocks_per_grid = (time.size + (self.threads_per_block - 1)) // self.threads_per_block
                self.get_time_evolution[blocks_per_grid, self.threads_per_block](sweep_parameters_device, time, time_end_points, time_step_integration, time_step_output, time_evolution_output)
            except:
                print("\033[31mspinsim error!!!\nnumba.roc could not jit get_field() function into a roc device function.\033[0m\n")
                raise

            time_evolution_output = time_evolution_output.copy_to_host()
            time = time.copy_to_host()

        state = np.empty((time_index_max, self.spin_quantum_number.dimension), np.complex128)
        self.get_state(state_init, state, time_evolution_output)

        results = Results(time, time_evolution_output, state, self.spin_calculator)
        return results

    @staticmethod
    @nb.njit
    def get_state(state_init:np.ndarray, state:np.ndarray, time_evolution:np.ndarray):
        """
        Use the stepwise time evolution operators in succession to find the quantum state timeseries of the 3 level atom.

        Parameters
        ----------
        state_init : :class:`numpy.ndarray` of :class:`numpy.complex128`
            The state (spin wavefunction) of the system at the start of the simulation.
        state : :class:`numpy.ndarray` of :class:`numpy.complex128` (time_index, state_index)
            The state (wavefunction) of the spin system in the lab frame, for each time sampled.
        time_evolution : :class:`numpy.ndarray` of :class:`numpy.complex128` (time_index, bra_state_index, ket_state_index)
            The evaluated time evolution operator between each time step.
            See :ref:`architecture` for some information.
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

class Utilities:
    """
    A on object that contains definitions of all of the device functions (functions compiled for use on the target device) used in the integrator.
    These device functions are compiled for the chosen target device on construction of the object.

    Attributes
    ----------
    conj(z) : :obj:`callable`
        Conjugate of a complex number.

        .. math::
            \\begin{align*}
            (a + ib)^* &= a - ib\\\\
            a, b &\\in \\mathbb{R}
            \\end{align*}

        Parameters:

        * **z** (:class:`numpy.complex128`) - The complex number to take the conjugate of.
        
        Returns

        * **cz** (:class:`numpy.complex128`) - The conjugate of z.

    expm1i(b) : :obj:`callable`
        For real input :math:`b`, returns :math:`\\exp(ib) - 1`, while avoiding floating point cancellation errors.

        Parameters:

        * **b** (:class:`numpy.float64`) - The imaginary component to exponentiate.
        
        Returns

        * **em1i** (:class:`numpy.complex128`) - The evalauted output.

    cos_exp_m1(a, b) : :obj:`callable`
        For real input :math:`a`, :math:`b`, returns :math:`\\cos(a)\\exp(ib) - 1`, while avoiding floating point cancellation errors.

        Parameters:

        * **a** (:class:`numpy.float64`) - The real component to take the cosine of.

        * **b** (:class:`numpy.float64`) - The imaginary component to exponentiate.
        
        Returns

        * **cem1** (:class:`numpy.complex128`) - The evalauted output.

    cos_m1(a, b) : :obj:`callable`
        For real input :math:`a`, returns :math:`\\cos(a) - 1`, while avoiding floating point cancellation errors.

        Parameters:

        * **a** (:class:`numpy.float64`) - The real component to take the cosine of.
        
        Returns

        * **cm1** (:class:`numpy.complex128`) - The evalauted output.

    set_to(operator, result) : :obj:`callable`
        Copy the contents of one matrix into another.

        .. math::
            (A)_{i, j} = (B)_{i, j}

        Parameters:
        
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

        Parameters:
        
        * **operator** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The matrix to set to :math:`1`.

    set_to_zero(operator) : :obj:`callable`
        Make a matrix the zero matrix.

        .. math::
            \\begin{align*}
            (A)_{i, j} &= \\0
            \\end{align*}

        Parameters:
        
        * **operator** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The matrix to set to :math:`0`.

    matrix_multiply(left, right, result) : :obj:`callable`
        Multiply matrices left and right together, to be returned in result.

        .. math::
            \\begin{align*}
            (LR)_{i,k} = \\sum_j (L)_{i,j} (R)_{j,k}
            \\end{align*}

        Parameters:
        
        * **left** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The matrix to left multiply by.
        * **right** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The matrix to right multiply by.
        * **result** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - A matrix to be filled with the result of the product.

    matrix_square_m1(operator, result) : :obj:`callable`
        For matrix :math:`A = 1 + a` :math:`S = A^2 = 1 + s`.
        Here the input is the residuals :math:`a`, and the output is :math:`s`.
        This is a way to evaluate :math:`s` without floating point cancellation error.
        Specifically,

        .. math::
            \\begin{align*}
            s &= S - 1\\\\
            &= A^2 - 1\\\\
            &= (2\\cdot 1 + a)a
            \\end{align*}

        Parameters:
        
        * **operator** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The residual of the matrix to square.
        * **result** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - A matrix to be filled with the residual of the result of the product.

    matrix_multiply_m1(left, right, result) : :obj:`callable`
        For matrices :math:`L = 1 + l` and :math:`R = 1 + r`, evaluates :math:`O = LR = 1 + o`.
        Here the inputs are the residuals :math:`l` and :math:`r`, and the output is :math:`o`.
        This is a way to evaluate :math:`o` without floating point cancellation error.
        Specifically,

        .. math::
            \\begin{align*}
            o &= O - 1\\\\
            &= LR - 1\\\\
            &= l + r + lr
            \\end{align*}

        Parameters:
        
        * **left** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The residual of the matrix to left multiply by.
        * **right** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The residual of the  matrix to right multiply by.
        * **result** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - A matrix to be filled with the residual of the result of the product.

    matrix_exponential_analytic(field_sample, result) : :obj:`callable`
        Calculates a :math:`\\mathfrak{su}(2)` matrix exponential based on its analytic form.

        .. warning::
            
            Only available for use with spin-half systems.
            Will not work with spin-one systems.

        Assumes the exponent is an imaginary  linear combination of :math:`\\mathfrak{su}(2)`, being,

        .. math::
            \\begin{align*}
                A &= -i(\\omega_x J_x + \\omega_y J_y + \\omega_z J_z),
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
                \\exp(A) &= \\exp(-i\\omega_x J_x - i\\omega_y J_y - i\\omega_z J_z)\\\\
                &= \\begin{pmatrix}
                    \\cos(\\frac{\\omega_r}{2}) - i\\frac{\\omega_z}{\\omega_r}\\sin(\\frac{\\omega_r}{2}) & -\\frac{\\omega_y + i\\omega_x}{\\omega_r}\\sin(\\frac{\\omega_r}{2})\\\\
                    \\frac{\\omega_y - i\\omega_x}{\\omega_r}\\sin(\\frac{\\omega_r}{2}) & \\cos(\\frac{\\omega_r}{2}) + i\\frac{\\omega_z}{\\omega_r}\\sin(\\frac{\\omega_r}{2})
                \\end{pmatrix}
            \\end{align*}

        with :math:`\\omega_r = \\sqrt{\\omega_x^2 + \\omega_y^2 + \\omega_z^2}`.

        Parameters:
        
        * **field_sample** (:class:`numpy.ndarray` of :class:`numpy.float64`, (y_index, x_index)) - The values of :math:`\\omega_x`, :math:`\\omega_y` and :math:`\\omega_z` respectively, as described above.
        * **result** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The matrix which the result of the exponentiation is to be written to.

    matrix_exponential_lie_trotter(field_sample, result) : :obj:`callable`
        Calculates a matrix exponential based on the Lie Product Formula,

        .. math::
            \\exp(A + B) = \\lim_{c \\to \\infty} \\left(\\exp\\left(\\frac{1}{c}A\\right) \\exp\\left(\\frac{1}{c}B\\right)\\right)^c.

        **For spin-half systems:**

        Assumes the exponent is an imaginary linear combination of a subspace of :math:`\\mathfrak{su}(2)`, being,

        .. math::
            \\begin{align*}
                A &= -i(\\omega_x J_x + \\omega_y J_y + \\omega_z J_z),
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
                \\exp(A) =& \\exp\\left(-i\\omega_x J_x - i\\omega_y J_y - i\\omega_z J_z\\right)\\\\
            =& \\exp\\left(2^{-\\tau}\\left(-i\\omega_x J_x - i\\omega_y J_y - i\\omega_z J_z\\right)\\right)^{2^\\tau}\\\\
            \\approx& \\biggl(\\exp\\left(-i\\frac12 2^{-\\tau} \\omega_z J_z\\right)\\exp\\left(-i\\left(2^{-\\tau} \\omega_\\phi J_\\phi\\right)\\right)\\exp\\left(-i\\frac12 2^{-\\tau} \\omega_z J_z\\right)\\biggr)^{2^\\tau}\\\\
            =& \\begin{pmatrix}
                \\cos\\left(\\frac{\\Phi}{2}\\right)e^{-iz} & -i\\sin\\left(\\frac{\\Phi}{2}\\right) e^{i\\phi}\\\\
                -i\\sin\\left(\\frac{\\Phi}{2}\\right) e^{-i\\phi} & \\cos\\left(\\frac{\\Phi}{2}\\right)e^{iz}
            \\end{pmatrix}^{2^\\tau}\\\\
            =& T^{2^\\tau}.
            \\end{align*}

        Here :math:`z = 2^{-\\tau}\\frac{\\omega_z}{2}`, :math:`\\Phi = 2^{-\\tau}\\sqrt{\\omega_x^2 + \\omega_y^2}`, and :math:`\\phi = \\mathrm{atan}2(\\omega_y, \\omega_x)`.

        **For spin-one systems**

        Assumes the exponent is an imaginary linear combination of a subspace of :math:`\\mathfrak{su}(3)`, being,

        .. math::
            \\begin{align*}
                A &= -i(\\omega_x J_x + \\omega_y J_y + \\omega_z J_z + \\omega_q Q),
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
                Q &= \\frac{1}{3}\\begin{pmatrix}
                    1 &  0 & 0 \\\\
                    0 & -2 & 0 \\\\
                    0 &  0 & 1
                \\end{pmatrix}
            \\end{align*}

        Then the exponential can be approximated as, for large :math:`\\tau`,

        .. math::
            \\begin{align*}
                \\exp(A) =& \\exp\\left(-i\\omega_x J_x - i\\omega_y J_y - i\\omega_z J_z - i\\omega_q Q\\right)\\\\
            =& \\exp\\left(2^{-\\tau}\\left(-i\\omega_x J_x - i\\omega_y J_y - i\\omega_z J_z - i\\omega_q Q\\right)\\right)^{2^\\tau}\\\\
            \\approx& \\biggl(\\exp\\left(-i\\frac12\\left(2^{-\\tau} \\omega_z J_z + 2^{-\\tau}\\omega_q Q\\right)\\right)\\nonumber\\\\
            &\\cdot\\exp\\left(-i\\left(2^{-\\tau} \\omega_\\phi J_\\phi\\right)\\right)\\nonumber\\\\
            &\\cdot\\exp\\left(-i\\frac12\\left(2^{-\\tau} \\omega_z J_z + 2^{-\\tau} \\omega_q Q\\right)\\right)\\biggr)^{2^\\tau}\\\\
            =& \\begin{pmatrix}
                \\left(\\cos\\left(\\frac{\\Phi}{2}\\right) e^{-iz}e^{-iq}\\right)^2 & \\frac{-i}{\\sqrt{2}} \\sin(\\Phi)e^{iq}e^{-iz}e^{-i\\phi} & -\\left(\\sin\\left(\\frac{\\Phi}{2}\\right)e^{iq}e^{-i\\phi}\\right)^2\\\\
                \\frac{-i}{\\sqrt{2}} \\sin(\\Phi)e^{iq}e^{-iz}e^{i\\phi} & \\cos(\\Phi)e^{i4q} & \\frac{-i}{\\sqrt{2}} \\sin(\\Phi)e^{iq}e^{iz}e^{-i\\phi}\\\\
                -\\left(\\sin\\left(\\frac{\\Phi}{2}\\right)e^{-iq}e^{i\\phi}\\right)^2 & \\frac{-i}{\\sqrt{2}} \\sin(\\Phi)e^{iq}e^{iz}e^{i\\phi} & \\left(\\cos\\left(\\frac{\\Phi}{2}\\right) e^{iz}e^{-iq}\\right)^2
            \\end{pmatrix}^{2^\\tau}.\\\\
            \\end{align*}
        
        Here :math:`z = 2^{-\\tau}\\frac{\\omega_z}{2}`, :math:`q = 2^{-\\tau}\\frac{\\omega_q}{6}`, :math:`\\Phi = 2^{-\\tau}\\sqrt{\\omega_x^2 + \\omega_y^2}`, and :math:`\\phi = \\mathrm{atan}2(\\omega_y, \\omega_x)`.
        Once :math:`T` is calculated, it is then recursively squared :math:`\\tau` times to obtain :math:`\\exp(A)`.

        Parameters:
        
        * **field_sample** (:class:`numpy.ndarray` of :class:`numpy.float64`, (y_index, x_index)) - The values of x, y and z (and q for spin-one) respectively, as described above.
        * **result** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The matrix which the result of the exponentiation is to be written to.
        * **number_of_squares** (:obj:`int`) - The number of squares to make to the approximate matrix (:math:`\\tau` above).

    matrix_exponential_lie_trotter_8(field_sample, result) : :obj:`callable`
        Calculates a matrix exponential based on the Lie Product Formula,

        .. math::
            \\exp(A + B) = \\lim_{c \\to \\infty} \\left(\\exp\\left(\\frac{1}{c}A\\right) \\exp\\left(\\frac{1}{c}B\\right)\\right)^c.

        .. warning::
            
            Only available for use with spin-one systems.
            Will not work with spin-half systems.

        Assumes the exponent is an imaginary linear combination elements of :math:`\\mathfrak{su}(3)`, being,

        .. math::
            \\begin{align*}
                A &= -i(\\omega_x J_x + \\omega_y J_y + \\omega_z J_z + \\omega_q Q + \\omega_{u1} U_1 + \\omega_{u2} U_2 + \\omega_{v1} V_1 + \\omega_{v2} V_2),
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
                Q &= \\frac{1}{3}\\begin{pmatrix}
                    1 &  0 & 0 \\\\
                    0 & -2 & 0 \\\\
                    0 &  0 & 1
                \\end{pmatrix},\\\\
                U_1 &= \\begin{pmatrix}
                    0 & 0 & 1 \\\\
                    0 & 0 & 0 \\\\
                    1 & 0 & 0
                \\end{pmatrix},&
                U_2 &= \\begin{pmatrix}
                    0 & 0 & -i \\\\
                    0 & 0 &  0 \\\\
                    i & 0 &  0
                \\end{pmatrix},\\\\
                V_1 &= \\frac{1}{\\sqrt{2}}\\begin{pmatrix}
                    0 &  1 &  0 \\\\
                    1 &  0 & -1 \\\\
                    0 & -1 &  0
                \\end{pmatrix},&
                V_2 &= \\frac{1}{\\sqrt{2}}\\begin{pmatrix}
                    0 & -i & 0 \\\\
                    i &  0 & i \\\\
                    0 & -i & 0
                \\end{pmatrix}.\\\\
            \\end{align*}

        Then the exponential can be approximated as, for large :math:`\\tau`,

        .. math::
            \\begin{align*}
            \\exp(A) =& \\exp\\biggl(-i\\omega_x J_x - i\\omega_y J_y - i\\omega_z J_z - i\\omega_q Q\\\\
            &- i\\omega_{u1} U_1 - i\\omega_{u2} U_2 - i\\omega_{v1} V_1 - i\\omega_{v2} V_2\\biggr)\\\\
            & \\exp\\biggl(2^{-\\tau}\\biggl(-i\\omega_x J_x - i\\omega_y J_y - i\\omega_z J_z - i\\omega_q Q\\\\
            &- i\\omega_{u1} U_1 - i\\omega_{u2} U_2 - i\\omega_{v1} V_1 - i\\omega_{v2} V_2\\biggr)\\biggr)^{2^\\tau}\\\\
            \\approx& \\biggl(\\exp\\left(-i2^{-\\tau} \\omega_\\phi J_\\phi\\right)\\exp\\left(-i2^{-\\tau} \\omega_{u\\phi} U_{u\\phi}\\right)\\\\
            &\\cdot\\exp\\left(-i2^{-\\tau} \\omega_{v\\phi} V_{v\\phi}\\right)\\exp\\left(-i2^{-\\tau} \\omega_z J_z -i2^{-\\tau} \\omega_q Q \\right)\\biggr)^{2^\\tau}\\\\
            =& \\biggl(\\begin{pmatrix}
                \\cos^2\\left(\\frac{\\Phi}{2}\\right) & \\frac{-i}{\\sqrt{2}} \\sin(\\Phi)e^{-i\\phi} & -\\left(\\sin\\left(\\frac{\\Phi}{2}\\right)e^{-i\\phi}\\right)^2\\\\
                \\frac{-i}{\\sqrt{2}} \\sin(\\Phi)e^{i\\phi} & \\cos\\left(\\Phi\\right) & \\frac{-i}{\\sqrt{2}} \\sin(\\Phi)e^{-i\\phi}\\\\
                -\\left(\\sin\\left(\\frac{\\Phi}{2}\\right)e^{i\\phi}\\right)^2 & \\frac{-i}{\\sqrt{2}} \\sin(\\Phi)e^{i\\phi} & \\cos^2\\left(\\frac{\\Phi}{2}\\right)
            \\end{pmatrix}\\\\
            &\\cdot \\begin{pmatrix}
                \\cos\\left(\\Phi_u\\right) & 0 & -i \\sin\\left(\\Phi_u\\right)e^{-i\\phi_u}\\\\
                0 & 1 & 0\\\\
                -i \\sin\\left(\\Phi_u\\right)e^{i\\phi_u} & 0 & \\cos\\left(\\Phi_u\\right)
            \\end{pmatrix}\\\\
            &\\cdot \\begin{pmatrix}
                \\cos^2\\left(\\frac{\\Phi_v}{2}\\right) & \\frac{-i}{\\sqrt{2}} \\sin(\\Phi_v)e^{-i\\phi_v} & \\left(\\sin\\left(\\frac{\\Phi_v}{2}\\right)e^{-i\\phi_v}\\right)^2\\\\
                \\frac{-i}{\\sqrt{2}} \\sin(\\Phi_v)e^{i\\phi_v} & \\cos\\left(\\Phi_v\\right) & \\frac{i}{\\sqrt{2}} \\sin(\\Phi_v)e^{-i\\phi_v}\\\\
                \\left(\\sin\\left(\\frac{\\Phi_v}{2}\\right)e^{i\\phi_v}\\right)^2 & \\frac{i}{\\sqrt{2}} \\sin(\\Phi_v)e^{i\\phi_v} & \\cos^2\\left(\\frac{\\Phi_v}{2}\\right)
            \\end{pmatrix}\\\\
            &\\cdot \\begin{pmatrix}
                e^{-iz - iq} & 0 & 0\\\\
                0 & e^{i2q} & 0\\\\
                0 & 0 & e^{iz - iq}
            \\end{pmatrix}\\biggr)^{2^\\tau}\\\\
            =& T^{2^\\tau}.
            \\end{align*}

        Here :math:`z = 2^{-\\tau}\\frac{\\omega_z}{2}`, :math:`q = 2^{-\\tau}\\frac{\\omega_q}{6}`, :math:`\\Phi = 2^{-\\tau}\\sqrt{\\omega_x^2 + \\omega_y^2}`, :math:`\\phi = \\mathrm{atan}2(\\omega_y, \\omega_x)`, :math:`\\Phi_u = 2^{-\\tau}\\sqrt{\\omega_{u1}^2 + \\omega_{u2}^2}`, :math:`\\phi_u = \\mathrm{atan}2(\\omega_{u1}, \\omega_{u2})`, :math:`\\Phi_v = 2^{-\\tau}\\sqrt{\\omega_{v1}^2 + \\omega_{v2}^2}`, and :math:`\\phi_v = \\mathrm{atan}2(\\omega_{v1}, \\omega_{v2})`.
        Once :math:`T` is calculated, it is then recursively squared :math:`\\tau` times to obtain :math:`\\exp(A)`.

        Parameters:

        * **field_sample** (:class:`numpy.ndarray` of :class:`numpy.float64`, (y_index, x_index)) - The values of x, y, z, q, u1, u2, v1 and v2 respectively, as described above.
        * **result** (:class:`numpy.ndarray` of :class:`numpy.complex128`, (y_index, x_index)) - The matrix which the result of the exponentiation is to be written to.
        * **number_of_squares** (:obj:`int`) - The number of squares to make to the approximate matrix (:math:`\\tau` above).
    """
    def __init__(self, spin_quantum_number:SpinQuantumNumber, device:Device, threads_per_block:int, number_of_squares:int):
        """
        Parameters
        ----------
        spin_quantum_number : :obj:`SpinQuantumNumber`
            The option to select whether the simulator will integrate a spin-half :obj:`SpinQuantumNumber.HALF`, or spin-one :obj:`SpinQuantumNumber.ONE` quantum system.
        device : :obj:`Device`
            The option to select which device will be targeted for integration.
            That is, whether the integrator is compiled for a CPU or GPU.
            Defaults to :obj:`Device.CUDA` if the system it is being run on is Nvidia Cuda compatible, and defaults to :obj:`Device.CPU` otherwise.
            See :obj:`Device` for all options and more details.
        threads_per_block : :obj:`int`
            The size of each thread block (workgroup), in terms of the number of threads (workitems) they each contain, when running on the GPU target devices :obj:`Device.CUDA` (:obj:`Device.ROC`).
            Defaults to 64.
            Modifying might be able to increase execution time for different GPU models.
        """
        jit_device = device.jit_device
        device_index = device.index

        number_of_hypercubes = math.ceil(number_of_squares/2)
        if number_of_hypercubes < 0:
            number_of_hypercubes = 0
        trotter_precision = 4**number_of_hypercubes
        # print(type(trotter_precision), trotter_precision)

        @jit_device
        def conj(z):
            return (z.real - 1j*z.imag)

        @jit_device
        def expm1i(i):
            return -2*(math.sin(i/2)**2) + 1j*math.sin(i)

        @jit_device
        def cos_exp_m1(c, e):
            return (expm1i(c + e) + expm1i(-c + e))/2

        @jit_device
        def cos_m1(t):
            return -2*(math.sin(t/2)**2)

        if spin_quantum_number == SpinQuantumNumber.HALF:
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
            def matrix_square_m1(operator, result):
                result[0, 0] = (2 + operator[0, 0])*operator[0, 0] + operator[0, 1]*operator[1, 0]
                result[1, 0] = operator[1, 0]*operator[0, 0] + (2 + operator[1, 1])*operator[1, 0]

                result[0, 1] = (2 + operator[0, 0])*operator[0, 1] + operator[0, 1]*operator[1, 1]
                result[1, 1] = operator[1, 0]*operator[0, 1] + (2 + operator[1, 1])*operator[1, 1]

            @jit_device
            def matrix_multiply_m1(left, right, result):
                result[0, 0] = (left[0, 0] + right[0, 0]) + (left[0, 0]*right[0, 0] + left[0, 1]*right[1, 0])
                result[1, 0] = (left[1, 0] + right[1, 0]) + (left[1, 0]*right[0, 0] + left[1, 1]*right[1, 0])

                result[0, 1] = (left[0, 1] + right[0, 1]) + (left[0, 0]*right[0, 1] + left[0, 1]*right[1, 1])
                result[1, 1] = (left[1, 1] + right[1, 1]) + (left[1, 0]*right[0, 1] + left[1, 1]*right[1, 1])

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

                    c = cos_exp_m1(r/2, 0)
                    s = math.sin(r/2)

                    result[0, 0] = c - 1j*z*s + 1
                    result[1, 0] = (y - 1j*x)*s
                    result[0, 1] = -(y + 1j*x)*s
                    result[1, 1] = c + 1j*z*s + 1
                else:
                    result[0, 0] = 1
                    result[1, 0] = 0
                    result[0, 1] = 0
                    result[1, 1] = 1

            @jit_device
            def matrix_exponential_lie_trotter(field_sample, result):
                a = math.sqrt(field_sample[0]*field_sample[0] + field_sample[1]*field_sample[1])
                if a > 0:
                    ep = (field_sample[0] + 1j*field_sample[1])/a
                else:
                    ep = 1
                a = a/trotter_precision

                Sa = -1j*math.sin(a/2)

                z = field_sample[2]/(2*trotter_precision)

                result[0, 0] = cos_exp_m1(a/2, -z)
                result[1, 0] = Sa*ep

                result[0, 1] = Sa/ep
                result[1, 1] = cos_exp_m1(a/2, z)

                if device_index == 0:
                    temporary = np.empty((2, 2), dtype = np.complex128)
                elif device_index == 1:
                    temporary = cuda.local.array((2, 2), dtype = np.complex128)
                elif device_index == 2:
                    temporary_group = roc.shared.array((threads_per_block, 2, 2), dtype = np.complex128)
                    temporary = temporary_group[roc.get_local_id(1), :, :]
                for power_index in range(number_of_hypercubes):
                    matrix_square_m1(result, temporary)
                    matrix_square_m1(temporary, result)
                result[0, 0] += 1
                result[1, 1] += 1

            def matrix_exponential_lie_trotter_8(field_sample, result):
                pass

        else:
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
            def matrix_square_m1(operator, result):
                result[0, 0] = (2 + operator[0, 0])*operator[0, 0] + operator[0, 1]*operator[1, 0] + operator[0, 2]*operator[2, 0]
                result[1, 0] = operator[1, 0]*operator[0, 0] + (2 + operator[1, 1])*operator[1, 0] + operator[1, 2]*operator[2, 0]
                result[2, 0] = operator[2, 0]*operator[0, 0] + operator[2, 1]*operator[1, 0] + (2 + operator[2, 2])*operator[2, 0]

                result[0, 1] = (2 + operator[0, 0])*operator[0, 1] + operator[0, 1]*operator[1, 1] + operator[0, 2]*operator[2, 1]
                result[1, 1] = operator[1, 0]*operator[0, 1] + (2 + operator[1, 1])*operator[1, 1] + operator[1, 2]*operator[2, 1]
                result[2, 1] = operator[2, 0]*operator[0, 1] + operator[2, 1]*operator[1, 1] + (2 + operator[2, 2])*operator[2, 1]

                result[0, 2] = (2 + operator[0, 0])*operator[0, 2] + operator[0, 1]*operator[1, 2] + operator[0, 2]*operator[2, 2]
                result[1, 2] = operator[1, 0]*operator[0, 2] + (2 + operator[1, 1])*operator[1, 2] + operator[1, 2]*operator[2, 2]
                result[2, 2] = operator[2, 0]*operator[0, 2] + operator[2, 1]*operator[1, 2] + (2 + operator[2, 2])*operator[2, 2]

            @jit_device
            def matrix_multiply_m1(left, right, result):
                result[0, 0] = (left[0, 0] + right[0, 0]) + (left[0, 0]*right[0, 0] + left[0, 1]*right[1, 0] + left[0, 2]*right[2, 0])
                result[1, 0] = (left[1, 0] + right[1, 0]) + (left[1, 0]*right[0, 0] + left[1, 1]*right[1, 0] + left[1, 2]*right[2, 0])
                result[2, 0] = (left[2, 0] + right[2, 0]) + (left[2, 0]*right[0, 0] + left[2, 1]*right[1, 0] + left[2, 2]*right[2, 0])

                result[0, 1] = (left[0, 1] + right[0, 1]) + (left[0, 0]*right[0, 1] + left[0, 1]*right[1, 1] + left[0, 2]*right[2, 1])
                result[1, 1] = (left[1, 1] + right[1, 1]) + (left[1, 0]*right[0, 1] + left[1, 1]*right[1, 1] + left[1, 2]*right[2, 1])
                result[2, 1] = (left[2, 1] + right[2, 1]) + (left[2, 0]*right[0, 1] + left[2, 1]*right[1, 1] + left[2, 2]*right[2, 1])

                result[0, 2] = (left[0, 2] + right[0, 2]) + (left[0, 0]*right[0, 2] + left[0, 1]*right[1, 2] + left[0, 2]*right[2, 2])
                result[1, 2] = (left[1, 2] + right[1, 2]) + (left[1, 0]*right[0, 2] + left[1, 1]*right[1, 2] + left[1, 2]*right[2, 2])
                result[2, 2] = (left[2, 2] + right[2, 2]) + (left[2, 0]*right[0, 2] + left[2, 1]*right[1, 2] + left[2, 2]*right[2, 2])

            @jit_device
            def matrix_exponential_analytic(field_sample, result, number_of_squares):
                pass

            @jit_device
            def matrix_exponential_lie_trotter(field_sample, result):
                a = math.sqrt(field_sample[0]*field_sample[0] + field_sample[1]*field_sample[1])
                if a > 0:
                    p = math.atan2(field_sample[1], field_sample[0])
                else:
                    p = 0
                a = a/trotter_precision
                Sa = math.sin(a/2)
                sa = -1j*math.sin(a)/sqrt2
                z = field_sample[2]/(2*trotter_precision)
                q = field_sample[3]/(6*trotter_precision)                

                save_cos_exp_m1 = cos_exp_m1(a/2, -z - q)
                result[0, 0] = save_cos_exp_m1*(save_cos_exp_m1 + 2)
                result[1, 0] = sa*cmath.exp(1j*(q + p - z))
                result[2, 0] = -(Sa**2)*cmath.exp(2*1j*(p - q))

                result[0, 1] = sa*cmath.exp(1j*(q - p - z))
                result[1, 1] = cos_exp_m1(a, 4*q)
                result[2, 1] = sa*cmath.exp(1j*(q + p + z))

                result[0, 2] = -(Sa**2)*cmath.exp(2*1j*(q - p))
                result[1, 2] = sa*cmath.exp(1j*(q - p + z))
                save_cos_exp_m1 = cos_exp_m1(a/2, z - q)
                result[2, 2] = save_cos_exp_m1*(save_cos_exp_m1 + 2)

                if device_index == 0:
                    temporary = np.empty((3, 3), dtype = np.complex128)
                elif device_index == 1:
                    temporary = cuda.local.array((3, 3), dtype = np.complex128)
                elif device_index == 2:
                    temporary_group = roc.shared.array((threads_per_block, 3, 3), dtype = np.complex128)
                    temporary = temporary_group[roc.get_local_id(1), :, :]
                for power_index in range(number_of_hypercubes):
                    matrix_square_m1(result, temporary)
                    matrix_square_m1(temporary, result)

                result[0, 0] += 1
                result[1, 1] += 1
                result[2, 2] += 1

            @jit_device
            def matrix_exponential_lie_trotter_8(field_sample, result):
                if device_index == 0:
                    temporary_1 = np.empty((3, 3), dtype = np.complex128)
                    temporary_2 = np.empty((3, 3), dtype = np.complex128)
                elif device_index == 1:
                    temporary_1 = cuda.local.array((3, 3), dtype = np.complex128)
                    temporary_2 = cuda.local.array((3, 3), dtype = np.complex128)
                elif device_index == 2:
                    temporary_group_1 = roc.shared.array((threads_per_block, 3, 3), dtype = np.complex128)
                    temporary_group_2 = roc.shared.array((threads_per_block, 3, 3), dtype = np.complex128)
                    temporary_1 = temporary_group_1[roc.get_local_id(1), :, :]
                    temporary_2 = temporary_group_2[roc.get_local_id(1), :, :]

                a = math.sqrt(field_sample[0]*field_sample[0] + field_sample[1]*field_sample[1])
                if a > 0:
                    ep = (field_sample[0] + 1j*field_sample[1])/a
                    a = a/trotter_precision
                    Sa = math.sin(a/2)
                    sa = -1j*math.sin(a)/sqrt2             
                    Cam1 = cos_m1(a/2)
                else:
                    ep = 1
                    Sa = 0
                    sa = 0
                    Cam1 = 0

                result[0, 0] = Cam1*(Cam1 + 2)
                result[1, 0] = sa*ep
                result[2, 0] = -(Sa*ep)**2

                result[0, 1] = sa*conj(ep)
                result[1, 1] = cos_m1(a)
                result[2, 1] = sa*ep

                result[0, 2] = -(Sa*conj(ep))**2
                result[1, 2] = sa*conj(ep)
                result[2, 2] = Cam1*(Cam1 + 2)

                a = math.sqrt(field_sample[6]*field_sample[6] + field_sample[7]*field_sample[7])
                if a > 0:
                    ep = (field_sample[6] + 1j*field_sample[7])/a
                    a = a/trotter_precision
                    Sa = math.sin(a/2)
                    sa = -1j*math.sin(a)/sqrt2             
                    Cam1 = cos_m1(a/2)
                else:
                    ep = 1
                    Sa = 0
                    sa = 0
                    Cam1 = 0

                temporary_1[0, 0] = Cam1*(Cam1 + 2)
                temporary_1[1, 0] = sa*ep
                temporary_1[2, 0] = -(Sa*ep)**2

                temporary_1[0, 1] = sa*conj(ep)
                temporary_1[1, 1] = cos_m1(a)
                temporary_1[2, 1] = -sa*ep

                temporary_1[0, 2] = -(Sa*conj(ep))**2
                temporary_1[1, 2] = -sa*conj(ep)
                temporary_1[2, 2] = Cam1*(Cam1 + 2)

                matrix_multiply_m1(result, temporary_1, temporary_2)

                a = math.sqrt(field_sample[4]*field_sample[4] + field_sample[5]*field_sample[5])
                if a > 0:
                    ep = (field_sample[4] + 1j*field_sample[5])/a
                    a = a/trotter_precision
                    Sa = math.sin(a)        
                    Cam1 = cos_m1(a)
                else:
                    ep = 1
                    ep = 1
                    Sa = 0
                    sa = 0
                    Cam1 = 0

                result[0, 0] = Cam1
                result[1, 0] = 0.0
                result[2, 0] = -(Sa*ep)**2

                result[0, 1] = sa*conj(ep)
                result[1, 1] = 0.0
                result[2, 1] = -sa*ep

                result[0, 2] = -(Sa*conj(ep))**2
                result[1, 2] = 0.0
                result[2, 2] = Cam1

                matrix_multiply_m1(result, temporary_2, temporary_1)

                a = field_sample[2]/trotter_precision
                ep = field_sample[3]/(3*trotter_precision)

                temporary_2[0, 0] = expm1i(-a - ep)
                temporary_2[1, 0] = 0.0
                temporary_2[2, 0] = 0.0

                temporary_2[0, 1] = 0.0
                temporary_2[1, 1] = expm1i(2*ep)
                temporary_2[2, 1] = 0.0

                temporary_2[0, 2] = 0.0
                temporary_2[1, 2] = 0.0
                temporary_2[2, 2] = expm1i(a - ep)

                matrix_multiply_m1(temporary_1, temporary_2, result)

                for power_index in range(number_of_hypercubes):
                    matrix_square_m1(result, temporary_1)
                    matrix_square_m1(temporary_1, result)

                result[0, 0] += 1
                result[1, 1] += 1
                result[2, 2] += 1

        self.conj = conj
        self.expm1i = expm1i
        self.cos_exp_m1 = cos_exp_m1
        self.cos_m1 = cos_m1
        self.set_to = set_to
        self.set_to_one = set_to_one
        self.set_to_zero = set_to_zero
        self.matrix_multiply = matrix_multiply
        self.matrix_multiply_m1 = matrix_multiply_m1
        self.matrix_exponential_analytic = matrix_exponential_analytic
        self.matrix_exponential_lie_trotter = matrix_exponential_lie_trotter
        self.matrix_exponential_lie_trotter_8 = matrix_exponential_lie_trotter_8
        self.matrix_square_m1 = matrix_square_m1