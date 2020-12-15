Motivation and features
=======================

Motivation
----------

Ultracold atoms have proven their effectiveness in state of the art technologies in quantum metrology\ :cite:`huang_quantum_2014` (the use of quantum mechanics to make precise measurements of small signals). Design of such measurement protocols requires many steps of verification, including simulation. This is especially important, since running real experiments can be expensive and time consuming, and thus it is more practical to debug such protocols quickly and cheaply on a computer. In general, any design of experiments using spin systems could benefit from a fast, accurate method of simulation.

In the past, the Spinor Bose Einstein Condensate Lab at Monash University used an in-house, cython based script AtomicPy\ :cite:`morris_qcmonkatomicpy_2018` (which this package is originally based off), and standard differential equation solvers to solve the Schroedinger equation for these spin one (sometimes simplified to spin half for speed) systems. However, these methods are not completely optimised for our use case, and therefore come with some issues.

First of all, while the execution time for these solvers is acceptable for running a small number of experiments, for certain experiments involving large arrays of independent atom clouds (which require thousands of simulations to be run), this time accumulates to the order of many hours.

Secondly, the Schroedinger equation has the geometric property of being norm persevering. In other words, the time evolution operator for a system between two points in time must be unitary. This is important, because all quantum states live on the boundary of a ball of length one in their Hilbert space. For many numerical methods like those in the Runge Kutta family, the approximations used not be norm preserving, and the evaluated quantum state may well diverge to infinite norm, or converge to zero if run for too many iterations.

Thirdly, our system (and similar spin systems) can be very oscillatory. In standard conditions for our application, the expected spin projection of a system can rotate in space (alternatively viewed as a point rotating around an abstract object known as a Bloch sphere) at a rate of 700kHz. Standard integration methods require very small time steps in order to accurately depict these oscillations. We found that the integration method used by AtomicPy has an accuracy of only order :math:`10^{-2}` when the time step is set to the small value of 10ns.

Techniques
----------

Time interval level parallelisation with Cuda
.............................................

:mod:`spinsim` uses :mod:`numba.cuda` to split up work on an individual simulation over many GPU cores. This is one of the features that greatly increases speed over AtomicPy. See :ref:`architecture` for full details. This is a different approach to Auer et al\ :cite:`auer_magnus_2018`, who use parallelisation of linear algebra operations to optimise the simulation of much larger dimensional quantum systems. :mod:`spinsim` focuses on the optimisation of small, spin half and spin one systems. In addition, :mod:`spinsim` has options to run in parallel on a CPU for if a Cuda compatible device is not available.

Magnus based commutator free integrator
.......................................

:mod:`spinsim` uses a fourth order Magnus based commutator free integrator\ :cite:`auer_magnus_2018`. This brings multiple advantages. Firstly, it is a geometric integrator, meaning it produces strictly unitary time evolution operators by taking advantage of the Lie structure behind spin one and spin half systems. Secondly, the high order nature of this technique allows for larger time steps while maintaining an accurate integration.

Use of a rotating frame
.......................

If there is a mainly constant term in the Hamiltonian that drives the main source of oscillations in the system, then moving into a frame that is rotating at the same rate that the system is oscillating at, removes this large term from the system. This reduces the need to take such large steps around the Bloch sphere, and thus allows for greater accuracy. Note, that sometimes the rotating frame (interaction picture) is commonly used to make rotating wave approximations for more manageable and analytic. This does not happen in :mod:`spinsim`, this frame transformation is completely lossless.

Features
--------

Speed and accuracy
..................

See :ref:`benchmark` for a more detailed look into performance.

On :mod:`spinsim`, a typical, 100ms, spin one simulation runs in less than 120ms on a mobile Nvidia GTX1070 (8GiB VRAM, 2048 cuda cores, 1.7GHz boost), with an error of less than :math:`10^{-6}`. In comparison, AtomicPy running the same simulation runs in 12s on a mobile Intel Core i7-8750H (16GiB RAM, 6 cores, 4.1GHz boost). :mod:`spinsim` running using the AtomicPy integration technique gives an error of :math:`10^{-2}`.

Note that this means that spin one simulations run in almost real time on the Nvidia GTX1070, and spin half simulations, running in just 30ms, run significantly faster than real time. Also not that these results are for a single simulation (not including compile time). Unlike with solutions of running full simulations all in parallel with each other, having thousands of simulations running concurrently is not required to take advantage of the speed of the :mod:`spinsim` package.

User defined python function as source
......................................

The user is required to write their own python function used as the Hamiltonian to drive the spin system. This means that :mod:`spinsim` can solve Schroedinger equations with many kinds of pulse sequences, including amplitude and frequency modulation, and other sweeps, with little setup.

When set to spin half mode, the :mod:`spinsim` package solves time dependent Schroedinger equations of the form

.. math::
   \frac{\mathrm{d}}{\mathrm{d}t}\psi(t) = -i 2\pi (f_x(t) J_x + f_y(t) J_y + f_z(t) J_z) \psi(t),

where :math:`i^2 = -1`, :math:`\psi(t) \in \mathbb{C}^2`, and the spin half spin projection operators are given by

.. math::
   \begin{align*}
      J_x &= \frac12\begin{pmatrix}
         0 & 1 \\
         1 & 0
      \end{pmatrix},
      &J_y &= \frac12\begin{pmatrix}
         0 & -i \\
         i &  0
      \end{pmatrix},
      &J_z &= \frac12\begin{pmatrix}
         1 &  0 \\
         0 & -1
      \end{pmatrix}.
   \end{align*}

And, when in spin one mode, :mod:`spinsim` can solve Schroedinger equations of the form

.. math::
   \frac{\mathrm{d}}{\mathrm{d}t}\psi(t) = -i 2\pi (f_x(t) J_x + f_y(t) J_y + f_z(t) J_z + f_q(t) Q) \psi(t).

where now :math:`\psi(t) \in \mathbb{C}^3`, and the spin one operators are given by

.. math::
   \begin{align*}
      J_x &= \frac{1}{\sqrt{2}}\begin{pmatrix}
         0 & 1 & 0 \\
         1 & 0 & 1 \\
         0 & 1 & 0
      \end{pmatrix},&
      J_y &= \frac{1}{\sqrt{2}}\begin{pmatrix}
         0 & -i &  0 \\
         i &  0 & -i \\
         0 &  i &  0
      \end{pmatrix},\\
      J_z &= \begin{pmatrix}
         1 & 0 &  0 \\
         0 & 0 &  0 \\
         0 & 0 & -1
      \end{pmatrix},&
      Q &= \frac{1}{3}\begin{pmatrix}
         1 &  0 & 0 \\
         0 & -2 & 0 \\
         0 &  0 & 1
      \end{pmatrix}.
   \end{align*}

:math:`J_x, J_y, J_z` are regular spin operators, and :math:`Q` is a quadratic operator, proportional to :math:`Q_{zz}` as defined by :cite:`hamley_spin-nematic_2012`, and :math:`Q_0` as defined by :cite:`di_dipolequadrupole_2010`. The quadratic operator is required, for example, when solving Schroedinger equations in atomic physics involving single photon radio transitions, where the quadratic coupling :math:`f_q(t)` is proportional to the square of the magnetic field strength :math:`|B(t)|^2`.

The user provides a :func:`numba.cuda.jit()`\ able function (or, in general, a function compilable for the user selected target device) that samples the Hamiltonian at a certain input time `time_sample`, which writes to the array `field_sample`, which has three (four) entries for spin half (one) representing the numerical values of :math:`f_x(t),f_y(t),f_z(t)` (:math:`f_q(t)`). There is also a second input, `field_modifier` which allows for multiple versions of a simulation to be swept over using a single compiled function.  This prevents time being wasted by compiling new integrators for each new field description. See :ref:`examples` for a tutorial of using this in practice, and :class:`spinsim.Simulator` for a full reference.