Quality control
===============

Benchmarks
----------

Benchmarks were performed using :mod:`neuralsense.benchmark`, by comparing state evaluations of many different typical metrology experiments and finding the mean error introduced when the fine (integration) time step is increased.

.. _fig_benchmark_spin_one:

.. figure:: ../../images/benchmark_comparison_spin_one.png

    Fine time step benchmark for spin one systems. CF4 is the Magnus commutator free integrator, HS is the two sample exponential integrator used in :mod:`AtomicPy`, MP is a single sample exponential integrator, RF is use of the rotating frame, and LF is lab frame (no use of the rotating frame). HS and MP results are drawn on top of each other due to their similarities.

:numref:`fig_benchmark_spin_one` shows the performance of :mod:`spinsim` when running in spin one mode. This shows that both using the Magnus based CF4 method and moving into a rotating frame give significant increases to accuracy. The HS (half step) method in the lab frame, with a time step of 10ns was the method used by :mod:`AtomicPy`, the previous code used by the group for simulating spin systems. Compared to this, the best performing :mod:`spinsim` method is 5 orders of magnitude more accurate, while executing in a time 2 orders of magnitude faster.

.. _fig_benchmark_spin_half_lt:

.. figure:: ../../images/benchmark_comparison_spin_half_lt.png

Fine time step benchmark for spin half systems, using the Lie Trotter based exponentiator. CF4 is the Magnus commutator free integrator, HS is the two sample exponential integrator used in :mod:`AtomicPy`, MP is a single sample exponential integrator, RF is use of the rotating frame, and LF is lab frame (no use of the rotating frame). HS and MP results are drawn on top of each other due to their similarities. Compared to this method, CF4 

From :numref:`fig_benchmark_spin_half_lt`, one gets essentially the same accuracy for each method when working in spin half mode compared to spin one, with no other changes.

.. _fig_benchmark_spin_half_a:

.. figure:: ../../images/benchmark_comparison_spin_half_a.png

Fine time step benchmark for spin half systems, using the analytic based exponentiator. CF4 is the Magnus commutator free integrator, HS is the two sample exponential integrator used in :mod:`AtomicPy`, MP is a single sample exponential integrator, RF is use of the rotating frame, and LF is lab frame (no use of the rotating frame). HS and MP results are drawn on top of each other due to their similarities.

:numref:`fig_benchmark_spin_half_a` shows that the Lie Trotter based exponentiator does limit the maximum accuracy obtainable, and for spin half systems, one can increase accuracy further (and decrease execution time) by using an analytic based exponentiator.

Testing
-------

The simulator as a whole has been functionally tested against well known analytic approximations of the behaviour spin systems. This was done for every combination of integrator settings possible when compiling the integrator. The system was benchmarked in terms of accuracy vs fine time step, again, using every possible combination of integrator settings. This confirms that no integrator diverges away from the limiting solution when the time step is decreased in magnitude. The Lie Trotter matrix exponentiator was extensively tested separately from the full system, as well as benchmarked separately.

These tests and benchmarks were run as part of the :mod:`neuralsense` package, ie the package that this simulator package was designed to be used for. The simulator has also been used as part of the measurement protocol being developed there, and it has been tested as part of those algorithms as well.

The kernel execution was profiled thoroughly, and changes were made to optimise VRAM and register usage and transfer. This was done specifically for the hardware of an Nvidia GTX1070, so one may get some performance increases by changing some GPU specific metaparameters when instantiating the :class:`spinsim.Simulator` object.

A good way to confirm that :mod:`spinsim` is functioning properly after an installation is to run the tutorial code provided and compare the outputs. Otherwise, one can run the benchmarks and simulation protocols in :mod:`neuralsense`.