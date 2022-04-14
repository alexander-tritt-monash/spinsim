..  documentation master file, created by
   sphinx-quickstart on Thu Nov 12 17:50:03 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to spinsim's documentation!
===================================

:mod:`spinsim` is a python package that simulates spin half and spin one quantum mechanical systems following a time dependent Shroedinger equation.
It makes use of the :class:`numba.cuda` llvm compiler\ :cite:`lam_numba_2015`, and other optimisations, to allow for fast and accurate evaluation on Nvidia Cuda\ :cite:`nickolls_scalable_2008` compatible systems using GPU parallelisation.

Bellow is a basic example of evaluating the state and bloch vector for simulating Larmor precession:

.. code-block:: python

   import spinsim
   import matplotlib.pyplot as plt
   import math

   # Define field for spin system
   def get_field_larmor(time_sample, sweep_parameters, field_sample):
      field_sample[0] = 0              # Zero field in x direction
      field_sample[1] = 0              # Zero field in y direction
      field_sample[2] = math.tau*1000  # Split spin z eigenstates by 1kHz

   # Initialise simulator instance
   simulator_larmor = spinsim.Simulator(get_field_larmor, spinsim.SpinQuantumNumber.HALF)
   # Evaluate a simulation
   results_larmor = simulator_larmor.evaluate(0e-3, 100e-3, 100e-9, 500e-9, spinsim.SpinQuantumNumber.HALF.plus_x)

   # Plot results
   plt.figure()
   plt.plot(results_larmor.time, results_larmor.spin)
   plt.legend(["x", "y", "z"])
   plt.xlim(0e-3, 2e-3)
   plt.xlabel("time (s)")
   plt.ylabel("spin expectation (hbar)")
   plt.title("Spin projection for Larmor precession")
   plt.show()

This results in

.. image:: ../../images/example_1_1.png

See :ref:`examples` for a tutorial on how to use the package.

See :ref:`package` for a complete reference to the package.

For more information or the design of the package, or to cite it in your own work, please see the preprint on arxiv: https://arxiv.org/abs/2204.05586

Installation and requirements
=============================

:mod:`spinsim` can be installed using

.. code-block:: sh

   pip install spinsim

And the source code can be cloned from the git repository

.. code-block:: sh

   git clone https://github.com/alexander-tritt-monash/spinsim.git

:mod:`spinsim` uses `poetry <https://python-poetry.org/>`_ as a package manager.
One can build and install from the source code using `poetry` commands.

To use the (default) cuda GPU parallelisation, one needs to have a `cuda compatible Nvidia GPU <https://developer.nvidia.com/cuda-gpus>`_.
For cuda mode to function, one also needs to install the `Nvidia cuda toolkit <https://developer.nvidia.com/cuda-toolkit>`_.
If cuda is not available on the system, the simulator will automatically parallelise over multicore CPUs instead.
See the documentation for :class:`spinsim.Simulator` for all simulation options, including how to set the target device manually.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   about
   spinsim

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
==========

.. bibliography:: ../../bib/spinsim.bib