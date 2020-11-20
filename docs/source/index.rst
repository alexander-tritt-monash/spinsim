..  documentation master file, created by
   sphinx-quickstart on Thu Nov 12 17:50:03 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to spinsim's documentation!
===================================

:mod:`spinsim` is a python package that simulates spin half and spin one quantum mechanical systems following a time dependent Shroedinger equation. It makes use of the :class:`numba.cuda` llvm compiler\ :cite:`lam_numba_2015`, and other optimisations, to allow for fast and accurate evaluation on Nvidia Cuda\ :cite:`nickolls_scalable_2008` compatible systems using GPU parallelisation.

Bellow is a basic example of evaluating the state and bloch vector for simulating Larmor precession:

.. code-block:: python

   import spinsim
   import numpy as np
   import matplotlib.pyplot as plt

   def get_source_larmor(time_sample, source_modifier, source_sample):
      source_sample[0] = 0            # Zero source in x direction
      source_sample[1] = 0            # Zero source in y direction
      source_sample[2] = 1000         # Split spin z eigenstates by 1kHz

   simulator_larmor = spinsim.Simulator(get_source_larmor, spinsim.SpinQuantumNumber.HALF)

   time_step_coarse = 500e-9
   time_step_fine = 100e-9
   time_end_points = np.asarray([0e-3, 100e-3], np.double)

   time_index_max = int((time_end_points[1] - time_end_points[0])/time_step_coarse)
   time = np.empty(time_index_max, np.double)

   state_init = np.asarray([1/np.sqrt(2), 1/np.sqrt(2)], np.cdouble)
   state = np.empty((time_index_max, 2), np.cdouble)
   spin = np.empty((time_index_max, 3), np.double)
   time_evolution = np.empty((time_index_max, 2, 2), np.cdouble)

   simulator_larmor.get_time_evolution(0, time, time_end_points, time_step_fine, time_step_coarse, time_evolution)
   simulator_larmor.get_state(state_init, state, time_evolution)
   simulator_larmor.get_spin(state, spin)

   plt.figure()
   plt.plot(time, spin)
   plt.legend(["x", "y", "z"])
   plt.xlim(0e-3, 2e-3)
   plt.xlabel("time (s)")
   plt.ylabel("spin expectation (hbar)")
   plt.title("Spin projection for Larmor precession")
   plt.show()

See :ref:`examples` for a tutorial on how to use the package.

See :ref:`package` for a complete reference to the package.

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