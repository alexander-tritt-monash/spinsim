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

   def get_field_larmor(time_sample, field_modifier, field_sample):
      field_sample[0] = 0            # Zero field in x direction
      field_sample[1] = 0            # Zero field in y direction
      field_sample[2] = 1000         # Split spin z eigenstates by 1kHz

   simulator_larmor = spinsim.Simulator(get_field_larmor, spinsim.SpinQuantumNumber.HALF)

   state_init = np.asarray([1/np.sqrt(2), 1/np.sqrt(2)], np.cdouble)

   results_larmor = simulator_larmor.evaluate(0, 0e-3, 100e-3, 100e-9, 500e-9, state_init)

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