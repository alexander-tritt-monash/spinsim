spinsim: a GPU optimised solver of spin one and half quantum systems
====================================================================

|bagdgePyPI| |bagdgePyPIDL| |bagdgePyPIV| |bagdgePyPIL| |bagdgeRTFD|

.. |bagdgePyPI| image:: https://img.shields.io/pypi/v/spinsim
    :alt: PyPI
.. |bagdgePyPIDL| image:: https://img.shields.io/pypi/dm/spinsim
    :alt: PyPI - Downloads
.. |bagdgePyPIV| image:: https://img.shields.io/pypi/pyversions/spinsim
    :alt: PyPI - Python Version
.. |bagdgePyPIL| image:: https://img.shields.io/pypi/l/spinsim
    :alt: PyPI - License
.. |bagdgeRTFD| image:: https://readthedocs.org/projects/spinsim/badge/?version=latest
    :target: https://spinsim.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

*spinsim* is a python package that simulates spin half and spin one quantum mechanical systems following a time dependent Shroedinger equation. It makes use of the *numba.cuda* llvm compiler, and other optimisations, to allow for fast and accurate evaluation on Nvidia Cuda compatible systems using GPU parallelisation.

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

.. image:: _docs/source/images/example_1_1.png