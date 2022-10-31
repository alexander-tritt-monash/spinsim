spinsim: a GPU optimised solver of spin one and half quantum systems
====================================================================

|bagdgePyPI| |bagdgePyPIDL| |bagdgePyPIV| |bagdgePyPIL| |bagdgeRTFD| |badgePreprint|

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
.. |badgePreprint| image:: https://img.shields.io/badge/preprint-arXiv-red
    :target: https://arxiv.org/abs/2204.05586
    :alt: Preprint

*spinsim* is a python package that simulates spin half and spin one quantum mechanical systems following a time dependent Shroedinger equation. It makes use of the *numba.cuda* llvm compiler, and other optimisations, to allow for fast and accurate evaluation on Nvidia Cuda compatible systems using GPU parallelisation.

If used in research, please cite us as:

.. code-block:: bibtex

    @article{tritt_spinsim_2022,
        title = {Spinsim: a {GPU} optimized python package for simulating spin-half and spin-one quantum systems},
        shorttitle = {Spinsim},
        url = {https://arxiv.org/abs/2204.05586v1},
        abstract = {The Spinsim python package simulates spin-half and spin-one quantum mechanical systems following a time dependent Shroedinger equation. It makes use of numba.cuda, which is an LLVM (Low Level Virtual Machine) compiler for Nvidia Cuda compatible systems using GPU parallelization. Along with other optimizations, this allows for speed improvements from 3 to 4 orders of magnitude while staying just as accurate, compared to industry standard packages. It is available for installation on PyPI, and the source code is available on github. The initial use-case for the Spinsim will be to simulate quantum sensing-based ultracold atom experiments for the Monash University School of Physics {\textbackslash}\& Astronomy spinor Bose-Einstein condensate (spinor BEC) lab, but we anticipate it will be useful in simulating any range of spin-half or spin-one quantum systems with time dependent Hamiltonians that cannot be solved analytically. These appear in the fields of nuclear magnetic resonance (NMR), nuclear quadrupole resonance (NQR) and magnetic resonance imaging (MRI) experiments and quantum sensing, and with the spin-one systems of nitrogen vacancy centres (NVCs), ultracold atoms, and BECs.},
        language = {en},
        urldate = {2022-04-14},
        author = {Tritt, Alex and Morris, Joshua and Hochstetter, Joel and Anderson, R. P. and Saunderson, James and Turner, L. D.},
        month = apr,
        year = {2022},
        file = {Full Text PDF:C\:\\Users\\atri27\\Zotero\\storage\\HZQB6T7G\\Tritt et al. - 2022 - Spinsim a GPU optimized python package for simula.pdf:application/pdf;Snapshot:C\:\\Users\\atri27\\Zotero\\storage\\AN4C4NGE\\2204.html:text/html},
    }

Alternatively, use a reference manager plugin on this repository to read CITATION.cff to automatically add the manuscript to your reference manager.
See |badgePreprint| to read our preprint.

Basic example
-------------

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

.. image:: https://github.com/alexander-tritt-monash/spinsim/blob/master/images/example_1_1.png

See our documentation |bagdgeRTFD| for more detailed examples.

Repository structure
--------------------

bib