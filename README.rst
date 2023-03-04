spinsim: a GPU optimised solver of spin one and half quantum systems
====================================================================

|bagdgePyPI| |bagdgePyPIDL| |bagdgePyPIV| |bagdgePyPIL| |bagdgeRTFD| |badgePreprint| |badgeManuscript|

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
.. |badgeManuscript| image:: https://img.shields.io/badge/manuscript-Computer%20Physics%20Communications-blue
    :target: https://www.sciencedirect.com/science/article/pii/S0010465523000462
    :alt: Manuscript

*spinsim* is a python package that simulates spin half and spin one quantum mechanical systems following a time dependent Shroedinger equation. It makes use of the *numba.cuda* llvm compiler, and other optimisations, to allow for fast and accurate evaluation on Nvidia Cuda compatible systems using GPU parallelisation.

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

See our `documentation <https://spinsim.readthedocs.io/en/latest/?badge=latest>`_ for more detailed examples.

Referencing spinsim
-------------------

If used in research, please cite us as:

.. code-block:: bibtex

    @article{tritt_spinsim_2023,
	title = {Spinsim: {A} {GPU} optimized python package for simulating spin-half and spin-one quantum systems},
	volume = {287},
	issn = {0010-4655},
	shorttitle = {Spinsim},
	url = {https://www.sciencedirect.com/science/article/pii/S0010465523000462},
	doi = {10.1016/j.cpc.2023.108701},
	abstract = {Spinsim simulates the quantum dynamics of unentangled spin-1/2 and spin-1 systems evolving under time-dependent control. While other solvers for the time-dependent Schrödinger equation optimize for larger state spaces but less temporally-rich control, spinsim is optimized for intricate time evolution of a minimalist system. Efficient simulation of individual or ensemble quanta driven by adiabatic sweeps, elaborate pulse sequences, complex signals and non-Gaussian noise is the primary target application. We achieve fast and robust evolution using a geometric integrator to bound errors over many steps, and split the calculation parallel-in-time on a GPU using the numba just-in-time compiler. Speed-up is three orders of magnitude over QuTip's sesolve and Mathematica's NDSolve, and four orders over SciPy's ivp\_solve for equal accuracy. Interfaced through python, spinsim should be useful for simulating robust state preparation, inversion and dynamical decoupling sequences in NMR and MRI, and in quantum control, memory and sensing applications with two- and three-level quanta.
Program summary
Program Title: Spinsim CPC Library link to program files: https://doi.org/10.17632/f6rdk4gyxr.1 Developer's repository link: https://github.com/alexander-tritt-monash/spinsim Licensing provisions: BSD 3-clause Programming language: Python (3.7 or greater) Nature of problem: Quantum sensing is a domain of quantum technology where the dynamics of quantum systems are used to infer properties of the systems' environments. The development of quantum sensing protocols is greatly sped-up by software simulations of the new techniques. Quantum sensing simulation benefits from temporally-rich control of individual quanta. However, current specialized time-dependent Schrödinger equation solvers are instead optimized only for simple pulses in large Hilbert spaces. Thus, there is a need for efficient simulation of individual or ensemble quanta driven by adiabatic sweeps, elaborate pulse sequences, complex signals and non-Gaussian noise. Solution method: Spinsim simulates the quantum dynamics of spin-1/2 and spin-1 systems evolving under time-dependent control. We first speed up the integration of the time-dependent Schrödinger equation by splitting the calculation parallel-in-time on a GPU using the numba [1] just-in-time compiler. We achieve fast and robust evolution using a geometric integrator to bound errors over many steps. A dynamic rotating frame transformation and Lie-Trotter decomposition are used to decrease effective step sizes on long and short time scales, respectively. Hence, each individual step is more accurate. Spinsim is interfaced via a python package, meaning it can be used by researchers inexperienced with geometric integrators and GPU parallelism. Additional comments including restrictions and unusual features: To use the (default) Nvidia cuda GPU parallelization, one needs to have a cuda compatible Nvidia GPU [2]. For cuda mode to function, one also needs to install the Nvidia cuda toolkit [3]. If cuda is not available on the system, the simulator will automatically parallelize over multicore CPUs instead. Developed and tested on Windows 10; tested on Windows 11. CPU functionality tested on MacOS 10.16 Big Sur (note that MacOS 10.14 Mojave and higher are not compatible with cuda hardware and software). The package (including cuda functionality) is in principle compatible with Linux, but functionality has not been tested.
References
[1]S.K. Lam, A. Pitrou, S. Seibert, in: Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC - LLVM'15, ACM Press, Austin, TX, 2015, pp. 1-6, doi:10.1145/2833157.2833162, http://dl.acm.org/citation.cfm?doid=2833157.2833162.[2]Nvidia, CUDA GPUs (Jun. 2012), https://developer.nvidia.com/cuda-gpus.[3]Nvidia, CUDA Toolkit (Jul. 2013), https://developer.nvidia.com/cuda-toolkit.},
	language = {en},
	urldate = {2023-03-03},
	journal = {Computer Physics Communications},
	author = {Tritt, Alex and Morris, Joshua and Hochstetter, Joel and Anderson, R. P. and Saunderson, James and Turner, L. D.},
	month = jun,
	year = {2023},
	keywords = {Geometric integrator, GPU, Magnus expansion, Spin dynamics, Time-dependent Schrödinger equation, Unitary evolution},
	pages = {108701}
}

Alternatively, one can use a reference manager plugin on this repository to read *CITATION.cff* to automatically add the manuscript to your reference manager.
Click `here <https://www.sciencedirect.com/science/article/abs/pii/S0010465523000462>`_ to read out manuscript, or `here <https://arxiv.org/abs/2204.05586>`_ to read our preprint.

Repository structure
--------------------

*README.rst* is the file you are currently reading.
*LICENSE.rst* contains the BSD 3-Clause Licence for use of this software.
*CITATION.cff* allows for automatic reference manager scanning of this repository.

*spinsim/__init__.py* contains the program itself.
For information on the program structure, please see our `documentation <https://spinsim.readthedocs.io/en/latest/?badge=latest>`_.
*pyproject.toml* allows one to create builds of this software using `poetry <https://python-poetry.org/>`_.

*bib/*, *docs/* and *images/* contain the files necessary for producing the documentation, as well as the image in this *README.rst* file.

*example.py*, *full_algebra_example.py*, *gaussian_example.py* and *gaussian_example.py* are plain text files for the examples given in the documentation.

