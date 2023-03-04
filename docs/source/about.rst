Usage and about
===============

Click `here <https://www.sciencedirect.com/science/article/pii/S0010465523000462>`_ to read our manuscript or `here <https://arxiv.org/abs/2204.05586>`_ to read our preprint.

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
    }