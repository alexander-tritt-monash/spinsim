.. _architecture:

Implementation
==============

Mathematical methods
--------------------

Parallelisation
...............

In general, :mod:`spinsim` solves the Schroedinger equation,

.. math::
    \begin{align*}
        \frac{\mathrm{d}}{\mathrm{d}t}\psi(t) &= -iH(t)\psi(t),
    \end{align*}

with :math:`i^2 = -1`, the state, :math:`\psi(t)` a time dependent, 3 (2) dimensional, unit complex vector, and :math:`H(t)` a time dependent 3 by 3 (2 by 2) complex matrix for spin one (spin half) systems.

Given that :math:`\psi(t)` is a unit vector, it is possible to write :math:`\psi(t)` in terms of a unitary transformation :math:`U(t, t_0)` of the state :math:`\psi(t_0)`, for any time :math:`t_0`. If the system is solved this way, using a geometric method, we can guarantee that :math:`\psi(t)` will always be a unit vector, which is not true with solvers in general.

.. math::
    \begin{align*}
        \psi(t) &= U(t, t_0)\psi(t_0)
    \end{align*}

This means that a time series for the state of the system can be evaluated by evaluating the time evolution operator between each of the sample times.

.. math::
    \begin{align*}
        \psi_k &= U_k\psi_{k-1},\\
        \psi(t_{k}) &= U(t_{k}, t_{k-1})\psi(t_{k-1}),\\
        t_k &= t_0 + \mathrm{D}t\cdot k
    \end{align*}

with :math:`\mathrm{D}t` the time step of the time series.

Also importantly, while each of the :math:`\psi_k` must be evaluated sequentially, the value of the :math:`U_k` is independent of the value of any :math:`\psi_{k_0}`, or any other :math:`U_{k_0}`. This means that the time evolution operators :math:`U_k` can all be calculated in parallel, and it allows :mod:`spinsim` to use GPU parallelisation on the level of time sample points, so a speed up is achieved even if just a single simulation is run.

Rotating frame
..............

If the rotating frame option is selected, the :math:`U_k` are first calculated within a local rotating frame as :math:`U^r_k`, which shrinks the size of the terms used in the calculation, increasing accuracy. The midpoint value :math:`f_r = f_z(t_k + \frac12\mathrm{D}t)` is sampled to be removed from the source. This transforms the source functions as follows,

.. math::
    \begin{align*}
        f^r_x(t) + if^r_y(t) &= e^{-i 2\pi f_r t}(f_x(t) + if_y(t)),\\
        f^r_z(t) &= f_z(t) - f_r,\\
        f^r_q(t) &= f_q(t), \textrm{ for spin one.}\\
    \end{align*}

This, (assuming that a midpoint sample is representative of an average value over the time interval), decreases the magnitude of :math:`f_z(t)`, while leaving the other source components at an equivalent magnitude. The rotation is then applied to obtain the lab frame time evolution operator :math:`U_k`,

.. math::
    \begin{align*}
        U_k &= \exp(-i 2 \pi f_r J_z \mathrm{D}t) U^r_k,\\
        U_k &= \begin{pmatrix}
            e^{-i 2\pi f_r \mathrm{D}t} & 0 & 0\\
            0 & 1 & 0\\
            0 & 0 & e^{i 2\pi f_r \mathrm{D}t}
        \end{pmatrix} U^r_k, \textrm{ for spin one}\\
        U_k &= \begin{pmatrix}
            e^{-i \pi f_r \mathrm{D}t} & 0\\
            0 & e^{i \pi f_r \mathrm{D}t}
        \end{pmatrix} U^r_k, \textrm{ for spin half.}
    \end{align*}

Magnus based integration method
...............................

The integration method used in :mod:`spinsim` is the CF4 method from :cite:`auer_magnus_2018`. Each of the :math:`U_k` are split into products of time evolution operators between times separated by a smaller timestep,

.. math::
    \begin{align*}
        U(t_k, t_{k-1}) &= U(t_k, t_k - \mathrm{d}t) \cdots U(t_{k-1} + 2\mathrm{d}t, t_{k-1} + \mathrm{d}t) U(t_{k-1} + \mathrm{d}t, t_{k-1})\\
        U_k &= u^k_{L-1} \cdots u^k_0,
    \end{align*}

with :math:`\mathrm{d}t` being the integration level, (ie, fine) time step.

The CF4 method is used to calculate each individual :math:`u^k_l`. Let the fine sample time be given by :math:`t_f = l\mathrm{d}t + t_k`. Then as part of the CF4 method, the source functions are sampled at particular times based on the second order Gauss–Legendre quadrature,

.. math::
    \begin{align*}
        f(t_1) &= (f_x(t_1), f_y(t_1), f_z(t_1), f_q(t_1)), \textrm{ with}\\
        t_1 &= t_f + \frac12 \mathrm{d}t\left(1 - \frac{1}{\sqrt{3}}\right),\\
        f(t_2) &= (f_x(t_2), f_y(t_2), f_z(t_2), f_q(t_2)), \textrm{ with}\\
        t_2 &= t_f + \frac12 \mathrm{d}t\left(1 + \frac{1}{\sqrt{3}}\right).
    \end{align*}

The fine time evolution operator can then be calculated using

.. math::
    \begin{align*}
        g_1 =& 2 \pi \mathrm{d}t \left(\frac{3 + 2 \sqrt{3}}{12} f(t_1) + \frac{3 - 2 \sqrt{3}}{12} f(t_2)\right)\\
        g_2 =& 2 \pi \mathrm{d}t \left(\frac{3 - 2 \sqrt{3}}{12} f(t_1) + \frac{3 + 2 \sqrt{3}}{12} f(t_2)\right)\\
        u =& \exp(-i \left( g_{2,x} J_x + g_{2,y} J_y + g_{2,z} J_z + g_{2,q} J_q\right))\\
        &\cdot\exp(-i \left( g_{1,x} J_x + g_{1,y} J_y + g_{1,z} J_z + g_{1,q} J_q\right)).
    \end{align*}

Exponentiator
.............

For all exponentiation, the exponentiator takes exponent in the form of

.. math::
    \begin{align*}
        E(g) &= \exp(-i (g_x J_x + g_y J_y + g_z J_z + g_q J_q)), \textrm{ with}\\
        g &= (g_x, g_y, g_z, g_q)
    \end{align*}


For spin half, the default exponentiator is in an analytic form. For spin one, an exponentiator based on the Lie Trotter product formula is used. Importantly, these two methods both use analytic forms of exponentials to construct the result, meaning that all calculated time evolution operators are unitary.

This also means that the package cannot solve arbitrary spin one quantum systems, as that would require the ability to exponentiate a point in the full, 8 dimensional Lie algebra of :math:`\mathfrak{su}(3)`, rather than just the four dimensional subspace spanned by the subalgebra :math:`\mathfrak{su}(2)` spanned by :math:`\{J_x, J_y, J_z\}`, and the single quadratic operator :math:`J_q`. Including the full algebra could be possible as a feature update if there is demand for it, though just including this subspace is sufficient for our application, and many others.

Software architecture
---------------------

Integrator architecture
.......................

The integrator in the :mod:`spinsim` package calls a :func:`numba.cuda.jit()`\ ed kernel to be run on a cuda capable Nvidia GPU in parallel, with a different thread being allocated to each of the :math:`U_k`. This returns when each of the :math:`U_k` have been evaluated.

The thread starts by calculating :math:`t_k` and, if the rotating frame is being used, :math:`f_r`. The latter is done by sampling a (:func:`numba.cuda.jit()`\ ed version of a) user provided python function describing how to sample the source Hamiltonian. The code then loops over each fine time step :math:`\mathrm{d}t` to calculate the fine time evolution operators :math:`u^k_l`.

Within the loop, the integrator enters a device function (ie a GPU subroutine, which is inline for speed) to sample :math:`f(t)`, as well as calculate :math:`e^{-i 2 \pi f_r t}`, at the sample times needed for the integration method. After this, it enters a second device function, which makes a rotating wave transformation as needed in a device function, before calculating :math:`g` values, and finally taking the matrix exponentiation in a device function. :math:`u^k_l` is premultiplied to :math:`U^r_k`, and the loop continues.

When the loop has finished, if the rotating frame is being used, :math:`U^r_k` is transformed to :math:`U_k` as above, and this is returned. Once all threads have executed, the state :math:`\psi_k` is calculated in a (CPU) :func:`numba.jit()`\ ed function from the :math:`U_k` and an initial condition :math:`\psi_{\mathrm{init}}`.

Compilation of integrator
.........................

The :mod:`spinsim` integrator is constructed and compiled just in time, using :func:`numba.cuda.jit()`. The particular device functions used are not predetermined, but are instead chosen based off user input to decide a closure. This structure has multiple advantages. Firstly, the source function :math:`f` is provided by the user as a plain python function (that must be :func:`numba.cuda.jit()` compatible). This allows users to define :math:`f` in a way that compiles and executes fast, does not put many restrictions on the form of the function, and returns the accurate results of analytic functions (compared to the errors seen in interpolation). It also allows the user to set metaparameters, and choose the features they want to use, in a way that does not require experience with the :mod:`numba.cuda` library. This was especially useful for running benchmarks comparing old integration methods to new ones, like CF4. The default settings should be optimal for most users, although tuning the values of cuda metaparameters `max_registers` and `threads_per_block` could improve performance for GPUs with a differing number of registers and cuda cores to the mobile GTX1070 used in testing here. It also allows the user to select a target device other than Cuda for compilation, so the simulator can run, using the same algorithm, on a multicore CPU in parallel if the user so chooses.

This functionality is interfaced through an object of class :class:`spinsim.Simulator`. The cuda kernel is defined as per the user's instructions on construction of the instance, and it is used by calling the method :func:`spinsim.Simulator.evaluate()`, which returns a results object including the time, state, time evolution operator, and expected spin projection (that is, Bloch vector; to be calculated as a lazy parameter if needed).