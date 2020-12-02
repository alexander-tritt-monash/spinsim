"""
Some device functions for doing complex scalar maths with :mod:`numba.cuda`.
"""

import math
import numpy as np
import numba as nb
from numba import cuda

#@cuda.jit(device = True, inline = True)
def conj(z):
    """
    Conjugate of a complex number.

    .. math::
        \\begin{align*}
        (a + ib)^* &= a - ib\\\\
         a, b &\\in \\mathbb{R}
        \\end{align*}

    Parameters
    ----------
    z : :class:`numpy.cdouble`
        The complex number to take the conjugate of.
    
    Returns
    -------
    cz : :class:`numpy.cdouble`
        The conjugate of z.
    """
    return (z.real - 1j*z.imag)

#@cuda.jit(device = True, inline = True)
def complex_abs(z):
    """
    The absolute value of a complex number.

    .. math::
        \\begin{align*}
        |a + ib| &= \\sqrt{a^2 + b^2}\\\\
         a, b &\\in \\mathbb{R}
        \\end{align*}
    
    Parameters
    ----------
    z : :class:`numpy.cdouble`
        The complex number to take the absolute value of.
    
    Returns
    -------
    az : :class:`numpy.double`
        The absolute value of z.
    """
    return math.sqrt(z.real**2 + z.imag**2)