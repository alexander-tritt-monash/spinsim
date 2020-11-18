"""
Some device functions for doing spin one maths with :mod:`numba.cuda`.
"""

import math
import numpy as np
import numba as nb
from numba import cuda
from . import scalar

# Important constants
cuda_debug = False
sqrt2 = math.sqrt(2)
sqrt3 = math.sqrt(3)
machine_epsilon = np.finfo(np.float64).eps*1000  # When to decide that vectors are parallel
# trotter_cutoff = 52

@cuda.jit(device = True, inline = True)
def matrix_exponential_analytic(source_sample, result):
    """
    Calculates a :math:`su(2)` matrix exponential based on its analytic form.

    Assumes the exponent is an imaginary  linear combination of :math:`su(2)`, being,

    .. math::
        \\begin{align*}
            A &= -i(x F_x + y F_y + z F_z),
        \\end{align*}
    
    with

    .. math::
        \\begin{align*}
            F_x &= \\frac{1}{2}\\begin{pmatrix}
                0 & 1 \\\\
                1 & 0
            \\end{pmatrix},&
            F_y &= \\frac{1}{2}\\begin{pmatrix}
                0 & -i \\\\
                i &  0
            \\end{pmatrix},&
            F_z &= \\frac{1}{2}\\begin{pmatrix}
                1 &  0  \\\\
                0 & -1 
            \\end{pmatrix}
        \\end{align*}

    Then the exponential can be calculated as

    .. math::
        \\begin{align*}
            \\exp(A) &= \\exp(-ix F_x - iy F_y - iz F_z)\\\\
            &= \\begin{pmatrix}
                \\cos(\\frac{r}{2}) - i\\frac{z}{r}\\sin(\\frac{r}{2}) & -\\frac{y + ix}{r}\\sin(\\frac{r}{2})\\\\
                 \\frac{y - ix}{r}\\sin(\\frac{r}{2}) & \\cos(\\frac{r}{2}) + i\\frac{z}{r}\\sin(\\frac{r}{2})
            \\end{pmatrix}
        \\end{align*}

    with :math:`r = \\sqrt{x^2 + y^2 + z^2}`.

    Parameters
    ----------
    exponent : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix to take the exponential of.
        
    result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix which the result of the exponentiation is to be written to.
    """
    x = source_sample[0]
    y = source_sample[1]
    z = source_sample[2]

    r = math.sqrt(x**2 + y**2 + z**2)

    if r > 0:
        x /= r
        y /= r
        z /= r

        c = math.cos(r/2)
        s = math.sin(r/2)

        result[0, 0] = c - 1j*z*s
        result[1, 0] = (y - 1j*x)*s
        result[0, 1] = -(y + 1j*x)*s
        result[1, 1] = c + 1j*z*s
    else:
        result[0, 0] = 1
        result[1, 0] = 0
        result[0, 1] = 0
        result[1, 1] = 1

@cuda.jit(device = True, inline = True)
def matrix_exponential_lie_trotter(source_sample, result, trotter_cutoff):
    """
    Calculates a matrix exponential based on the Lie Product Formula,

    .. math::
        \\exp(A + B) = \\lim_{c \\to \\infty} \\left(\\exp\\left(\\frac{1}{c}A\\right) \\exp\\left(\\frac{1}{c}B\\right)\\right)^c.

    Assumes the exponent is an imaginary  linear combination of a subspace of :math:`su(3)`, being,

    .. math::
        \\begin{align*}
            A &= -i(x F_x + y F_y + z F_z + q F_q),
        \\end{align*}
    
    with

    .. math::
        \\begin{align*}
            F_x &= \\frac{1}{\\sqrt{2}}\\begin{pmatrix}
                0 & 1 & 0 \\\\
                1 & 0 & 1 \\\\
                0 & 1 & 0
            \\end{pmatrix},&
            F_y &= \\frac{1}{\\sqrt{2}}\\begin{pmatrix}
                0 & -i &  0 \\\\
                i &  0 & -i \\\\
                0 &  i &  0
            \\end{pmatrix},\\\\
            F_z &= \\begin{pmatrix}
                1 & 0 &  0 \\\\
                0 & 0 &  0 \\\\
                0 & 0 & -1
            \\end{pmatrix},&
            F_q &= \\frac{1}{3}\\begin{pmatrix}
                1 &  0 & 0 \\\\
                0 & -2 & 0 \\\\
                0 &  0 & 1
            \\end{pmatrix}
        \\end{align*}

    Then the exponential can be approximated as, for large :math:`\\tau`,

    .. math::
        \\begin{align*}
            \\exp(A) &= \\exp(-ix F_x - iy F_y - iz F_z - iq F_q)\\\\
            &= \\exp(2^{-\\tau}(-ix F_x - iy F_y - iz F_z - iq F_q))^{2^\\tau}\\\\
            &\\approx (\\exp(-i(2^{-\\tau} x) F_x) \\exp(-i(2^{-\\tau} y) F_y) \\exp(-i(2^{-\\tau} z F_z + (2^{-\\tau} q) F_q)))^{2^\\tau}\\\\
            &= \\begin{pmatrix}
                \\frac{e^{-i\\left(Z + \\frac{Q}{3}\\right)}(c_X + c_Y - i s_Xs_Y)}{2} & \\frac{e^{i\\frac{2Q}{3}} (-s_Y -i c_Y s_X)}{\\sqrt{2}} & \\frac{e^{-i\\left(-Z + \\frac{Q}{3}\\right)}(c_X - c_Y + i s_Xs_Y)}{2} \\\\
                \\frac{e^{-i\\left(Z + \\frac{Q}{3}\\right)} (-i s_X + c_X s_Y)}{\\sqrt{2}} & e^{i\\frac{2Q}{3}} c_X c_Y & \\frac{e^{-i(Z - \\frac{Q}{3})} (-i s_X - c_X s_Y)}{\\sqrt{2}} \\\\
                \\frac{e^{-i\\left(Z + \\frac{Q}{3}\\right)}(c_X - c_Y - i s_Xs_Y)}{2} & \\frac{e^{i\\frac{2Q}{3}} (s_Y -i c_Y s_X)}{\\sqrt{2}} & \\frac{e^{-i\\left(-Z + \\frac{Q}{3}\\right)}(c_X + c_Y + i s_Xs_Y)}{2}
            \\end{pmatrix}^{2^\\tau}\\\\
            &= T^{2^\\tau},
        \\end{align*}

    with

    .. math::
        \\begin{align*}
            X &= 2^{-\\tau}x,\\\\
            Y &= 2^{-\\tau}y,\\\\
            Z &= 2^{-\\tau}z,\\\\
            Q &= 2^{-\\tau}q,\\\\
            c_{\\theta} &= \\cos(\\theta),\\\\
            s_{\\theta} &= \\sin(\\theta).
        \\end{align*}
    
    Once :math:`T` is calculated, it is then recursively squared :math:`\\tau` times to obtain :math:`\\exp(A)`.

    Parameters
    ----------
    exponent : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix to take the exponential of.

        .. warning::
            Will overwrite the original contents of this as part of the algorithm.
        
    result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix which the result of the exponentiation is to be written to.
    trotter_cutoff : `int`
        The number of squares to make to the approximate matrix (:math:`\\tau` above).
    """
    hyper_cube_amount = math.ceil(trotter_cutoff/2)
    if hyper_cube_amount < 0:
        hyper_cube_amount = 0
    precision = 4**hyper_cube_amount
    
    x = source_sample[0]/(2*precision)
    y = source_sample[1]/(2*precision)
    z = source_sample[2]/(2*precision)

    cx = math.cos(x)
    sx = math.sin(x)
    cy = math.cos(y)
    sy = math.sin(y)

    cisz = math.cos(z) + 1j*math.sin(z)

    result[0, 0] = (cx*cy - 1j*sx*sy)/cisz
    result[1, 0] = (cx*sy -1j*sx*cy)/cisz

    result[0, 1] = -(cx*sy + 1j*sx*cy)*cisz
    result[1, 1] = (cx*cy + 1j*sx*sy)*cisz

    temporary = cuda.local.array((2, 2), dtype = nb.complex128)
    for power_index in range(hyper_cube_amount):
        matrix_multiply(result, result, temporary)
        matrix_multiply(temporary, temporary, result)

@cuda.jit(device = True, inline = True)
def matrix_exponential_trotter(exponent, result, trotter_cutoff):
    """
    Calculates a matrix exponential based on the Lie Product Formula,

    .. math::
        \\exp(A + B) = \\lim_{c \\to \\infty} \\left(\\exp\\left(\\frac{1}{c}A\\right) \\exp\\left(\\frac{1}{c}B\\right)\\right)^c.

    Assumes the exponent is an imaginary  linear combination of a subspace of :math:`su(3)`, being,

    .. math::
        \\begin{align*}
            A &= -i(x F_x + y F_y + z F_z + q F_q),
        \\end{align*}
    
    with

    .. math::
        \\begin{align*}
            F_x &= \\frac{1}{\\sqrt{2}}\\begin{pmatrix}
                0 & 1 & 0 \\\\
                1 & 0 & 1 \\\\
                0 & 1 & 0
            \\end{pmatrix},&
            F_y &= \\frac{1}{\\sqrt{2}}\\begin{pmatrix}
                0 & -i &  0 \\\\
                i &  0 & -i \\\\
                0 &  i &  0
            \\end{pmatrix},\\\\
            F_z &= \\begin{pmatrix}
                1 & 0 &  0 \\\\
                0 & 0 &  0 \\\\
                0 & 0 & -1
            \\end{pmatrix},&
            F_q &= \\frac{1}{3}\\begin{pmatrix}
                1 &  0 & 0 \\\\
                0 & -2 & 0 \\\\
                0 &  0 & 1
            \\end{pmatrix}
        \\end{align*}

    Then the exponential can be approximated as, for large :math:`\\tau`,

    .. math::
        \\begin{align*}
            \\exp(A) &= \\exp(-ix F_x - iy F_y - iz F_z - iq F_q)\\\\
            &= \\exp(2^{-\\tau}(-ix F_x - iy F_y - iz F_z - iq F_q))^{2^\\tau}\\\\
            &\\approx (\\exp(-i(2^{-\\tau} x) F_x) \\exp(-i(2^{-\\tau} y) F_y) \\exp(-i(2^{-\\tau} z F_z + (2^{-\\tau} q) F_q)))^{2^\\tau}\\\\
            &= \\begin{pmatrix}
                \\frac{e^{-i\\left(Z + \\frac{Q}{3}\\right)}(c_X + c_Y - i s_Xs_Y)}{2} & \\frac{e^{i\\frac{2Q}{3}} (-s_Y -i c_Y s_X)}{\\sqrt{2}} & \\frac{e^{-i\\left(-Z + \\frac{Q}{3}\\right)}(c_X - c_Y + i s_Xs_Y)}{2} \\\\
                \\frac{e^{-i\\left(Z + \\frac{Q}{3}\\right)} (-i s_X + c_X s_Y)}{\\sqrt{2}} & e^{i\\frac{2Q}{3}} c_X c_Y & \\frac{e^{-i(Z - \\frac{Q}{3})} (-i s_X - c_X s_Y)}{\\sqrt{2}} \\\\
                \\frac{e^{-i\\left(Z + \\frac{Q}{3}\\right)}(c_X - c_Y - i s_Xs_Y)}{2} & \\frac{e^{i\\frac{2Q}{3}} (s_Y -i c_Y s_X)}{\\sqrt{2}} & \\frac{e^{-i\\left(-Z + \\frac{Q}{3}\\right)}(c_X + c_Y + i s_Xs_Y)}{2}
            \\end{pmatrix}^{2^\\tau}\\\\
            &= T^{2^\\tau},
        \\end{align*}

    with

    .. math::
        \\begin{align*}
            X &= 2^{-\\tau}x,\\\\
            Y &= 2^{-\\tau}y,\\\\
            Z &= 2^{-\\tau}z,\\\\
            Q &= 2^{-\\tau}q,\\\\
            c_{\\theta} &= \\cos(\\theta),\\\\
            s_{\\theta} &= \\sin(\\theta).
        \\end{align*}
    
    Once :math:`T` is calculated, it is then recursively squared :math:`\\tau` times to obtain :math:`\\exp(A)`.

    Parameters
    ----------
    exponent : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix to take the exponential of.

        .. warning::
            Will overwrite the original contents of this as part of the algorithm.
        
    result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix which the result of the exponentiation is to be written to.
    trotter_cutoff : `int`
        The number of squares to make to the approximate matrix (:math:`\\tau` above).
    """
    x = (1j*exponent[1, 0]).real
    y = (1j*exponent[1, 0]).imag
    z = (1j*(exponent[0, 0])).real

    # hyper_cube_amount = 2*math.ceil(trotter_cutoff/4 + math.log(math.fabs(x) + math.fabs(y) + math.fabs(z) + math.fabs(q))/(4*math.log(2.0)))
    hyper_cube_amount = math.ceil(trotter_cutoff/2)
    if hyper_cube_amount < 0:
        hyper_cube_amount = 0
    precision = 4**hyper_cube_amount
    
    x /= precision
    y /= precision
    z /= precision

    cx = math.cos(x)
    sx = math.sin(x)
    cy = math.cos(y)
    sy = math.sin(y)

    cisz = math.cos(z) + 1j*math.sin(z)

    result[0, 0] = (cx*cy - 1j*sx*sy)/cisz
    result[1, 0] = (cx*sy -1j*sx*cy)/cisz

    result[0, 1] = -(cx*sy + 1j*sx*cy)*cisz
    result[1, 1] = (cx*cy + 1j*sx*sy)*cisz

    for power_index in range(hyper_cube_amount):
        matrix_multiply(result, result, exponent)
        matrix_multiply(exponent, exponent, result)

@cuda.jit(device = True, inline = True)
def matrix_exponential_taylor(exponent, result, cutoff):
    """
    Calculate a matrix exponential using a Taylor series. The matrix being exponentiated is complex, and of any dimension.

    The exponential is approximated as

    .. math::
        \\begin{align*}
        \\exp(A) &= \\sum_{k = 0}^\\infty \\frac{1}{k!} A^k\\\\
        &\\approx \\sum_{k = 0}^{c - 1} \\frac{1}{k!} A^k\\\\
        &= \\sum_{k = 0}^{c - 1} T_k, \\textrm{ with}\\\\
        T_0 &= 1,\\\\
        T_k &= \\frac{1}{k} T_{k - 1} A.
        \\end{align*}

    Parameters
    ----------
    exponent : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix to take the exponential of.
    result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix which the result of the exponentiation is to be written to.
    cutoff : `int`
        The number of terms in the Taylor expansion (:math:`c` above).
    """
    T = cuda.local.array((2, 2), dtype = nb.complex128)
    TOld = cuda.local.array((2, 2), dtype = nb.complex128)
    set_to_one(T)
    set_to_one(result)

    # exp(A) = 1 + A + A^2/2 + ...
    for taylor_index in range(cutoff):
        # TOld = T
        for x_index in nb.prange(2):
            for y_index in nb.prange(2):
                TOld[y_index, x_index] = T[y_index, x_index]
        # T = TOld*A/n
        for x_index in nb.prange(2):
            for y_index in nb.prange(2):
                T[y_index, x_index] = 0
                for z_index in range(2):
                    T[y_index, x_index] += (TOld[y_index, z_index]*exponent[z_index, x_index])/(taylor_index + 1)
        # E = E + T
        for x_index in nb.prange(2):
            for y_index in nb.prange(2):
                result[y_index, x_index] += T[y_index, x_index]

@cuda.jit(device = True, inline = True)
def norm2(z):
    """
    The 2 norm of a complex vector.

    .. math::
        \|a + ib\|_2 = \\sqrt {\\left(\\sum_i a_i^2 + b_i^2\\right)}

    Parameters
    ----------
    z : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (index)
        The vector to take the 2 norm of.
    Returns
    -------
    nz : :class:`numpy.double`
        The 2 norm of z.
    """
    return math.sqrt(z[0].real**2 + z[0].imag**2 + z[1].real**2 + z[1].imag**2)

@cuda.jit(device = True, inline = True)
def inner(left, right):
    """
    The inner (maths convention dot) product between two complex vectors. 
    
    .. note::
        The mathematics definition is used here rather than the physics definition, so the left vector is scalar.conjugated. Thus the inner product of two orthogonal vectors is 0.

    .. math::
        \\begin{align*}
        l \\cdot r &\\equiv \\langle l, r \\rangle\\\\
        l \\cdot r &= \\sum_i (l_i)^* r_i
        \\end{align*}

    Parameters
    ----------
    left : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (index)
        The vector to left multiply in the inner product.
    right : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (index)
        The vector to right multiply in the inner product.
    
    Returns
    -------
    d : :class:`numpy.cdouble`
        The inner product of l and r.
    """
    return scalar.conj(left[0])*right[0] + scalar.conj(left[1])*right[1]

@cuda.jit(device = True, inline = True)
def set_to(operator, result):
    """
    Copy the contents of one matrix into another.

    .. math::
        (A)_{i, j} = (B)_{i, j}

    Parameters
    ----------
    operator : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix to copy from.
    result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix to copy to.
    """
    for x_index in range(2):
        for y_index in range(2):
            result[y_index, x_index] = operator[y_index, x_index]

@cuda.jit(device = True, inline = True)
def set_to_one(operator):
    """
    Make a matrix the multiplicative identity, ie, :math:`1`.

    .. math::
        \\begin{align*}
        (A)_{i, j} &= \\delta_{i, j}\\\\
        &= \\begin{cases}
            1,&i = j\\\\
            0,&i\\neq j
        \\end{cases}
        \\end{align*}

    Parameters
    ----------
    operator : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix to set to :math:`1`.
    """
    operator[0, 0] = 1
    operator[1, 0] = 0

    operator[0, 1] = 0
    operator[1, 1] = 1

@cuda.jit(device = True, inline = True)
def set_to_zero(operator):
    """
    Make a matrix the additive identity, ie, :math:`0`.

    .. math::
        \\begin{align*}
        (A)_{i, j} = 0
        \\end{align*}

    Parameters
    ----------
    operator : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix to set to :math:`0`.
    """
    operator[0, 0] = 0
    operator[1, 0] = 0

    operator[0, 1] = 0
    operator[1, 1] = 0

@cuda.jit(device = True, inline = True)
def matrix_multiply(left, right, result):
    """
    Multiply matrices left and right together, to be returned in result.

    .. math::
        \\begin{align*}
        (LR)_{i,k} = \\sum_j (L)_{i,j} (R)_{j,k}
        \\end{align*}

    Parameters
    ----------
    left : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix to left multiply by.
    right : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix to right multiply by.
    result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        A matrix to be filled with the result of the product.
    """
    result[0, 0] = left[0, 0]*right[0, 0] + left[0, 1]*right[1, 0]
    result[1, 0] = left[1, 0]*right[0, 0] + left[1, 1]*right[1, 0]

    result[0, 1] = left[0, 0]*right[0, 1] + left[0, 1]*right[1, 1]
    result[1, 1] = left[1, 0]*right[0, 1] + left[1, 1]*right[1, 1]

@cuda.jit(device = True, inline = True)
def adjoint(operator, result):
    """
    Takes the hermitian adjoint of a matrix.

    .. math::
        \\begin{align*}
        A^\\dagger &\\equiv A^H\\\\
        (A^\\dagger)_{y,x} &= ((A)_{x,y})^*
        \\end{align*}
    
    Matrix can be in :math:`\\mathbb{C}^{2\\times2}` or :math:`\\mathbb{C}^{3\\times3}`.

    Parameters
    ----------
    operator : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The operator to take the adjoint of.
    result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        An array to write the resultant adjoint to.
    """
    result[0, 0] = scalar.conj(operator[0, 0])
    result[1, 0] = scalar.conj(operator[0, 1])

    result[0, 1] = scalar.conj(operator[1, 0])
    result[1, 1] = scalar.conj(operator[1, 1])