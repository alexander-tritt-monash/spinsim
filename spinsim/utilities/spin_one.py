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
    
    x = source_sample[0]/precision
    y = source_sample[1]/precision
    z = source_sample[2]/precision
    q = source_sample[3]/precision

    cx = math.cos(x)
    sx = math.sin(x)
    cy = math.cos(y)
    sy = math.sin(y)

    cisz = math.cos(z + q/3) - 1j*math.sin(z + q/3)
    result[0, 0] = 0.5*cisz*(cx + cy - 1j*sx*sy)
    result[1, 0] = cisz*(-1j*sx + cx*sy)/sqrt2
    result[2, 0] = 0.5*cisz*(cx - cy - 1j*sx*sy)

    cisz = math.cos(2*q/3) + 1j*math.sin(2*q/3)
    result[0, 1] = cisz*(-sy - 1j*cy*sx)/sqrt2
    result[1, 1] = cisz*cx*cy
    result[2, 1] = cisz*(sy - 1j*cy*sx)/sqrt2

    cisz = math.cos(z - q/3) + 1j*math.sin(z - q/3)
    result[0, 2] = 0.5*cisz*(cx - cy + 1j*sx*sy)
    result[1, 2] = cisz*(-1j*sx - cx*sy)/sqrt2
    result[2, 2] = 0.5*cisz*(cx + cy + 1j*sx*sy)

    temporary = cuda.local.array((3, 3), dtype = nb.complex128)
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
    # if exponent.shape[0] == 2:
    #     x = (1j*exponent[1, 0]).real
    #     y = (1j*exponent[1, 0]).imag
    #     z = (1j*(exponent[0, 0])).real
    # else:
    x = (1j*exponent[1, 0]).real*sqrt2
    y = (1j*exponent[1, 0]).imag*sqrt2
    z = (1j*(exponent[0, 0] + 0.5*exponent[1, 1])).real
    q = (-1.5*1j*exponent[1, 1]).real

    # hyper_cube_amount = 2*math.ceil(trotter_cutoff/4 + math.log(math.fabs(x) + math.fabs(y) + math.fabs(z) + math.fabs(q))/(4*math.log(2.0)))
    hyper_cube_amount = math.ceil(trotter_cutoff/2)
    if hyper_cube_amount < 0:
        hyper_cube_amount = 0
    precision = 4**hyper_cube_amount
    
    x /= precision
    y /= precision
    z /= precision
    q /= precision

    cx = math.cos(x)
    sx = math.sin(x)
    cy = math.cos(y)
    sy = math.sin(y)

    # if exponent.shape[0] == 2:
    #     cisz = math.cos(z) + 1j*math.sin(z)

    #     result[0, 0] = (cx*cy - 1j*sx*sy)/cisz
    #     result[1, 0] = (cx*sy -1j*sx*cy)/cisz

    #     result[0, 1] = -(cx*sy + 1j*sx*cy)*cisz
    #     result[1, 1] = (cx*cy + 1j*sx*sy)*cisz
    # else:
    cisz = math.cos(z + q/3) - 1j*math.sin(z + q/3)
    result[0, 0] = 0.5*cisz*(cx + cy - 1j*sx*sy)
    result[1, 0] = cisz*(-1j*sx + cx*sy)/sqrt2
    result[2, 0] = 0.5*cisz*(cx - cy - 1j*sx*sy)

    cisz = math.cos(2*q/3) + 1j*math.sin(2*q/3)
    result[0, 1] = cisz*(-sy - 1j*cy*sx)/sqrt2
    result[1, 1] = cisz*cx*cy
    result[2, 1] = cisz*(sy - 1j*cy*sx)/sqrt2

    cisz = math.cos(z - q/3) + 1j*math.sin(z - q/3)
    result[0, 2] = 0.5*cisz*(cx - cy + 1j*sx*sy)
    result[1, 2] = cisz*(-1j*sx - cx*sy)/sqrt2
    result[2, 2] = 0.5*cisz*(cx + cy + 1j*sx*sy)

    for power_index in range(hyper_cube_amount):
        matrix_multiply(result, result, exponent)
        matrix_multiply(exponent, exponent, result)

@cuda.jit(device = True, inline = True)
def matrix_exponential_rotating_wave_hybrid(exponent, result, trotter_cutoff):
    """
    An experimental matrix exponential based off :func:`matrix_exponential_lie_trotter()` and the rotating wave approximation. Not as accurate as :func:`matrix_exponential_lie_trotter()`.

    Parameters
    ----------
    exponent : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix to take the exponential of.
    result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix which the result of the exponentiation is to be written to.
    trotter_cutoff : `int`
        The number of squares to make to the approximate matrix.
    """
    cisz = (1j*(exponent[0, 0] + 0.5*exponent[1, 1]))
    cisz = math.cos(cisz.real) + 1j*math.sin(cisz.real)

    q = (-1.5*1j*exponent[1, 1]).real
    A = scalar.complex_abs(exponent[1, 0])*sqrt2
    cisa = 1j*exponent[1, 0]*sqrt2/(A*cisz)
    
    hyper_cube_amount = 0
    if q != 0.0:
        hyper_cube_amount = 2*math.ceil(trotter_cutoff/4 + math.log(A + math.fabs(q))/(4*math.log(2.0)))
        if hyper_cube_amount < 0:
            hyper_cube_amount = 0
    precision = 4**hyper_cube_amount

    q /= precision
    A /= precision

    cisq = math.cos(q/6) + 1j*math.sin(q/6)

    result[0, 0] = 0.5*(math.cos(A) + 1)*(cisq*cisq)
    result[1, 0] = -1j*math.sin(A)/(cisa*cisq*sqrt2)
    result[2, 0] = 0.5*(math.cos(A) - 1)*((cisq/cisa)*(cisq/cisa))

    result[0, 1] = result[1, 0]*(cisa*cisa)
    result[1, 1] = math.cos(A)/(cisq*cisq*cisq*cisq)
    result[2, 1] = result[1, 0]

    result[0, 2] = result[2, 0]*(cisa*cisa*cisa*cisa)
    result[1, 2] = result[0, 1]
    result[2, 2] = result[0, 0]

    for power_index in range(hyper_cube_amount):
        matrix_multiply(result, result, exponent)
        matrix_multiply(exponent, exponent, result)

    result[0, 0] /= cisz
    result[1, 0] /= cisz
    result[2, 0] /= cisz

    result[0, 2] *= cisz
    result[1, 2] *= cisz
    result[2, 2] *= cisz

@cuda.jit(device = True, inline = True)
def matrix_exponential_cross_product(exponent, result):
    """
    Calculate a matrix exponential by diagonalisation using the cross product. This is the main method used in the old cython code.

    Parameters
    ----------
    exponent : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix to take the exponential of.
    result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix which the result of the exponentiation is to be written to.
    """
    diagonal = cuda.local.array(3, dtype = nb.float64)          # Eigenvalues
    winding = cuda.local.array(3, dtype = nb.complex128)        # e^i eigenvalues
    rotation = cuda.local.array((3, 3), dtype = nb.complex128)  # Eigenvectors
    shifted = cuda.local.array((3, 3), dtype = nb.complex128)   # Temp, for eigenvalue algorithm
    scaled = cuda.local.array((3, 3), dtype = nb.complex128)    # Temp, for eigenvalue algorithm

    set_to_one(rotation)

    # Find eigenvalues
    # Based off a trigonometric solution to the cubic characteristic equation
    # First shift and scale the matrix to a nice position
    shift = 1j*(exponent[0, 0] + exponent[1, 1] + exponent[2, 2])/3
    for x_index in range(3):
        for y_index in range(3):
            shifted[y_index, x_index] = 1j*exponent[y_index, x_index]
            if y_index == x_index:
                shifted[y_index, x_index] -= shift
    matrix_multiply(shifted, shifted, scaled)
    scale = math.sqrt((scaled[0, 0] + scaled[1, 1] + scaled[2, 2]).real/6)
    if scale > 0:
        for x_index in range(3):
            for y_index in range(3):
                scaled[y_index, x_index] = shifted[y_index, x_index]/scale
    # Now find the determinant of the shifted and scaled matrix
    det_scaled = 0
    for x_index in range(3):
        det_partial_plus = 1
        det_partial_minus = 1
        for y_index in range(3):
            det_partial_plus *= scaled[y_index, (x_index + y_index)%3]
            det_partial_minus *= scaled[2 - y_index, (x_index + y_index)%3]
        det_scaled += (det_partial_plus - det_partial_minus).real
    # The eigenvalues of the matrix are related to the determinant of the scaled matrix
    for diagonal_index in range(3):
        diagonal[diagonal_index] = scale*2*math.cos((1/3)*math.acos(det_scaled/2) + (2/3)*math.pi*diagonal_index)

    # First eigenvector
    # Eigenvector y of eigenvalue s of A are in the kernel of (A - s 1).
    # Something something first isomorphism theorem =>
    # y is parallel to all vectors in the coimage of (A - s 1), ie the image of (A - s 1)* (hermitian adjoint,
    # sorry, can't type a dagger, will use maths notation), ie the column space of (A - s 1)*. Love me some linear
    # algebra. Can find such a vector by cross producting vectors in this coimage.

    # Find (A - s 1)*
    for x_index in range(3):
        for y_index in range(3):
            shifted[y_index, x_index] = 1j*exponent[y_index, x_index]
            if y_index == x_index:
                shifted[y_index, x_index] -= diagonal[0]
    adjoint(shifted, scaled)
    
    # Normalise vectors in the coimage of (A - s 1)
    hasAlready_found_it = False
    for x_index in range(3):
        norm = norm2(scaled[:, x_index])
        if norm > machine_epsilon:
            for y_index in range(3):
                scaled[y_index, x_index] /= norm
        else:
            # If the column of (A - s 1)* has size 0, then e1 is an eigenvector.
            # Not more work needs to be done to find it.
            for y_index in range(3):
                if x_index == y_index:
                    rotation[y_index, 0] = 1
                else:
                    rotation[y_index, 0] = 0
                hasAlready_found_it = True

    if not hasAlready_found_it:
        # Find the cross product of vectors in the coimage
        cross(scaled[:, 0], scaled[:, 1], rotation[:, 0])
        norm = norm2(rotation[:, 0])
        if norm > machine_epsilon:
            # If these vectors cross to something non-zero then we've found one
            for x_index in range(3):
                rotation[x_index, 0] /= norm
        else:
            # Otherwise these are parallel, and we should cross other vectors to see if that helps
            cross(scaled[:, 1], scaled[:, 2], rotation[:, 0])
            norm = norm2(rotation[:, 0])
            if norm > machine_epsilon:
                for x_index in range(3):
                    rotation[x_index, 0] /= norm
            else:
                # If it's still zero the we should have found it already, so panic.
                print("RIIIIP")

    # Second eigenvector
    for x_index in range(3):
        for y_index in range(3):
            shifted[y_index, x_index] = 1j*exponent[y_index, x_index]
            if y_index == x_index:
                shifted[y_index, x_index] -= diagonal[1]
    adjoint(shifted, scaled)
    hasAlready_found_it = False
    for x_index in range(3):
        norm = norm2(scaled[:, x_index])
        if norm > machine_epsilon:
            for y_index in range(3):
                scaled[y_index, x_index] /= norm
        else:
            for y_index in range(3):
                if x_index == y_index:
                    rotation[y_index, 1] = 1
                else:
                    rotation[y_index, 1] = 0
                hasAlready_found_it = True
    # If not part of same eigenspace as the first one the proceed as normal,
    # otherwise find a vector orthogonal to the first eigenvector
    if math.fabs(diagonal[0] - diagonal[1]) > machine_epsilon:
        if not hasAlready_found_it:
            cross(scaled[:, 0], scaled[:, 1], rotation[:, 1])
            norm = norm2(rotation[:, 1])
            if norm > machine_epsilon:
                for x_index in range(3):
                    rotation[x_index, 1] /= norm
            else:
                cross(scaled[:, 1], scaled[:, 2], rotation[:, 1])
                norm = norm2(rotation[:, 1])
                if norm > machine_epsilon:
                    for x_index in range(3):
                        rotation[x_index, 1] /= norm
                else:
                    print("RIIIIP")
    else:
        cross(scaled[:, 0], rotation[:, 0], rotation[:, 1])
        norm = norm2(rotation[:, 1])
        if norm > machine_epsilon:
            for x_index in range(3):
                rotation[x_index, 1] /= norm
        else:
            cross(scaled[:, 1], rotation[:, 0], rotation[:, 1])
            norm = norm2(rotation[:, 1])
            if norm > machine_epsilon:
                for x_index in range(3):
                    rotation[x_index, 1] /= norm
            else:
                print("RIIIIP")

    # Third eigenvector
    # The third eigenvector is orthogonal to the first two
    cross(rotation[:, 0], rotation[:, 1], rotation[:, 2])

    # winding = exp(-j w)
    for x_index in range(3):
        winding[x_index] = math.cos(diagonal[x_index]) - 1j*math.sin(diagonal[x_index])
    # U = exp( Q* diag(-j w) Q ) = Q* diag(exp(-j w)) Q
    for x_index in range(3):
        for y_index in range(3):
            result[y_index, x_index] = 0
            for z_index in range(3):
                result[y_index, x_index] += rotation[y_index, z_index]*winding[z_index]*scalar.conj(rotation[x_index, z_index])

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
    if exponent.shape[0] == 2:
        T = cuda.local.array((2, 2), dtype = nb.complex128)
        TOld = cuda.local.array((2, 2), dtype = nb.complex128)
    else:
        T = cuda.local.array((3, 3), dtype = nb.complex128)
        TOld = cuda.local.array((3, 3), dtype = nb.complex128)
    set_to_one(T)
    set_to_one(result)

    # exp(A) = 1 + A + A^2/2 + ...
    for taylor_index in range(cutoff):
        # TOld = T
        for x_index in nb.prange(exponent.shape[0]):
            for y_index in nb.prange(exponent.shape[0]):
                TOld[y_index, x_index] = T[y_index, x_index]
        # T = TOld*A/n
        for x_index in nb.prange(exponent.shape[0]):
            for y_index in nb.prange(exponent.shape[0]):
                T[y_index, x_index] = 0
                for z_index in range(exponent.shape[0]):
                    T[y_index, x_index] += (TOld[y_index, z_index]*exponent[z_index, x_index])/(taylor_index + 1)
        # E = E + T
        for x_index in nb.prange(exponent.shape[0]):
            for y_index in nb.prange(exponent.shape[0]):
                result[y_index, x_index] += T[y_index, x_index]

@cuda.jit(device = True, inline = True)
def matrix_exponential_ql(exponent, result):
    """
    This is based on a method in the old cython code - this was found to not function as intended either here or in the cython code. Assumes the exponent is an imaginary  linear combination of a subspace of :math:`su(3)`, being,

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
    
    Parameters
    ----------
    exponent : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix to take the exponential of.
    result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (y_index, x_index)
        The matrix which the result of the exponentiation is to be written to.
    cutoff : `int`
        The number of terms in the Taylor expansion (:math:`c` above).
    """
    diagonal = cuda.local.array(3, dtype = nb.float64)
    off_diagonal = cuda.local.array(2, dtype = nb.float64)
    winding = cuda.local.array(3, dtype = nb.complex128)
    rotation = cuda.local.array((3, 3), dtype = nb.complex128)

    set_to_one(result)

    # initialise
    #       ( e^ i phi                  ) ( Bz            Bphi/sqrt2             ) ( e^ i phi                  )
    # -i    (             1             ) ( Bphi/sqrt2                Bphi/sqrt2 ) (             1             )
    #       (                 e^ -i phi ) (               Bphi/sqrt2  -Bz        ) (                 e^ -i phi )

    diagonal[0] = -exponent[0, 0].imag
    diagonal[1] = -exponent[1, 1].imag
    diagonal[2] = -exponent[2, 2].imag
    off_diagonal[0] = scalar.complex_abs(exponent[1, 0])
    off_diagonal[1] = scalar.complex_abs(exponent[2, 1])
    # off_diagonal[2] = 0

    set_to_one(rotation)
    if off_diagonal[0] > 0:
        rotation[0, 0] = 1j*exponent[1, 0]/off_diagonal[0]
    if off_diagonal[1] > 0:
        rotation[2, 2] = 1j*exponent[2, 1]/off_diagonal[1]

    for off_diagonal_index in range(0, 2):
        iteration_index = 0
        while True:
            # If floating point arithmetic can't tell the difference between the size of the off diagonals and zero, then we're good and can stop
            for offDiagonal_compare_index in range(off_diagonal_index, 2):
                order_of_magnitude = math.fabs(diagonal[offDiagonal_compare_index]) + math.fabs(diagonal[offDiagonal_compare_index + 1])
                if math.fabs(off_diagonal[offDiagonal_compare_index]) + order_of_magnitude == order_of_magnitude:
                    break
                # if math.fabs(off_diagonal[offDiagonal_compare_index])/order_of_magnitude < 1e-6:
                #     break
            if offDiagonal_compare_index == off_diagonal_index:
                break
            
            iteration_index += 1
            if iteration_index > 60:
                print("yote")
                break

            temporaryG = 0.5*(diagonal[off_diagonal_index + 1] - diagonal[off_diagonal_index])/off_diagonal[off_diagonal_index]
            size = math.sqrt(temporaryG**2 + 1.0)
            if temporaryG > 0:
                temporaryG = diagonal[offDiagonal_compare_index] - diagonal[off_diagonal_index] + off_diagonal[off_diagonal_index]/(temporaryG + size)
            else:
                temporaryG = diagonal[offDiagonal_compare_index] - diagonal[off_diagonal_index] + off_diagonal[off_diagonal_index]/(temporaryG - size)

            diagonalise_sine = 1.0
            diagonalise_cosine = 1.0
            temporaryP = 0.0
            temporaryB = 0.0
            for offDiagonal_calculation_index in range(offDiagonal_compare_index - 1, off_diagonal_index - 1, -1):
                temporaryF = diagonalise_sine*off_diagonal[offDiagonal_calculation_index]
                temporaryB = diagonalise_cosine*off_diagonal[offDiagonal_calculation_index]
                if math.fabs(temporaryF) > math.fabs(temporaryG):
                    diagonalise_cosine = temporaryG/temporaryF
                    size = math.sqrt(diagonalise_cosine**2 + 1.0)
                    off_diagonal[offDiagonal_calculation_index + 1] = temporaryF*size
                    diagonalise_sine = 1.0/size
                    diagonalise_cosine *= diagonalise_sine
                else:
                    diagonalise_sine = temporaryF/temporaryG
                    size = math.sqrt(diagonalise_sine**2 + 1.0)
                    off_diagonal[offDiagonal_calculation_index + 1] = temporaryG*size
                    diagonalise_cosine = 1.0/size
                    diagonalise_sine *= diagonalise_cosine
            temporaryG = diagonal[offDiagonal_calculation_index + 1] - temporaryP
            size = (diagonal[offDiagonal_calculation_index] - temporaryG)*diagonalise_sine + 2.0*diagonalise_cosine*temporaryB
            temporaryP = diagonalise_sine*size
            diagonal[offDiagonal_calculation_index + 1] = temporaryG + temporaryP
            temporaryG = diagonalise_cosine*size - temporaryB

            for rotation_index in range(0, 3):
                rotation_previous = rotation[rotation_index, offDiagonal_calculation_index + 1]
                rotation[rotation_index, offDiagonal_calculation_index + 1] = diagonalise_sine*rotation[rotation_index, offDiagonal_calculation_index] + diagonalise_cosine*rotation_previous
                rotation[rotation_index, offDiagonal_calculation_index] = diagonalise_cosine*rotation[rotation_index, offDiagonal_calculation_index] - diagonalise_sine*rotation_previous

            diagonal[off_diagonal_index] -= temporaryP
            off_diagonal[off_diagonal_index] = temporaryG
            off_diagonal[offDiagonal_compare_index] = 0.0
    # winding = exp(-j w)
    for x_index in range(3):
        winding[x_index] = math.cos(diagonal[x_index]) - 1j*math.sin(diagonal[x_index])
    if cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x == 0:
        print(diagonal[0])
    # U = exp( Q* diag(-j w) Q ) = Q* diag(exp(-j w)) Q
    for x_index in range(3):
        for y_index in range(3):
            result[y_index, x_index] = 0
            for z_index in range(3):
                result[y_index, x_index] += winding[z_index]*scalar.conj(rotation[x_index, z_index])*rotation[y_index, z_index]
    return

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
    # Original spin 1:
    return math.sqrt(z[0].real**2 + z[0].imag**2 + z[1].real**2 + z[1].imag**2 + z[2].real**2 + z[2].imag**2)

@cuda.jit(device = True, inline = True)
def cross(left, right, result):
    """
    The cross product of two vectors in :math:`\\mathbb{C}^3`.
    
    .. note::
        The mathematics definition is used here rather than the physics definition. This is the scalar.conjugate of the real cross product, since this produces a vector orthogonal to the two inputs.

    .. math::
        \\begin{align*}
        (l \\times r)_1 &= (l_2 r_3 - l_3 r_2)^*,\\\\
        (l \\times r)_2 &= (l_3 r_1 - l_1 r_3)^*,\\\\
        (l \\times r)_3 &= (l_1 r_2 - l_2 r_1)^*,\\\\
        l, r &\\in \\mathbb{C}^3
        \\end{align*}

    Parameters
    ----------
    left : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (index)
        The vector to left multiply in the cross product.
    right : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (index)
        The vector to right multiply in the cross product.
    result : :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` of :class:`numpy.cdouble`, (index)
        An array for the resultant vector to be written to.
    """
    result[0] = scalar.conj(left[1]*right[2] - left[2]*right[1])
    result[1] = scalar.conj(left[2]*right[0] - left[0]*right[2])
    result[2] = scalar.conj(left[0]*right[1] - left[1]*right[0])

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
    return scalar.conj(left[0])*right[0] + scalar.conj(left[1])*right[1] + scalar.conj(left[2])*right[2]

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
    result[0, 0] = operator[0, 0]
    result[1, 0] = operator[1, 0]
    result[2, 0] = operator[2, 0]

    result[0, 1] = operator[0, 1]
    result[1, 1] = operator[1, 1]
    result[2, 1] = operator[2, 1]

    result[0, 2] = operator[0, 2]
    result[1, 2] = operator[1, 2]
    result[2, 2] = operator[2, 2]

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
    operator[2, 0] = 0

    operator[0, 1] = 0
    operator[1, 1] = 1
    operator[2, 1] = 0

    operator[0, 2] = 0
    operator[1, 2] = 0
    operator[2, 2] = 1

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
    operator[2, 0] = 0

    operator[0, 1] = 0
    operator[1, 1] = 0
    operator[2, 1] = 0

    operator[0, 2] = 0
    operator[1, 2] = 0
    operator[2, 2] = 0

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
    result[0, 0] = left[0, 0]*right[0, 0] + left[0, 1]*right[1, 0] + left[0, 2]*right[2, 0]
    result[1, 0] = left[1, 0]*right[0, 0] + left[1, 1]*right[1, 0] + left[1, 2]*right[2, 0]
    result[2, 0] = left[2, 0]*right[0, 0] + left[2, 1]*right[1, 0] + left[2, 2]*right[2, 0]

    result[0, 1] = left[0, 0]*right[0, 1] + left[0, 1]*right[1, 1] + left[0, 2]*right[2, 1]
    result[1, 1] = left[1, 0]*right[0, 1] + left[1, 1]*right[1, 1] + left[1, 2]*right[2, 1]
    result[2, 1] = left[2, 0]*right[0, 1] + left[2, 1]*right[1, 1] + left[2, 2]*right[2, 1]

    result[0, 2] = left[0, 0]*right[0, 2] + left[0, 1]*right[1, 2] + left[0, 2]*right[2, 2]
    result[1, 2] = left[1, 0]*right[0, 2] + left[1, 1]*right[1, 2] + left[1, 2]*right[2, 2]
    result[2, 2] = left[2, 0]*right[0, 2] + left[2, 1]*right[1, 2] + left[2, 2]*right[2, 2]

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
    result[2, 0] = scalar.conj(operator[0, 2])

    result[0, 1] = scalar.conj(operator[1, 0])
    result[1, 1] = scalar.conj(operator[1, 1])
    result[2, 1] = scalar.conj(operator[1, 2])

    result[0, 2] = scalar.conj(operator[2, 0])
    result[1, 2] = scalar.conj(operator[2, 1])
    result[2, 2] = scalar.conj(operator[2, 2])