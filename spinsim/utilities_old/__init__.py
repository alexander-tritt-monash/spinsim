"""
Some device functions for doing maths and interpolation with :mod:`numba.cuda`.
"""

import math
import numpy as np
import numba as nb
from numba import cuda

from . import scalar
from . import spin_half
from . import spin_one

#@cuda.jit(device = True, inline = True)
def interpolate_source_linear(source, time_sample, time_step_source, source_sample):
    """
    Samples the source using linear interpolation.

    Parameters
    ----------
    source : :class:`numpy.ndarray` of :class:`numpy.double` (spatial_index, time_source_index)
        The input source of the environment of the spin system. In units of Hz.
    time_sample : `float`
        The time to be sampled.
    time_step_source : `float`
        The time step between source samples.
    source_sample : :class:`numpy.ndarray` of :class:`numpy.double` (spatial_index)
        The interpolated value of source.
    """
    time_sample = time_sample/time_step_source
    time_index_source = int(time_sample)
    time_sample -= time_index_source

    time_index_source_next = time_index_source + 1
    if time_index_source_next > source.shape[0] - 1:
        time_index_source_next = source.shape[0] - 1

    for spacial_index in range(source.shape[1]):
        source_sample[spacial_index] = (time_sample*(source[time_index_source_next, spacial_index] - source[time_index_source, spacial_index]) + source[time_index_source, spacial_index])

#@cuda.jit(device = True, inline = True)
def interpolate_source_cubic(source, time_sample, time_step_source, source_sample):
    """
    Samples the source using cubic interpolation.

    Parameters
    ----------
    source : :class:`numpy.ndarray` of :class:`numpy.double` (spatial_index, time_source_index)
        The input source of the environment of the spin system. In units of Hz.
    time_sample : `float`
        The time to be sampled.
    time_step_source : `float`
        The time step between source samples.
    source_sample : :class:`numpy.ndarray` of :class:`numpy.double` (spatial_index)
        The interpolated value of source.
    """
    time_sample = time_sample/time_step_source
    time_index_source = int(time_sample)
    time_sample -= time_index_source

    time_index_source_next = time_index_source + 1
    if time_index_source_next > source.shape[0] - 1:
        time_index_source_next = source.shape[0] - 1
    time_indexSource_next_next = time_index_source_next + 1
    if time_indexSource_next_next > source.shape[0] - 1:
        time_indexSource_next_next = source.shape[0] - 1
    time_index_source_previous = time_index_source - 1
    if time_index_source_previous < 0:
        time_index_source_previous = 0
    for spacial_index in range(source.shape[1]):
        gradient = (source[time_index_source_next, spacial_index] - source[time_index_source_previous, spacial_index])/2
        gradient_next = (source[time_indexSource_next_next, spacial_index] - source[time_index_source, spacial_index])/2

        source_sample[spacial_index] = (\
            (2*(time_sample**3) - 3*(time_sample**2) + 1)*source[time_index_source, spacial_index] + \
            ((time_sample**3) - 2*(time_sample**2) + time_sample)*gradient + \
            (-2*(time_sample**3) + 3*(time_sample**2))*source[time_index_source_next, spacial_index] + \
            (time_sample**3 - time_sample**2)*gradient_next)