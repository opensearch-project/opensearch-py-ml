# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""wavelet_tools

Tools for Haar discrete wavelet transform and signal reconstruction.

Functions
---------
circularize
    Circularizes a given filter to a specified length T.

haar_matrix
    Construct the DWT matrix for a Haar wavelet decomposition.

haar_approx
    Piecewise constant approximation of a signal by reconstruction
    from a truncated Haar wavelet representation.
"""

import math

# imports
import torch
import torch.nn.functional as F


def haar_approx(signal: torch.Tensor, truncate: int) -> torch.Tensor:
    """
    Generate the Haar DWT matrix for a given signal length (must be a power of 2).

    :param signal:
    :type signal: torch.Tensor
    :param truncate:
    :type truncate:  int
    :return: Haar DWT matrix
    :rtype: torch.Tensor
    """
    N = len(signal)
    J = math.ceil(
        math.log(N) / math.log(2)
    )  # change of base req'd as log2 not supported in torchscript
    is_pow2 = math.floor(math.log(N) / math.log(2)) == J

    # zero pad signal if N is not power of 2
    if not is_pow2:
        signal = F.pad(signal, (0, int(2**J - N)))

    # transform with Haar DWT
    haar_mat = __haar_matrix(int(2**J))
    haar_rep = torch.matmul(haar_mat, signal)

    # truncate haar representation if specified
    if truncate is not None:
        if not is_pow2:
            truncate += J  # additional slack to account for effects of zero-padding

        ixs = torch.argsort(torch.pow(haar_rep, 2))
        not_top_b = ixs[:-truncate]
        haar_rep[not_top_b] = 0

    # return piecewise constant haar approximation
    approx = torch.matmul(haar_mat.t(), haar_rep)

    return approx[:N]  # remove any zero-padding


def __haar_matrix(T: int) -> torch.Tensor:
    """
    Generate the Haar DWT matrix for a given signal length (must be a power of 2).

    :param T:
    :type T: int
    :return: Haar DWT matrix
    :rtype: torch.Tensor
    """

    J = int(
        math.log(T) / math.log(2)
    )  # change of base req'd as log2 not supported in torchscript
    T = int(T)
    mat = torch.zeros((0, T))
    smooth_mat = torch.eye(T)

    # wavelet and scale filters are hardcoded local variables
    wavelet = torch.FloatTensor([-1 / math.sqrt(2), 1 / math.sqrt(2)])
    scale = torch.FloatTensor([1 / math.sqrt(2), 1 / math.sqrt(2)])

    for j in range(J):
        # circularize filters to length T / 2**j
        t = int(T / 2**j)
        scirc = __circularize(scale, t)
        dcirc = __circularize(wavelet, t)

        # construct detail matrix via even shifts
        detail_mat = torch.vstack([torch.roll(dcirc, 2 * x) for x in range(t // 2)])

        # update DWT matrix
        mat = torch.vstack((mat, torch.mm(detail_mat, smooth_mat)))

        # update smooth
        new_smooth = torch.vstack([torch.roll(scirc, 2 * x) for x in range(t // 2)])
        smooth_mat = torch.mm(new_smooth, smooth_mat)

    return torch.vstack(
        (mat, smooth_mat)
    )  # add final smooth resolution before returning


def __circularize(signal: torch.Tensor, T: int) -> torch.Tensor:
    """
    This method circularize a given signal to length T.

    :param signal: signal
    :type signal: torch tensor
    :param T: signal
    :type T: int
    """

    N = len(signal)

    if T >= N:  # T >= length of the signal : circularization is just zero-padding
        circ = F.pad(signal, (0, T - N))
    else:  # T < length of the signal : requires periodic summation of filter
        newlen = T * (N // T + 1)
        circ = F.pad(signal, (0, int(newlen - N)))
        circ = torch.sum(torch.reshape(circ, (int(N // T + 1), int(T))), dim=0)
    return circ
