# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""wavelet_piecewise_const

Tools for fast piecewise-constant signal approximation from Haar wavelet reconstruction.

Functions
---------
piecewise_const_to_segs
    Segment-based representation of a piecewise constant signal.

blockwise_dp_segment
    DP-based piecewise constant approximation of a piecewise-constant signal.

blockwise_seg_to_approx
    Convert output of blockwise_dp_segment to a signal-length approximation.

lavielle_criterion
    Criterion for selecting number of segments.

wavelet_piecewise_const
    Fast piecewise constant approximation via Haar wavelet reconstruction.

tranform_piecewise_const
    Convert piecewise constant signal to activity score.
"""
import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .wavelet_tools import haar_approx


def wavelet_piecewise_const(
    data: torch.Tensor, m: int, max_dp_levels: int, is_haar_approx: bool
) -> torch.Tensor:
    """
    Fast, wavelet-based piecewise constant signal approximation for row m of the M x T input data tensor.
    :param data:
    :type data: torch.Tensor
    :param m:
    :type m: int
    :param max_dp_levels
    :type max_dp_levels: int
    :param is_haar_approx
    :type is_haar_approx: bool
    :return:
    :rtype: torch.Tensor
    """
    seq = data[m, :]
    if (torch.abs(torch.diff(seq)) > 0).sum() == 0:
        return seq  # avoid cost and numerical errors from needless approx
    elif is_haar_approx:  # compute Haar piecewise constant approx
        N = len(seq)
        J = math.ceil(math.log(N) / math.log(2))
        wav_approx = haar_approx(seq, truncate=J * max_dp_levels)
    else:
        wav_approx = seq  # no wavelet approximation -> slower DP

    # refine with block wise DP
    segs, costs = block_wise_dp_segment(wav_approx, max_dp_levels)

    # select segmentation level with lavielle criterion
    nseg, _ = lavielle_criterion(costs[:, -1])

    return block_wise_seg_to_approx(wav_approx, segs[nseg])


def block_wise_dp_segment(
    data: torch.Tensor, max_lvls: int
) -> Tuple[Dict[int, List[int]], torch.Tensor]:
    """
    Piecewise constant approximation to a piecewise constant signal, i.e. the best approximation at a more granular
        (fewer segments) resolution, via dynamic programming.
    :param data:
    :type data: torch.Tensor
    :param max_lvls:
    :type max_lvls: int
    :return:
    :rtype: Tuple[Dict[int, List[int]], torch.Tensor]
    """

    vals, counts = piecewise_const_to_segs(data)
    ssum: List[float] = [
        float(x) for x in torch.cumsum(vals * counts, 0)
    ]  # need to do this as tensor.tolist() gives a torch.jit runtime error
    sqsum: List[float] = [
        float(x) for x in torch.cumsum(torch.pow(vals, 2) * counts, 0)
    ]
    scounts: List[float] = [float(x) for x in torch.cumsum(counts, 0)]

    N = len(vals)
    segs = {int(i + 1): [0] for i in range(max_lvls)}
    costs = torch.zeros((max_lvls, N))

    # initialize cost and segmentation in base case w/ b=0 (no jumps)
    segs[1] = [0, int(N)]
    best_last_segs = [[0]] * N

    costs[0, :] = torch.zeros(N)
    for i in range(N):
        costs[0, i] = sqerr_seg_list(0, i, ssum, sqsum, scounts)

    # recursion
    for b in range(1, max_lvls):
        best_segs: List[List[int]] = []

        for i in range(N):
            if i <= b:  # more breaks than segs in data -> 0 cost
                seg_i: List[int] = []
                for x in range(
                    i + 1
                ):  # expand list comp here as torchscript can't handle it
                    seg_i += [x]
                best_segs += [seg_i]
                costs[b, i] = 0

            else:  # DP recursion
                addcosts = torch.zeros(i)
                for j in range(i):
                    addcosts[j] = sqerr_seg_list(j + 1, i, ssum, sqsum, scounts)
                new_jump_costs = costs[b - 1, :i] + addcosts

                new_jump_ix = int(torch.argmin(new_jump_costs))
                new_jump_cost = new_jump_costs[new_jump_ix]

                best_segs += [best_last_segs[new_jump_ix] + [new_jump_ix + 1]]
                costs[b, i] = new_jump_cost

        best_last_segs = best_segs
        segs[b + 1] = best_segs[-1] + [int(N)]

    return segs, costs


def lavielle_criterion(costs: torch.Tensor) -> Tuple[int, torch.Tensor]:
    """
    Criterion to automatically select number of segments in a piecewise constant approximation.
        Reference: https://hal.inria.fr/inria-00070662/document
    :param costs:
    :type costs: torch.Tensor
    :return:
    :rtype: Tuple[int, torch.Tensor]
    """
    s = 0.75  # hardcode cutoff at paper suggestion
    kmax = len(costs)
    normcosts = (costs[-1] - costs) / (costs[-1] - costs[0]) * (kmax - 1) + 1

    # catch NaNs arising from div-by-zero above
    # only happens when cost[1] = cost[kmax], so correct choice is 1 segment
    normcosts[normcosts.isnan()] = -1

    d = torch.zeros(len(costs))
    d[0] = (
        4 * torch.max(torch.abs(normcosts)) + 1.1 * s
    )  # upper bound and always strictly > s
    for i in range(1, len(costs) - 1):
        d[i] = (
            normcosts[i - 1] - 2 * normcosts[i] + normcosts[i + 1]
        )  # discrete 2nd deriv

    return int(torch.max(torch.nonzero(d > s)[:, 0]) + 1), d  # largest k s.t. d[k]>s


def block_wise_seg_to_approx(data: torch.Tensor, seg: List[int]) -> torch.Tensor:
    """
    Convert blockwise DP segmentation output to piecewise constant approximation.
    :param costs:
    :type costs: torch.Tensor
    :return:
    :rtype: Tuple[int, torch.Tensor]
    """
    vals, counts = piecewise_const_to_segs(data)
    scounts = F.pad(
        torch.cumsum(counts, 0), (1, 0)
    )  # prepend zero to get indices right

    apx = torch.zeros(len(data))
    for i in range(1, len(seg)):
        e = seg[i]  # DP segments are left-closed, right-open
        s = seg[i - 1]

        eix = scounts[e]
        six = scounts[s]

        apx[six:eix] = torch.mean(data[six:eix])

    return apx


def piecewise_const_to_segs(signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Represent a piecewise-constant signal in terms of its segments, each given by a value and length.
    :param signal:
    :type signal: torch.Tensor
    :return: output -> the output list of unique scalar elements, counts ->
        represents the number of occurrences for each unique value or tensor.
    :rtype: torch.Tensor, torch.Tensor
    """
    vals, counts = torch.unique_consecutive(signal, return_counts=True)
    return vals, counts


def sqerr_seg_list(
    st: int, e: int, ssum: List[float], sqsum: List[float], scounts: List[float]
) -> float:
    """
    Blockwise version of square loss.
    :param st:
    :type st: int
    :param e:
    :type e: int
    :param ssum:
    :type ssum: List[float]
    :param sqsum:
    :type sqsum: List[float]
    :param scounts:
    :type scounts: List[float]
    :return:
    :rtype:
    """

    assert e >= st
    s = ssum[e] - ssum[st - 1] if st > 0 else ssum[e]
    sq = sqsum[e] - sqsum[st - 1] if st > 0 else sqsum[e]
    n = scounts[e] - scounts[st - 1] if st > 0 else scounts[e]
    return sq - (1 / n) * s**2


def transform_piecewise_const(sig: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """
    Convert piecewise constant approximation to activity score.
    :param sig: signal
    :type sig: torch.Tensor
    :param bandwidth:
    :type bandwidth: float
    :return:
    :rtype: torch.Tensor
    """
    sigdiff = torch.abs(torch.diff(sig))

    # logic required to avoid bad representation of spikes
    chg = torch.where(sigdiff > 0)[0]
    for ix in chg:
        if torch.any(chg == ix + 1):
            sigdiff[ix + 1] = 0

    m = torch.linalg.norm(sigdiff)
    if m != 0:
        sigdiff /= m  # note this may amplify small variations

    blurred = gconv(sigdiff, bandwidth)
    return blurred


def gconv(data: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """
    1-D convolution with boundary padding.
    :param data:
    :type data: torch.Tensor
    :param bandwidth:
    :type bandwidth: float
    :return:
    :rtype: torch.Tensor
    """

    filt = gfilt(bandwidth)
    padlen = len(filt) // 2
    padded = boundary_pad(data, (padlen, padlen))

    spdat = torch.unsqueeze(torch.unsqueeze(padded, 0), 0)
    sfilt = torch.unsqueeze(torch.unsqueeze(filt, 0), 0)
    res = F.conv1d(spdat, sfilt)
    return res.squeeze()


def gfilt(sigma: float, truncate: float = 4.0) -> torch.Tensor:
    """
    Pytorch Gaussian filter with truncation.
    :param sigma: sigma
    :type sigma: float
    :param truncate:
    :type truncate: float, default value 4.0
    :return:
    :rtype: torch.Tensor
    """

    lw = int(truncate * sigma + 0.5)
    xs = torch.arange(-lw, lw + 1)
    sigma2 = math.pow(sigma, 2)
    weights = torch.exp(-0.5 / sigma2 * torch.pow(xs, 2))
    weights = weights / torch.sum(weights)

    return weights


def boundary_pad(sig: torch.Tensor, lens: List[int]) -> torch.Tensor:
    """
    Pad signal by extending value at boundary ('nearest' in scipy).
    :param sig: signal
    :type sig: float
    :param lens:
    :type lens: List[int]
    :return:
    :rtype: torch.Tensor
    """

    lpadded = F.pad(sig, (lens[0], 0), mode="constant", value=sig[0])
    padded = F.pad(lpadded, (0, lens[1]), mode="constant", value=sig[-1])
    return padded
