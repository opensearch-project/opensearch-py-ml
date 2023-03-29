# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""
Tests of correctness and compilation for the wavelet module
of metrics correlation.
"""

import os

import numpy as np
import pywt
import torch

import opensearch_py_ml.ml_models.metrics_correlation.wavelet_piecewise_const as wavpc
import opensearch_py_ml.ml_models.metrics_correlation.wavelet_tools as wavtools


def test_haar_approx():
    # no truncation -> should return original signal
    T = 32
    for _ in range(5):
        sig = torch.normal(mean=0, std=1, size=[T])
        torch.testing.assert_close(wavtools.haar_approx(sig, T), sig)


def test_haar_truncate():
    # test truncation against pywavelet library
    trunc = 10
    for _ in range(5):
        sig = torch.normal(mean=0, std=1, size=[32])
        signp = sig.numpy()

        dec = pywt.wavedec(signp, "haar")
        arr, slices = pywt.coeffs_to_array(dec)
        not_top = np.argsort(arr**2)[:-trunc]
        arr[not_top] = 0
        rec = pywt.waverec(
            pywt.array_to_coeffs(arr, slices, output_format="wavedec"), "haar"
        )

        ptapx = wavtools.haar_approx(sig, truncate=trunc)

        torch.testing.assert_close(ptapx, torch.FloatTensor(rec))


def test_haar_padding():
    # test zero padding
    T = 17
    for _ in range(5):
        sig = torch.normal(mean=0, std=1, size=[T])

        dec = pywt.wavedec(sig, "haar", mode="zero")
        rec = pywt.waverec(dec, "haar", mode="zero")[:T]

        ptapx = wavtools.haar_approx(sig, truncate=T)

        torch.testing.assert_close(ptapx, torch.FloatTensor(rec))


def test_lavielle_criterion():
    # no curvature (linear cost curve) -> select k=1
    costs = torch.arange(5) * -1 + 10
    k, _ = wavpc.lavielle_criterion(costs)
    assert k == 1

    # elbow at 3 -> select k=3
    costs = 1 / torch.arange(2, 12) ** 4
    k, _ = wavpc.lavielle_criterion(costs)
    assert k == 3


def test_blockwise_DP_seg():
    T = 32
    max_lvls = 5
    for _ in range(5):
        sig = torch.normal(mean=0, std=1, size=[T])
        segs, costs = wavpc.block_wise_dp_segment(sig, max_lvls=max_lvls)

        for m in range(max_lvls):
            # at each level i, segs should be len i+1 w/ increasing indices
            assert len(segs[m + 1]) == m + 2
            assert np.all(np.diff(segs[m + 1]) > 0)

            # at each level i, costs should not decrease across rows
            assert np.all(np.diff(costs[m, :]) >= 0)

        # costs should decrease down columns
        assert np.all(np.diff(costs[:, -1]) <= 0)


def test_seg_to_approx():
    T = 32
    for _ in range(5):
        sig = torch.normal(mean=0, std=1, size=[T])

        # generate segments
        nseg = np.random.choice(np.arange(1, 5))
        seg = (
            [0]
            + list(np.sort(np.random.choice(np.arange(1, T), nseg, replace=False)))
            + [T]
        )

        apx = wavpc.block_wise_seg_to_approx(sig, seg)

        # approximation should have # unique vals = # segments
        assert len(torch.unique(apx)) == len(seg) - 1

        for i in range(len(seg) - 1):
            seg_start = seg[i]
            seg_end = seg[i + 1] - 1

            # approximation should be constant inside segments
            assert len(torch.unique(apx[seg_start : (seg_end + 1)])) == 1

            # segments are left-closed, right open
            assert apx[seg_start] == apx[seg_end]
            if i < (len(seg) - 2):
                assert apx[seg_start] != apx[seg_end + 1]


def test_transform_piecewise_constant():
    T = 32

    # transform of all-zero input is all-zeros
    sig = torch.zeros(T)
    for bw in [0.1, 1.0, 10.0]:
        scores = wavpc.transform_piecewise_const(sig, bw)
        torch.testing.assert_close(sig[:-1], scores)

    # scores should always be non-negative
    sig[: T // 2] -= 10.0
    for _ in range(5):
        ix = np.random.choice(T)
        sig[ix:] += torch.normal(0, 1, [1]).item() * 5
        scores = wavpc.transform_piecewise_const(sig, 1.0)
        assert torch.all(scores >= 0)

    # blurring should reduce sharp changes
    # see https://en.wikipedia.org/wiki/Young%27s_convolution_inequality
    ixs = np.random.choice(T, 3, replace=False)
    sig[ixs] += 20

    scores = wavpc.transform_piecewise_const(sig, 1.0)
    assert torch.max(scores) <= torch.max(torch.diff(sig))


def test_torchscript_scripting():
    # test scripting and save / load of scripted model
    model = WavTest()

    try:
        jit_processing = torch.jit.script(model)
    except Exception as exec:
        assert False, f"Failed to script model with exception {exec}"

    try:
        torch.jit.save(jit_processing, "wavtest.pt")
    except Exception as exec:
        assert False, f"Failed to save model with exception {exec}"

    try:
        _ = torch.jit.load("wavtest.pt")
    except Exception as exec:
        assert False, f"Failed to load scripted model with exception {exec}"

    os.remove("wavtest.pt")


def test_torchscript_output():
    # check scripted model output is same as pytorch
    sig = torch.normal(mean=0, std=1, size=[1, 32])

    model = WavTest()
    ptscores = model(sig, 5, 2.0)

    model = WavTest()
    jit_processing = torch.jit.script(model)
    torch.jit.save(jit_processing, "wavtest.pt")
    loaded_model = torch.jit.load("wavtest.pt")

    tsscores = loaded_model.forward(sig, 5, 2.0)

    torch.testing.assert_close(ptscores, tsscores, rtol=0, atol=0)

    os.remove("wavtest.pt")


class WavTest(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, sig: torch.Tensor, max_dp_levels: int, bandwidth: float):
        apx = wavpc.wavelet_piecewise_const(sig, 0, max_dp_levels, True)
        scores = wavpc.transform_piecewise_const(apx, bandwidth)
        return scores
