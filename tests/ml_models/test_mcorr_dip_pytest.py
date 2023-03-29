# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""
Tests of correctness and compilation for the dip test module
of metrics correlation.

"True" values here can be verified by comparison to established
implementations such as the R package 'diptest'.
"""

import os

import numpy as np
import torch

import opensearch_py_ml.ml_models.metrics_correlation.dip as dip


def test_dip_statistic():
    '''
    test correctness by computing the dip statistic on 
    various input sequences. comparisons are made to the
    equivalent output from the R package 'diptest', the 
    canonical implementation of this method.
    '''

    T = 128

    # uniform eCDF
    unifseq = torch.arange(T)
    assert torch.abs(dip.dip(unifseq) - 0.003906) <= 1e-6

    # uniform with large spike
    unifspike = torch.arange(T)
    unifspike[:50] = 20

    assert torch.abs(dip.dip(unifspike) - 0.085426) <= 1e-6

    # unimodal Gaussian
    xs = torch.arange(-T // 2, T // 2 + 1)
    x = torch.exp(-(xs**2) / 20)

    assert torch.abs(dip.dip(x) - 0.010517) <= 1e-6

    # max of 0.25 is (approx) achieved for (approx) bimodal data
    apx_bimodal = torch.cat(
        ((torch.normal(1.0, 0.001, [T // 2])), torch.normal(2.0, 0.001, [T // 2]))
    )

    assert dip.dip(apx_bimodal) >= 0.2475 and dip.dip(apx_bimodal) <= 0.25

    # overlapping bimodal (mixture of Gaussians)
    x_mog = torch.exp(-((xs - 5.0) ** 2) / 5) + torch.exp(-((xs + 5.0) ** 2) / 10)

    assert torch.abs(dip.dip(x_mog) - 0.013016) <= 1e-6


def test_dip_pval():
    '''
    as above, reference values (this time for p-values)
    are obtained from R package 'diptest'
    '''

    # bimodal : p near zero
    T = 256
    xa = torch.arange(T + 1) / (T / 10)
    xb = torch.arange(T + 1) / (T / 30)
    x = torch.cat((xa - 10, xb + 15))
    _, p = dip.diptest(x)

    assert p > 0 and p < 0.01

    # unimodal : p near one
    x = torch.cat((xa + 15, xb + 5))
    _, p = dip.diptest(x)

    assert p < 1.0 and p > 0.98

    # in-between cases to check interpolation
    x = torch.cat((xa - 7, xb + 5))
    _, p = dip.diptest(x)

    assert np.abs(p - 0.57) < 0.01

    T = 512
    xa = torch.arange(T + 1) / (T / 10)
    xb = torch.arange(T + 1) / (T / 30)
    x = torch.cat((xa - 7, xb + 5))
    _, p = dip.diptest(x)

    assert np.abs(p - 0.09) < 0.01

    T = 1024
    xa = torch.arange(T + 1) / (T / 10)
    xb = torch.arange(T + 1) / (T / 30)
    x = torch.cat((xa - 6, xb + 5))
    _, p = dip.diptest(x)

    assert np.abs(p - 0.54) < 0.01


def test_interp():
    xs = torch.linspace(0, 1, 12) * 2 * torch.pi
    ys = torch.sin(xs)

    # should linearly interpolate within region
    for _ in range(5):
        newx = torch.rand([1])
        newy = dip.interp(newx, xs, ys)
        newy_np = np.interp(newx.numpy(), xs.numpy(), ys.numpy())

        assert np.abs(newy - newy_np) <= 1e-6

    # should return endpoints if outside (i.e. constant interp)
    newy_low = dip.interp(torch.tensor([-1.0]), xs, ys)
    newy_hi = dip.interp(torch.tensor([10.0]), xs, ys)

    assert newy_low == ys[0].item()
    assert newy_hi == ys[-1].item()


def test_torchscript_scripting():
    # test scripting and save / load of scripted model
    model = DipTest()

    try:
        jit_processing = torch.jit.script(model)
    except Exception as exec:
        assert False, f"Failed to script model with exception {exec}"

    try:
        torch.jit.save(jit_processing, "diptest.pt")
    except Exception as exec:
        assert False, f"Failed to save model with exception {exec}"

    try:
        _ = torch.jit.load("diptest.pt")
    except Exception as exec:
        assert False, f"Failed to load scripted model with exception {exec}"

    os.remove("diptest.pt")


def test_torchscript_output():
    # check scripted model output is same as pytorch
    sig = torch.normal(mean=0, std=1, size=[128])

    model = DipTest()
    ptD, ptP = model(sig)

    model = DipTest()
    jit_processing = torch.jit.script(model)
    torch.jit.save(jit_processing, "diptest.pt")
    loaded_model = torch.jit.load("diptest.pt")

    tsD, tsP = loaded_model.forward(sig)

    assert np.allclose(ptD, tsD, rtol=0, atol=0)
    assert np.allclose(ptP, tsP, rtol=0, atol=0)

    os.remove("diptest.pt")


class DipTest(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, sig: torch.Tensor):
        return dip.diptest(sig)
