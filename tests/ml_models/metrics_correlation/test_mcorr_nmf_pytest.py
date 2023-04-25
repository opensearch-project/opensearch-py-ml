# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""
Tests of correctness and compilation for the NMF test module
of metrics correlation.
"""

import os

import numpy as np
import torch
from sklearn.decomposition import NMF as skNMF
from sklearn.decomposition._nmf import _initialize_nmf

import opensearch_py_ml.ml_models.metrics_correlation.nmf as nmf


def test_nmf_init():
    # check correctness of nndsvd-a initialization
    T = 32
    M = 5
    n_component = 3

    nmf_model = nmf.NMF()
    for _ in range(5):
        X = torch.normal(mean=0, std=1, size=[M, T]) ** 2

        Wsk, Hsk = _initialize_nmf(X.numpy(), n_component, init="nndsvda")
        Wpt, Hpt = nmf_model.initialize(X, n_component)

        W_rel_err = np.linalg.norm(Wpt.numpy() - Wsk) / np.linalg.norm(Wsk)
        H_rel_err = np.linalg.norm(Hpt.numpy() - Hsk) / np.linalg.norm(Hsk)

        # check relative errors for matrix-level similarity
        # and to avoid failing due to ignorable elementwise
        # numerical differences
        assert W_rel_err <= 1e-4
        assert H_rel_err <= 1e-4


def test_nmf_forward():
    # compare multiplicative-update iterations
    T = 32
    M = 5
    iters = [5, 20, 50]

    n_component = 3
    nmf_model = nmf.NMF()
    sk_model = skNMF(n_components=n_component, solver="mu", init="nndsvda", max_iter=10)

    for _ in range(5):
        X = torch.normal(mean=0, std=1, size=[M, T]) ** 2

        for iter in iters:
            sk_model = skNMF(
                n_components=n_component, solver="mu", init="nndsvda", max_iter=iter
            )
            Wsk = sk_model.fit_transform(X.numpy())
            Hsk = sk_model.components_

            Wpt, Hpt = nmf_model(X, k=3, max_iter=iter, tol=1e-8)

            W_rel_err = np.linalg.norm(Wpt.numpy() - Wsk) / np.linalg.norm(Wsk)
            H_rel_err = np.linalg.norm(Hpt.numpy() - Hsk) / np.linalg.norm(Hsk)

            # check relative errors for matrix-level similarity
            # and to avoid failing due to ignorable elementwise
            # numerical differences
            assert W_rel_err <= 0.01
            assert H_rel_err <= 0.01


def test_torchscript_scripting():
    # test scripting and save / load of scripted model
    model = nmf.NMF()

    try:
        jit_processing = torch.jit.script(model)
    except Exception as exec:
        assert False, f"Failed to script model with exception {exec}"

    try:
        torch.jit.save(jit_processing, "nmftest.pt")
    except Exception as exec:
        assert False, f"Failed to save model with exception {exec}"

    try:
        _ = torch.jit.load("nmftest.pt")
    except Exception as exec:
        assert False, f"Failed to load scripted model with exception {exec}"

    os.remove("nmftest.pt")


def test_torchscript_output():
    # check scripted model output is same as pytorch
    X = torch.normal(mean=0, std=1, size=[5, 32]) ** 2

    model = nmf.NMF()
    ptW, ptH = model(X, 3, 200, 1e-6)

    model = nmf.NMF()
    jit_processing = torch.jit.script(model)
    torch.jit.save(jit_processing, "nmftest.pt")
    loaded_model = torch.jit.load("nmftest.pt")

    tsW, tsH = loaded_model.forward(X, 3, 200, 1e-6)

    torch.testing.assert_close(ptW, tsW, rtol=0, atol=0)
    torch.testing.assert_close(ptH, tsH, rtol=0, atol=0)

    os.remove("nmftest.pt")
