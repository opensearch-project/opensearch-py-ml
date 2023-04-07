# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""
Tests of correctness and compilation for metrics correlation.
"""

import os

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score

import opensearch_py_ml.ml_models.metrics_correlation.mcorr as mcorr

TESTDATA_FILENAME = os.path.join(
    os.path.dirname(os.path.abspath("__file__")),
    "tests",
    "ml_models",
    "metrics_correlation",
    "mcorr_data.pt",
)


def test_level_invariance():
    """
    Results should not depend on level of metrics.
    Test by comparing results on randomly shifted data.
    """
    test_data = torch.load(TESTDATA_FILENAME)["mcorr_level"]
    model = mcorr.MCorr()

    M, _ = test_data.shape
    events_noshift = model.forward(test_data, max_events=5)

    for _ in range(3):
        shifts = torch.normal(mean=0, std=100, size=[M, 1])
        events_shift = model.forward(test_data + shifts, max_events=5)

        assert len(events_noshift) == len(events_shift)
        for e_ns, e_s in zip(events_noshift, events_shift):
            torch.testing.assert_close(
                e_ns["event_window"], e_s["event_window"], atol=0, rtol=0
            )
            torch.testing.assert_close(
                e_ns["suspected_metrics"], e_s["suspected_metrics"], atol=0, rtol=0
            )


def test_const_metrics():
    """
    An important property is that metrics with no variation
    are guaranteed to never be assigned to an event.
    """
    test_data = torch.load(TESTDATA_FILENAME)["mcorr_const"]
    M, T = test_data.shape
    model = mcorr.MCorr()

    # add constant metrics to normal data, check results same
    events_before = model.forward(test_data, max_events=3)

    M_const = 5
    const_dat = torch.ones((M_const, T)) * torch.normal(0, 20, size=[M_const, 1])
    const_ix = list(range(M, M + M_const))
    full_data = torch.cat((test_data, const_dat), dim=0)

    events_with_const = model.forward(full_data, max_events=3)

    assert len(events_before) == len(events_with_const)
    for e_b, e_wc in zip(events_before, events_with_const):
        torch.testing.assert_close(
            e_b["event_window"], e_wc["event_window"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            e_b["suspected_metrics"], e_wc["suspected_metrics"], atol=0, rtol=0
        )

        # no constant metrics in e_wc events
        assert len(set(e_wc["suspected_metrics"]).intersection(set(const_ix))) == 0

    # all-constant metrics should have no events
    events_const = model.forward(const_dat)
    assert len(events_const) == 0


def test_metrics_correlation():
    """
    Evaluate event finding and metrics assignment.
    """
    test_data = torch.load(TESTDATA_FILENAME)["mcorr_test"]
    model = mcorr.MCorr()

    for dataset in test_data:
        dat = dataset["data"]
        M, _ = dat.shape

        labels = dataset["labels"]

        event_preds = model.forward(dat, max_events=5)

        # correct # events
        assert len(event_preds) == len(labels)

        precs, recs = compute_event_accuracy(event_preds, labels, M)

        # accurate metrics assignment per event
        assert np.all([x > 0.7 for x in precs])
        assert np.all([x > 0.5 for x in recs])


def compute_event_accuracy(event_preds, event_labels, M):
    precs = []
    recs = []
    for e in event_preds:
        ran = e["event_window"]
        lt = None
        for k in event_labels.keys():
            if k > ran[0] and k < ran[1]:
                lt = k

        true_mets = event_labels[lt] if lt else []

        tm_bin = np.zeros(M)
        tm_bin[true_mets] = 1

        pm_bin = np.zeros(M)
        pm_bin[e["suspected_metrics"].tolist()] = 1

        precs += [precision_score(tm_bin, pm_bin)]
        recs += [recall_score(tm_bin, pm_bin)]

        return precs, recs


def test_torchscript_scripting():
    # test scripting and save / load of scripted model
    model = mcorr.MCorr()

    try:
        jit_processing = torch.jit.script(model)
    except Exception as exec:
        assert False, f"Failed to script model with exception {exec}"

    try:
        torch.jit.save(jit_processing, "mcorrtest.pt")
    except Exception as exec:
        assert False, f"Failed to save model with exception {exec}"

    try:
        _ = torch.jit.load("mcorrtest.pt")
    except Exception as exec:
        assert False, f"Failed to load scripted model with exception {exec}"

    os.remove("mcorrtest.pt")


def test_torchscript_output():
    # data with nontrivial event structure
    sig = torch.normal(mean=0, std=1, size=[5, 128])

    model = mcorr.MCorr()
    pt_events = model(sig)

    model = mcorr.MCorr()
    jit_processing = torch.jit.script(model)
    torch.jit.save(jit_processing, "mcorrtest.pt")
    loaded_model = torch.jit.load("mcorrtest.pt")

    ts_events = loaded_model.forward(sig)

    # check same number of events
    assert len(pt_events) == len(ts_events)

    # check events are identical
    for pte, tse in zip(pt_events, ts_events):
        for k in pte.keys():
            torch.testing.assert_close(pte[k], tse[k])

    os.remove("mcorrtest.pt")
