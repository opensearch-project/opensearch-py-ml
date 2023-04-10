# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.
from typing import Dict, List

import torch

from .event_detection import assign_metrics_to_events, find_events
from .nmf import NMF
from .utils import BANDWIDTH, DT_PVAL, NMF_MAX_ITER, NMF_TOL, OMP_TOL
from .wavelet_piecewise_const import transform_piecewise_const, wavelet_piecewise_const


class MCorr(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bandwidth = BANDWIDTH
        self.nmf_max_iter = NMF_MAX_ITER
        self.nmf_tol = NMF_TOL
        self.NMF = NMF()

        self.dt_pval = DT_PVAL
        self.omp_tol = OMP_TOL

    def forward(
        self, metrics: torch.Tensor, max_events: int = 5, wavelet_approx: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Main entry point of the Metrics Correlation algorithm.

        :param metrics:
        :type metrics: torch.Tensor
        :param max_events:
        :type max_events: int
        :param wavelet_approx
        :type wavelet_approx: int
        :return:
        :rtype: List[int, Dict[str, torch.Tensor]]
        """
        # Validate inputs
        M, T = metrics.shape
        if M >= T:
            raise ValueError(
                "The number of metrics to correlate must be smaller than the length of each time series."
            )

        #
        # Step 1: piecewise constant approx. and activity scoring for input metrics
        #
        num_metrics, T = metrics.shape
        max_events = min(max_events, num_metrics)
        self.bandwidth = float(max(T // 32, 1))
        activity_scores = self.compute_activity_scores(
            metrics, max_events, wavelet_approx
        )

        #
        # Step 2: event detection and metrics assignment
        #
        loadings, patterns = self.NMF(
            activity_scores,
            max_events,
            max_iter=self.nmf_max_iter,
            tol=self.nmf_tol,
        )

        events = find_events(patterns, dt_pval=self.dt_pval)
        events = assign_metrics_to_events(events, activity_scores, omp_tol=self.omp_tol)

        return events

    def compute_activity_scores(
        self, metrics: torch.Tensor, max_events: int, wavelet_approx: bool
    ) -> torch.Tensor:
        M, T = metrics.shape
        activity_scores = torch.zeros((M, T - 1))
        for m in range(
            M
        ):  # NOTE: ideally we'd be able to parallelize this, as in the non-script version
            apx = wavelet_piecewise_const(metrics, m, 2 * max_events, wavelet_approx)
            activity_scores[m, :] = transform_piecewise_const(apx, self.bandwidth)
        return activity_scores
