# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from typing import Dict, List

import torch

from .dip import diptest


def find_events(
    patterns: torch.Tensor, dt_pval: float
) -> List[Dict[str, torch.Tensor]]:
    """
    Select events from the set of patterns obtained by decomposition
    of the activity scores.

    :param patterns: Tensor of shape (p, T), with $p$ the number of patterns
        and $T$ the pattern length.
    :type patterns: torch.Tensor
    :param dt_pval: P-value threshold for event detection. Passed to the dip
        test.
    :type dt_pval: float
    :return: List of detected events. Elements of the list are
        dictionaries with keys 'range' and 'event'; 'range' gives the interval
        over which the event occurred, and 'event' is a length-T array of the
        event intensity.
    :rtype: List[int, Dict[str, torch.Tensor]]
    """

    candidates = find_unimodal_events(patterns, dt_pval)
    events = merge_events(candidates)
    return events


def find_unimodal_events(patterns: torch.Tensor, dt_pval: float) -> torch.Tensor:
    """
    Given a set of patterns, return those that are unimodal. Unimodality is
    evaluated via the dip test.

    :param patterns: Tensor of shape (p, T), with $p$ the number of patterns
        and $T$ the pattern length.
    :type patterns: torch.Tensor
    :param dt_pval: P-value threshold for event detection. Passed to the dip
        test.
    :type dt_pval: float
    :return: Tensor of shape (p_u, T), with $p_u$ the number of unimodal
        patterns detected.
    :rtype: torch.Tensor
    """
    E = patterns.shape[0]
    dt_pvals = torch.zeros(E)

    for e in range(E):
        _, pval = diptest(patterns[e, :])
        dt_pvals[e] = pval

    return patterns[dt_pvals > dt_pval, :]


def merge_events(
    candidates: torch.Tensor,
) -> List[Dict[str, torch.Tensor]]:
    """
    Merge candidate events if they have sufficient overlap. Returns a
    structured object containing all final events, with information on
    their duration and intensity.

    :param candidates: Tensor of shape (p, T), with $p$ the number of
        candidate (i.e. unimodal) patterns and $T$ the pattern length.
    :type candidates: torch.Tensor
    :return: List of detected events. Elements of the list are
        dictionaries with keys 'event_window' and 'event_pattern';
        'event_window' gives the interval over which the event occurred,
        and 'event_pattern' is a length-T array of the event intensity.
    :rtype: List[int, Dict[str, torch.Tensor]]
    """
    E, T = candidates.shape

    starts = torch.zeros(E)
    ends = torch.zeros(E)
    for e in range(E):  # define event start and end by cumsum quantiles
        event = candidates[e, :]
        ecdf = torch.cumsum(event, dim=0) / torch.sum(event)
        starts[e] = torch.sum(ecdf < 0.1)
        ends[e] = torch.sum(ecdf < 0.9)
    ix = torch.argsort(starts)

    sstart = starts[ix]
    send = ends[ix]
    sevents = candidates[ix, :]

    merged: List[
        Dict[str, torch.Tensor]
    ] = []  # merge in linear pass over time dimension
    currstart = torch.tensor([-1])
    currend = torch.tensor([-1])
    currevent = torch.ones(T) * -1.0

    for i, s in enumerate(sstart):
        if i == 0:  # first event
            currstart = s
            currend = send[i]
            currevent = sevents[i, :]
        elif s > currend:  # start of new event; record previous one
            merged.append(
                {
                    "event_window": torch.stack([currstart, currend]),
                    "event_pattern": currevent,
                }
            )
            currstart = s
            currend = send[i]
            currevent = sevents[i, :]
        else:  # update current event
            currend = torch.max(torch.stack([currend, send[i]]))
            currevent += sevents[i, :]

    # record final event
    if currstart > 0:
        merged.append(
            {
                "event_window": torch.stack([currstart, currend]),
                "event_pattern": currevent,
            }
        )
    return merged


def assign_metrics_to_events(
    events: List[Dict[str, torch.Tensor]],
    activity_scores: torch.Tensor,
    omp_tol: float,
) -> List[Dict[str, torch.Tensor]]:
    """
    Given a set of activity scores and the events detected, assign each
    activity score (i.e. metric) to an event. Each metric can be assigned
    to zero, one, or multiple events.


    :param events: Events detected in the activity score data.
    :type events: List[int, Dict[str, torch.Tensor]]
    :param activity_scores: The activiy scores from which the events were
        computed.
    :type activity_scores: torch.Tensor
    :param omp_tol: Threshold for event assignment via orthogonal matching
        pursuit. Each next event is assigned to the metric if it explains a
        proportion of the residual variance at least equal to `omp_tol`.
    :type omp_tol: float
    :return: A list with the same structure as the `events` parameter,
        where each element (event) now also has a 'metrics' field listing the
        indices of the metrics assigned to that event.
    :rtype: List[int, Dict[str, torch.Tensor]]
    """
    (
        M,
        T,
    ) = activity_scores.shape  # loop over metrics, getting their event assignments

    massign = {m: omp_assign(events, activity_scores[m, :], omp_tol) for m in range(M)}

    # invert this to get list of metrics per event
    evs: List[int] = []
    for v in massign.values():
        for ev in v:
            if ev not in evs:
                evs += [ev]

    for e in evs:
        mlist: List[int] = []
        for k, v in massign.items():
            if int(e) in v:
                mlist += [k]

        events[int(e)]["suspected_metrics"] = torch.tensor(mlist)

    return events


def omp_assign(
    events: List[Dict[str, torch.Tensor]],
    activity_score: torch.Tensor,
    omp_tol: float,
) -> List[int]:
    """
    Orthogonal matching pursuit for event assignment. Assignment is defined in
    terms of explanation: a metric is assigned to an event if that event
    explains a significant fraction of the metric's activity score. For each
    metric, the set of explanatory events is determined by orthogonal matching
    pursuit regression of the activity score against the set of event
    intensities.

    :param events: The events used to explain a metric's activity score.
        Assumed to have the same structure as the output of `find_events()`.
    :type events: dict[int, dict[str, torch.Tensor]]
    :param activity_score: 1-D activity score of the metric.
    :type activity_score: torch.Tensor
    :param omp_tol: Threshold for event assignment via orthogonal matching
        pursuit. Each next event is assigned to the metric if it explains a
        proportion of the residual variance at least equal to `omp_tol`.
    :type omp_tol: float
    :return: List of events assigned to the metric.
    :rtype: List[int]
    """

    if torch.var(activity_score, dim=0, unbiased=False) == 0:
        return []  # skip assignment if activity score is flat

    # initialize for OMP loop
    assign: List[int] = []
    assigned_events = torch.empty((len(activity_score), 0))
    cvg = False
    events_left: List[int] = torch.arange(len(events)).tolist()
    r = activity_score

    if len(events_left) == 0:  # case where no events were found upstream
        return assign

    while not cvg:
        # get remaining event patterns
        event_mat = torch.stack(
            [events[i]["event_pattern"] for i in events_left], dim=1
        )
        event_mat /= torch.linalg.norm(event_mat, dim=0)

        # find event that best explains activity score remainder
        corrs = torch.matmul(event_mat.t(), r)
        eix = int(torch.argmax(torch.abs(corrs)))

        # add it if it sufficiently reduces approximation error
        assigned_events_trial = torch.cat(
            (assigned_events, event_mat[:, eix].unsqueeze(1)), dim=1
        )
        coef = torch.linalg.solve(
            torch.mm(assigned_events_trial.t(), assigned_events_trial),
            torch.matmul(assigned_events_trial.t(), activity_score),
        )
        rnew = activity_score - torch.matmul(assigned_events_trial, coef)

        rel_sqerr_decrease = torch.sum(rnew**2) / torch.sum(r**2)
        if rel_sqerr_decrease < omp_tol:
            r = rnew
            assign += [events_left[eix]]
            assigned_events = assigned_events_trial
            events_left.pop(eix)

            if len(events_left) == 0:
                cvg = True  # end if no events remain

        else:
            cvg = True  # no remaining events sufficiently explain data

    return assign
