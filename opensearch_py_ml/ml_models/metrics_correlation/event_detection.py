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
) -> Dict[int, Dict[str, torch.Tensor]]:

    """
    find events

    :param patterns:
    :type patterns: torch.Tensor
    :param dt_pval:
    :type dt_pval: float
    :return:
    :rtype: Dict[int, Dict[str, torch.Tensor]]
    """

    candidates = find_unimodal_events(patterns, dt_pval)
    events = merge_events(candidates)
    return events


def find_unimodal_events(patterns: torch.Tensor, dt_pval: float) -> torch.Tensor:
    """
    find unimodal events

    :param patterns:
    :type patterns: torch.Tensor
    :param dt_pval:
    :type dt_pval: float
    :return:
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
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    merge events

    :param candidates:
    :type candidates: torch.Tensor
    :return:
    :rtype: Dict[int, Dict[str, torch.Tensor]]
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

    merged: Dict[
        int, Dict[str, torch.Tensor]
    ] = {}  # merge in linear pass over time dimension
    currstart = torch.tensor([-1])
    currend = torch.tensor([-1])
    currevent = torch.ones(T) * -1.0
    eventnum = 1

    for i, s in enumerate(sstart):
        if i == 0:  # first event
            currstart = s
            currend = send[i]
            currevent = sevents[i, :]
        elif s > currend:  # start of new event; record previous one
            merged[eventnum] = {
                "range": torch.stack([currstart, currend]),
                "event": currevent,
            }
            eventnum += 1
            currstart = s
            currend = send[i]
            currevent = sevents[i, :]
        else:  # update current event
            currend = torch.max(torch.stack([currend, send[i]]))
            currevent += sevents[i, :]

    # record final event
    if currstart > 0:
        merged[eventnum] = {
            "range": torch.stack([currstart, currend]),
            "event": currevent,
        }
    return merged


def assign_metrics_to_events(
    events: Dict[int, Dict[str, torch.Tensor]],
    activity_scores: torch.Tensor,
    omp_tol: float,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    use an assignment scheme to map metrics to events.

    :param events:
    :type events: dict[int, dict[str, torch.Tensor]]
    :param activity_scores:
    :type activity_scores: torch.Tensor
    :param omp_tol:
    :type omp_tol: float
    :return:
    :rtype: Dict[int, Dict[str, torch.Tensor]]
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

        events[int(e)]["metrics"] = torch.tensor(mlist)

    return events


def omp_assign(
    events: Dict[int, Dict[str, torch.Tensor]],
    activity_score: torch.Tensor,
    omp_tol: float,
) -> List[int]:
    """
    omp assign

    :param events:
    :type events: dict[int, dict[str, torch.Tensor]]
    :param activity_score:
    :type activity_score: torch.Tensor
    :param omp_tol:
    :type omp_tol: float
    :return:
    :rtype: List[int]
    """

    if torch.var(activity_score, dim=0, unbiased=False) == 0:
        return []  # skip assignment if activity score is flat

    # initialize for OMP loop
    assign: List[int] = []
    assigned_events = torch.empty((len(activity_score), 0))
    cvg = False
    events_left = list(events.keys())
    r = activity_score

    if len(events_left) == 0:  # case where no events were found upstream
        return assign

    while not cvg:
        # get remaining event patterns
        event_mat = torch.stack([events[i]["event"] for i in events_left], dim=1)
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
            cvg = True  # no remaining events sufficiently explain data, so return

    return assign
