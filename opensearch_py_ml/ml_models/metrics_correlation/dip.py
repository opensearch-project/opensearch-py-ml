# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from typing import Tuple

import torch


def dip(x: torch.Tensor) -> float:
    """
    Fast computation of the dip test statistic. See: https://www.jstor.org/stable/pdf/2241144.pdf (paper)
    and https://www.jstor.org/stable/pdf/2347485.pdf (pseudocode)
    Debugged and validated against the R package of Prof. Martin Maechler, ETH Zurich:
    https://rdrr.io/cran/diptest/

    :param x:
    :type x: torch.Tensor
    :return:
    :rtype: float
    """

    x = torch.sort(x).values  # just assume unsorted

    if x[0] == x[-1]:
        return (
            0.0  # constant is technically unimodal, but we want to reject these events
        )

    n = len(x)
    low = 0
    high = n - 1

    # establish indices mn over which combination is req's for GCM fit
    mn = torch.zeros(n)
    for j in range(1, n):
        mn[j] = j - 1
        while True:
            mnj = int(mn[j].item())
            mnmnj = int(mn[mnj].item())
            a = float(mnj - mnmnj)
            b = float(j - mnj)
            if mnj == 0 or (x[j] - x[mnj]) * a < (x[mnj] - x[mnmnj]) * b:
                break
            mn[j] = mnmnj

    # establish indices mj over which combination is req's for LCM fit
    mj = torch.zeros(n)
    mj[n - 1] = n - 1
    for k in range(n - 2, -1, -1):
        mj[k] = k + 1
        while True:
            mjk = int(mj[k].item())
            mjmjk = int(mj[mjk].item())
            a = float(mjk - mjmjk)
            b = float(k - mjk)
            if mjk == n - 1 or (x[k] - x[mjk]) * a < (x[mjk] - x[mjmjk]) * b:
                break
            mj[k] = mjmjk

    #
    # Start the cycling
    #
    dstar = 1.0
    while True:
        # collect GCM change points
        gcm = [high]
        i = 0
        while gcm[i] > low:
            gcm.append(int(mn[gcm[i]]))
            i += 1
        ig = i
        l_gcm = i
        ix = ig - 1

        gcm.append(0)

        # collect LCM change points
        lcm = [low]
        i = 0
        while lcm[i] < high:
            lcm.append(int(mj[lcm[i]]))
            i += 1
        ih = i
        l_lcm = i
        iv = 1

        lcm.append(0)

        # compute dx
        d = 0.0
        if l_lcm != 2 or l_gcm != 2:
            while True:
                # compute dx
                gcmix = gcm[ix]
                lcmiv = lcm[iv]

                if gcmix > lcmiv:
                    gcmi1 = gcm[ix + 1]
                    dx = (lcmiv - gcmi1 + 1) - (x[lcmiv] - x[gcmi1]) * (
                        gcmix - gcmi1
                    ) / (x[gcmix] - x[gcmi1])
                    iv += 1
                    if dx >= d:
                        d = dx
                        ig = ix + 1
                        ih = iv - 1
                else:
                    lcmiv1 = lcm[iv - 1]
                    dx = (x[gcmix] - x[lcmiv1]) * (lcmiv - lcmiv1) / (
                        x[lcmiv] - x[lcmiv1]
                    ) - (gcmix - lcmiv1 - 1)
                    ix -= 1
                    if dx >= d:
                        d = dx
                        ig = ix + 1
                        ih = iv

                if ix < 0:
                    ix = 0
                if iv > l_lcm:
                    iv = l_lcm

                if gcm[ix] == lcm[iv]:
                    break
        else:
            d = 1.0

        if d < dstar:
            break

        # compute dip for cvx minorant
        dip_l = 0.0
        for j in range(ig, l_gcm):
            max_t = 1.0
            jb = gcm[j + 1]
            je = gcm[j]
            if je - jb > 1 and x[je] != x[jb]:
                C = torch.log(torch.tensor(je - jb)) - torch.log(
                    x[je] - x[jb]
                )  # log space for numeric stability
                for jjj in range(jb, je + 1):
                    xx = torch.log(x[jjj] - x[jb])
                    t = (jjj - jb + 1) - torch.exp(xx + C)
                    if max_t < t:
                        max_t = t

            if dip_l < max_t:
                dip_l = max_t

        # compute dip for cnc majorant
        dip_u = 0.0
        for j in range(ih, l_lcm):
            max_t = 1.0
            jb = lcm[j]
            je = lcm[j + 1]
            if je - jb > 1 and x[je] != x[jb]:
                C = torch.log(torch.tensor(je - jb)) - torch.log(
                    x[je] - x[jb]
                )  # log space for numeric stability
                for jjj in range(jb, je + 1):
                    xx = torch.log(x[jjj] - x[jb])
                    t = torch.exp(xx + C) - (jjj - jb - 1)
                    if max_t < t:
                        max_t = t

            if dip_u < max_t:
                dip_u = max_t

        # print('dip_l, dip_u = ({},{})'.format(dip_l, dip_u))

        # update current max dip
        if dip_u > dip_l:
            dipnew = dip_u
        else:
            dipnew = dip_l

        if dstar < dipnew:
            dstar = dipnew

        if low == gcm[ig] and high == lcm[ih]:
            # print('no improvement in low / high -> end')
            break
        else:
            low = gcm[ig]
            high = lcm[ih]

    # end logic
    dstar = dstar / (2 * n)

    return dstar  # mn, mj, gcm, lcm, dstar


def diptest(x: torch.Tensor) -> Tuple[float, float]:
    """
    Testing the dip

    :param x:
    :type x: torch.Tensor
    :return:
    :rtype: Tuple[float, float]
    """

    # torchScript doesn't support tensor values as global variable so we can't put these
    # constant values in utils.py

    N_VALS: torch.Tensor = torch.tensor(
        [100, 200, 500, 700, 1000, 2000, 5000, 7000, 10000, 20000, 50000]
    )

    P_VALS: torch.Tensor = torch.tensor(
        [0.01, 0.05, 0.1, 0.4, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995]
    )

    CRIT_VALS: torch.Tensor = torch.tensor(
        [
            [
                0.02274363,
                0.02554336,
                0.027407,
                0.03354036,
                0.03543797,
                0.04132709,
                0.04705229,
                0.05068449,
                0.05893375,
                0.06286501,
            ],
            [
                0.0164563,
                0.01849594,
                0.0197459,
                0.02423924,
                0.02556999,
                0.02949423,
                0.03375041,
                0.0364896,
                0.04232255,
                0.04458542,
            ],
            [
                0.01074585,
                0.01199022,
                0.01276506,
                0.01560221,
                0.01645565,
                0.01905045,
                0.02185605,
                0.0237823,
                0.02756447,
                0.02871819,
            ],
            [
                0.00898803,
                0.01011504,
                0.01083064,
                0.01324822,
                0.01399304,
                0.01624022,
                0.01859142,
                0.02021879,
                0.02353281,
                0.02476247,
            ],
            [
                0.00761489,
                0.00856395,
                0.00915069,
                0.01115789,
                0.01175641,
                0.01367194,
                0.01555408,
                0.01687476,
                0.01977919,
                0.02083874,
            ],
            [
                0.0053385,
                0.00607165,
                0.00645562,
                0.00788072,
                0.00832072,
                0.00965699,
                0.01112998,
                0.01212624,
                0.01416395,
                0.01507873,
            ],
            [
                0.00346535,
                0.00388841,
                0.00413141,
                0.00503782,
                0.00531415,
                0.00617108,
                0.00703597,
                0.0076569,
                0.00886423,
                0.00926628,
            ],
            [
                0.00287961,
                0.00325313,
                0.00348236,
                0.00426688,
                0.00449341,
                0.00521093,
                0.00598266,
                0.00652404,
                0.00755206,
                0.00792809,
            ],
            [
                0.00243663,
                0.00272667,
                0.00291825,
                0.00356491,
                0.00376222,
                0.00435947,
                0.00498846,
                0.00541417,
                0.006259,
                0.00656881,
            ],
            [
                0.00171876,
                0.00193273,
                0.00206471,
                0.00252897,
                0.00267032,
                0.00309158,
                0.00354587,
                0.00384297,
                0.00443779,
                0.00471282,
            ],
            [
                0.00109227,
                0.00124015,
                0.00131565,
                0.00159665,
                0.00168592,
                0.00195405,
                0.00224709,
                0.00242406,
                0.00276954,
                0.00293705,
            ],
        ]
    )

    n = len(x)
    D = float(dip(x))

    i1 = int(torch.searchsorted(N_VALS, n))
    i0 = int(i1 - 1)

    # if n falls outside range of n_vals, select endpoint instead
    if i0 < 0:
        i0 = 0
    if i1 > (len(N_VALS) - 1):
        i1 = len(N_VALS) - 1

    n0 = N_VALS[i0]
    n1 = N_VALS[i1]
    if n0 != n1:
        fn = 1.0 * (n - n0) / (n1 - n0)
    else:
        fn = torch.tensor([0.0])  # catch div-by-zero in endpoint case
    sn = torch.sqrt(torch.tensor([n]))
    y0 = sn * CRIT_VALS[i0]
    y1 = sn * CRIT_VALS[i1]
    sD = sn * D

    p = 1.0 - interp(sD, y0 + fn * (y1 - y0), P_VALS)
    return D, p


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> float:
    """
    Intercept

    :param x:
    :type x: torch.Tensor
    :param xp:
    :type xp: torch.Tensor
    :param fp:
    :type fp: torch.Tensor
    :return:
    :rtype: float
    """

    if torch.all(x < xp):  # constant interpolation outside sD range
        return float(fp[0])
    elif torch.all(x > xp):
        return float(fp[-1])
    else:
        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])

        i1 = torch.searchsorted(xp, x) - 1
        i1 = torch.clamp(i1, 0, len(m) - 1)

        return float(m[i1] * x + b[i1])
