# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

from typing import Tuple

import torch


class NMF(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def step(
        self, V: torch.Tensor, W: torch.Tensor, H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multiplicative update; see https://arxiv.org/pdf/1010.1763.pdf
        Variable definitions here are consistent with notation in this paper.

        Implementation follows sklearn 'mu' type updates, see here:
        https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/decomposition/_nmf.py#L540-L742

        Note that these proceed in W, H order, as opposed to H, W in the paper.

        :param V: V is a data metrics of dimensions F × N with non-negative entries
        :type V: torch.Tensor
        :param W: W is a data metrics of dimensions F × K with non-negative entries
        :type W: torch.Tensor
        :param H: H is a data metrics of dimensions K × N with non-negative entries
        :type H: torch.Tensor
        :return:
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        # W update
        VHt = torch.mm(V, H.t())
        WHHt = torch.mm(W, torch.mm(H, H.t()))
        WHHt[WHHt == 0] = 1.2e-7
        W = torch.mul(W, torch.div(VHt, WHHt))  # note these ops are element-wise

        # H update
        WtV = torch.mm(W.t(), V)
        WtWH = torch.mm(torch.mm(W.t(), W), H)
        WtWH[WtWH == 0] = 1.2e-7
        H = torch.mul(H, torch.div(WtV, WtWH))

        return W, H

    def initialize(self, V, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        'NNDSVD-A' initialization, as in sklearn.

        For nonnegative SVD initialization see https://www.cb.uu.se/~milan/histo/before2011august/Boutsidis.pdf

        NNDSVD-A first computes NNDSVD as above, then fills any zeros in W or H with the mean of V.

        :param V:
        :type V: torch.Tensor
        :param k:
        :type k: int
        :return:
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        U, S, Vh = torch.linalg.svd(V, full_matrices=False)

        d = U.shape[1]  # factors are identified up to sign. make sure they're positive
        for dd in range(d):
            if torch.any(U[:, dd] < 0):
                U[:, dd] *= -1
                Vh[dd, :] *= -1

        k = int(k)

        f, n = V.shape

        W = torch.zeros((f, k))
        H = torch.zeros((k, n))

        W[:, 0] = torch.sqrt(S[0]) * U[:, 0]
        H[0, :] = torch.sqrt(S[0]) * Vh[0, :]

        for j in range(1, k):
            xp = torch.nn.functional.relu(U[:, j])
            xn = torch.nn.functional.relu(-U[:, j])
            yp = torch.nn.functional.relu(Vh[j, :])
            yn = torch.nn.functional.relu(-Vh[j, :])

            mp = torch.norm(xp) * torch.norm(yp)
            mn = torch.norm(xn) * torch.norm(yn)
            if mp > mn:
                u = xp / torch.norm(xp)
                v = yp / torch.norm(yp)
                sig = mp
            else:
                u = xn / torch.norm(xn)
                v = yn / torch.norm(yn)
                sig = mn

            W[:, j] = torch.sqrt(S[j] * sig) * u
            H[j, :] = torch.sqrt(S[j] * sig) * v

        W[W == 0] = V.mean()
        H[H == 0] = V.mean()

        return W, H

    def forward(
        self, V: torch.Tensor, k: int, max_iter: int, tol: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param V:
        :type V: torch.Tensor
        :param k:
        :type k: int
        :param max_iter:
        :type max_iter: int
        :param tol:
        :type tol: float
        :return:
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        W, H = self.initialize(V, k)  # NNDSVD-A init
        init_err = torch.linalg.norm(V - torch.mm(W, H))
        prev_err = init_err

        for i in range(1, max_iter + 1):
            W, H = self.step(V, W, H)

            if i % 10 == 0:  # check rel err stopping criterion
                err = torch.linalg.norm(V - torch.mm(W, H))
                if (prev_err - err) / init_err < tol:
                    break
                prev_err = err

            # TODO: convergence warning for case i == max_iter:

        return W, H
