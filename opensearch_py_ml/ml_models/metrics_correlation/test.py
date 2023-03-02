# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import mcorr
import torch

jit_processing = torch.jit.script(mcorr.MCorr())

# reasonable-size synthetic data
T = 20
M = 2
event_times = [64]
# simdat = torch.FloatTensor(np.stack(synth_data.sim_multiple_events(T, M, event_times, noise=True).values()))
#
# print(simdat)
# simdat = torch.FloatTensor(np.stack(synth_data.sim_multiple_events(T, M, event_times, noise=True).values()))

# print(jit_processing.forward(simdat, 5))
print(
    jit_processing.forward(
        torch.Tensor(
            [
                [0, 0, 0, 1, 2, 3, 3, 3, 2, 1, 0, 0, 0],
                [0, 0, 0, 10, 10, 15, 15, 15, 12, 10, 0, 0, 0],
                [0, 0, 0, 2, 4, 6, 6, 6, 4, 2, 0, 0, 0],
            ]
        )
    )
)

print(jit_processing.code)

# torch.jit.save(jit_processing, "nmf_mcorr.pt")
