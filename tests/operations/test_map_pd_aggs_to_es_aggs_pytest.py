# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

from opensearch_py_ml.utils import (
    MEAN_ABSOLUTE_DEVIATION,
    STANDARD_DEVIATION,
    VARIANCE,
)
from opensearch_py_ml.operations import Operations


def test_all_aggs():
    os_aggs = Operations._map_pd_aggs_to_os_aggs(
        [
            "min",
            "max",
            "mean",
            STANDARD_DEVIATION,
            VARIANCE,
            MEAN_ABSOLUTE_DEVIATION,
            "count",
            "nunique",
            "median",
            "quantile",
        ],
        percentiles=[0.2, 0.5, 0.8],
    )

    assert os_aggs == [
        ("extended_stats", "min"),
        ("extended_stats", "max"),
        ("extended_stats", "avg"),
        ("extended_stats", "std_deviation"),
        ("extended_stats", "variance"),
        "median_absolute_deviation",
        "value_count",
        "cardinality",
        ("percentiles", (50.0,)),
        (
            "percentiles",
            (
                0.2,
                0.5,
                0.8,
            ),
        ),
    ]


def test_extended_stats_optimization():
    # Tests that when '<agg>' and an 'extended_stats' agg are used together
    # that ('extended_stats', '<agg>') is used instead of '<agg>'.
    os_aggs = Operations._map_pd_aggs_to_os_aggs(["count", "nunique"])
    assert os_aggs == ["value_count", "cardinality"]

    for pd_agg in [VARIANCE, STANDARD_DEVIATION]:
        extended_os_agg = Operations._map_pd_aggs_to_os_aggs([pd_agg])[0]

        os_aggs = Operations._map_pd_aggs_to_os_aggs([pd_agg, "nunique"])
        assert os_aggs == [extended_os_agg, "cardinality"]

        os_aggs = Operations._map_pd_aggs_to_os_aggs(["count", pd_agg, "nunique"])
        assert os_aggs == ["value_count", extended_os_agg, "cardinality"]


def test_percentiles_none():
    os_aggs = Operations._map_pd_aggs_to_os_aggs(["count", "min", "quantile"])

    assert os_aggs == ["value_count", "min", ("percentiles", (50.0,))]
