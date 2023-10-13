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

import pytest
import time
import os
from opensearchpy import OpenSearch, helpers
from sklearn.datasets import load_iris

@pytest.fixture
def iris_index_client(opensearch_client: OpenSearch):
    index_name = "test__index__iris_data"
    index_mapping = {
        "mappings": {
            "properties": {
                "sepal_length": {"type": "float"},
                "sepal_width": {"type": "float"},
                "petal_length": {"type": "float"},
                "petal_width": {"type": "float"},
                "species": {"type": "keyword"}
            }
        }
    }
    
    if opensearch_client.indices.exists(index=index_name):
        opensearch_client.indices.delete(index=index_name)
    opensearch_client.indices.create(index=index_name, body=index_mapping)

    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target
    iris_species = [iris.target_names[i] for i in iris_target]

    actions = [
        {   '_index': index_name,
            "_source":{
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width,
                "species": species
            }
        }
        for (sepal_length, sepal_width, petal_length, petal_width), species in zip(iris_data, iris_species)
    ]

    helpers.bulk(opensearch_client, actions)
    # without the sleep, test is failing.
    time.sleep(2)

    yield opensearch_client, index_name

    opensearch_client.indices.delete(index=index_name)