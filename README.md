## About

`opensearch-py-ml` is a Python client that provides a suite of data analytics and machine learning tools for OpenSearch.
It is a fork of [eland](https://github.com/elastic/eland), which provides data analysis and machine learning
support for Elasticsearch.

`opensearch-py-ml` lets users call OpenSearch indices and manipulate them as if they were pandas DataFrames, supporting
complex filtering and aggregation operations. It also provides rudimentary support for uploading models to OpenSearch
clusters using the [ml-commons](https://github.com/opensearch-project/ml-commons) plugin, and provides integration with
AWS SageMaker, allowing users to upload OpenSearch indices to deployed SageMaker endpoints for real-time prediction.