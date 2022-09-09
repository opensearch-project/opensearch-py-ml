## About

`opensearch-py-ml` is a Python client that provides a suite of data analytics and machine learning tools for OpenSearch.
It is a fork of [eland](https://github.com/elastic/eland), which provides data analysis and machine learning
support for Elasticsearch.

`opensearch-py-ml` lets users call OpenSearch indices and manipulate them as if they were pandas DataFrames, supporting
complex filtering and aggregation operations. It also provides rudimentary support for uploading models to OpenSearch
clusters using the [ml-commons](https://github.com/opensearch-project/ml-commons) plugin, and provides integration with
AWS SageMaker, allowing users to upload OpenSearch indices to deployed SageMaker endpoints for real-time prediction.

Project hand-off doc: https://quip-amazon.com/XAIMAu2XK3Ph/opensearch-py-ml-project-handoff-doc

## Testing `opensearch-py-ml`

This package relies on a *minimal-security* version of OpenSearch 2.2 running on a connection to localhost:9200, with a
dev fork of `ml-commons` installed on the OpenSearch cluster: https://github.com/LEFTA98/ml-commons/tree/opensearch-2.2

Once this is up and running, run the `setup_tests.py` file, then run the pytests in whatever manner you would prefer to
run unit tests.

## To-dos

This is an early proof-of-concept missing many features, and a lot of work still needs to be done to bring this in-line
with a package like eland. Many files in the `full_fork` branch have been deleted from this branch, because they are
code that is still adapted to `eland` and has not yet been changed. List of things to do:

- Write integration tests
- Rewriting `LICENSE.txt`, `CHANGELOG.rst`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and this `README` to reflect OpenSearch
- Support for Docker
- Support for continuous integration
- Regenerating Sphinx docs
- Creating tutorials for `opensearch-py-ml` in both notebook and video form
=======
AWS SageMaker, allowing users to upload OpenSearch indices to deployed SageMaker endpoints for real-time prediction.
