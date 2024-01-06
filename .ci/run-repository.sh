#!/usr/bin/env bash

# Called by entry point `run-test` use this script to add your repository specific task commands
# Once called opensearch is up and running and the following parameters are available to this script

# OPENSEARCH_VERSION -- version e.g Major.Minor.Patch(-Prelease)
# OPENSEARCH_URL -- The url at which opensearch is reachable
# network_name -- The docker network name
# NODE_NAME -- The docker container name also used as opensearch node name

# When run in CI the test-matrix is used to define additional variables
# TEST_SUITE -- defaults to `oss` in `run-tests`

set -e

echo -e "\033[34;1mINFO:\033[0m URL ${opensearch_url}\033[0m"
echo -e "\033[34;1mINFO:\033[0m EXTERNAL OS URL ${external_opensearch_url}\033[0m"
echo -e "\033[34;1mINFO:\033[0m VERSION ${OPENSEARCH_VERSION}\033[0m"
echo -e "\033[34;1mINFO:\033[0m TASK_TYPE: ${TASK_TYPE}\033[0m"
echo -e "\033[34;1mINFO:\033[0m TEST_SUITE ${TEST_SUITE}\033[0m"
echo -e "\033[34;1mINFO:\033[0m PYTHON_VERSION ${PYTHON_VERSION}\033[0m"
echo -e "\033[34;1mINFO:\033[0m PYTHON_CONNECTION_CLASS ${PYTHON_CONNECTION_CLASS}\033[0m"
echo -e "\033[34;1mINFO:\033[0m PANDAS_VERSION ${PANDAS_VERSION}\033[0m"

echo -e "\033[1m>>>>> Build [opensearch-project/opensearch-py-ml container] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>\033[0m"

docker build \
       --file .ci/Dockerfile.client \
       --tag opensearch-project/opensearch-py-ml \
       --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
       .

echo -e "\033[1m>>>>> Run [opensearch-project/opensearch-py-ml container] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>\033[0m"


if [[ "$TASK_TYPE" == "test" ]]; then
  # Set up OpenSearch cluster & Run integration and unit tests (Invoked by integration.yml workflow)
  docker run \
  --network=${network_name} \
  --env "STACK_VERSION=${STACK_VERSION}" \
  --env "OPENSEARCH_URL=${opensearch_url}" \
  --env "OPENSEARCH_VERSION=${OPENSEARCH_VERSION}" \
  --env "TEST_SUITE=${TEST_SUITE}" \
  --env "PYTHON_CONNECTION_CLASS=${PYTHON_CONNECTION_CLASS}" \
  --env "TEST_TYPE=server" \
  --name opensearch-py-ml-test-runner \
  opensearch-project/opensearch-py-ml \
  nox -s "test-${PYTHON_VERSION}"
  
  docker cp opensearch-py-ml-test-runner:/code/opensearch-py-ml/junit/ ./junit/
  docker rm opensearch-py-ml-test-runner
elif [[ "$TASK_TYPE" == "doc" ]]; then
  # Set up OpenSearch cluster & Run docs (Invoked by build_deploy_doc.yml workflow)
  docker run \
  --network=${network_name} \
  --env "STACK_VERSION=${STACK_VERSION}" \
  --env "OPENSEARCH_URL=${opensearch_url}" \
  --env "OPENSEARCH_VERSION=${OPENSEARCH_VERSION}" \
  --env "TEST_SUITE=${TEST_SUITE}" \
  --env "PYTHON_CONNECTION_CLASS=${PYTHON_CONNECTION_CLASS}" \
  --env "TEST_TYPE=server" \
  --name opensearch-py-ml-doc-runner \
  opensearch-project/opensearch-py-ml \
  nox -s "docs-${PYTHON_VERSION}"
  
  docker cp opensearch-py-ml-doc-runner:/code/opensearch-py-ml/docs/build/ ./docs/
  docker rm opensearch-py-ml-doc-runner
elif [[ "$TASK_TYPE" == "trace" ]]; then
  # Set up OpenSearch cluster & Run model autotracing (Invoked by model_uploader.yml workflow)
  echo -e "\033[34;1mINFO:\033[0m MODEL_ID: ${MODEL_ID}\033[0m"
  echo -e "\033[34;1mINFO:\033[0m MODEL_VERSION: ${MODEL_VERSION}\033[0m"
  echo -e "\033[34;1mINFO:\033[0m TRACING_FORMAT: ${TRACING_FORMAT}\033[0m"
  echo -e "\033[34;1mINFO:\033[0m EMBEDDING_DIMENSION: ${EMBEDDING_DIMENSION:-N/A}\033[0m"
  echo -e "\033[34;1mINFO:\033[0m POOLING_MODE: ${POOLING_MODE:-N/A}\033[0m"
  echo -e "\033[34;1mINFO:\033[0m MODEL_DESCRIPTION: ${MODEL_DESCRIPTION:-N/A}\033[0m"

  docker run \
  --network=${network_name} \
  --env "STACK_VERSION=${STACK_VERSION}" \
  --env "OPENSEARCH_URL=${opensearch_url}" \
  --env "OPENSEARCH_VERSION=${OPENSEARCH_VERSION}" \
  --env "TEST_SUITE=${TEST_SUITE}" \
  --env "PYTHON_CONNECTION_CLASS=${PYTHON_CONNECTION_CLASS}" \
  --env "TEST_TYPE=server" \
  --name opensearch-py-ml-trace-runner \
  opensearch-project/opensearch-py-ml \
  nox -s "trace-${PYTHON_VERSION}" -- ${MODEL_ID} ${MODEL_VERSION} ${TRACING_FORMAT} -ed ${EMBEDDING_DIMENSION} -pm ${POOLING_MODE} -md ${MODEL_DESCRIPTION:+"$MODEL_DESCRIPTION"}
  
  docker cp opensearch-py-ml-trace-runner:/code/opensearch-py-ml/upload/ ./upload/
  docker cp opensearch-py-ml-trace-runner:/code/opensearch-py-ml/trace_output/ ./trace_output/
  docker rm opensearch-py-ml-trace-runner
fi
