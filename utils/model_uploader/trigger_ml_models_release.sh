# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

# This program is run by "Model Auto-tracing & Uploading" 
# & "Model Listing Uploading" workflow (See model_uploader.yml 
# & model_listing_uploader.yml) to trigger ml-models-release 
# Jenkins workflow.


JENKINS_URL="https://build.ci.opensearch.org"
JENKINS_TRIGGER_TOKEN=""
JENKINS_PARAMS=""
GITHUB_USER=""
GITHUB_TOKEN=""

while getopts "u:d:t:p:" opt; do
  case $opt in
    t)
      JENKINS_TRIGGER_TOKEN="$OPTARG"
      ;;
    d)
      JENKINS_PARAMS="$OPTARG"
      ;;  
    u)
      GITHUB_USER="$OPTARG"
      ;;
    p)
      GITHUB_TOKEN="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [ -z "$JENKINS_TRIGGER_TOKEN" ]; then
  echo "Error: TRIGGER_TOKEN is required. Use -t option to provide it."
  exit 1
fi

if [ -z "$JENKINS_PARAMS" ]; then
  echo "Error: JENKINS_PARAMS is required. Use -d option to provide it."
  exit 1
fi

if [ -z "$GITHUB_USER" ]; then
  echo "Error: GITHUB_USER is required. Use -u option to provide it."
  exit 1
fi

if [ -z "$GITHUB_TOKEN" ]; then
  echo "Error: GITHUB_TOKEN is required. Use -p option to provide it."
  exit 1
fi

TIMEPASS=0
TIMEOUT=3600
RESULT="null"

JENKINS_REQ=$(curl -s -XPOST \
             -H "Authorization: Bearer $JENKINS_TRIGGER_TOKEN" \
             -H "Content-Type: application/json" \
             "$JENKINS_URL/generic-webhook-trigger/invoke" \
             --data "$JENKINS_PARAMS")

echo "Trigger ml-models-release Jenkins workflows"
echo $JENKINS_PARAMS
echo $JENKINS_REQ

QUEUE_URL=$(echo $JENKINS_REQ | jq --raw-output '.jobs."ml-models-release".url')
echo "QUEUE_URL: $QUEUE_URL"
echo "Wait for jenkins to start workflow" && sleep 15

echo "Check if queue exist in Jenkins after triggering"
if [ -z "$QUEUE_URL" ] || [ "$QUEUE_URL" != "null" ]; then
    WORKFLOW_URL=$(curl -s -XGET ${JENKINS_URL}/${QUEUE_URL}api/json --user ${GITHUB_USER}:${GITHUB_TOKEN} | jq --raw-output .executable.url)
    echo "WORKFLOW_URL: $WORKFLOW_URL"
    echo "Use queue information to find build number in Jenkins if available"
    if [ -z "$WORKFLOW_URL" ] || [ "$WORKFLOW_URL" != "null" ]; then
        RUNNING="true"
        echo "Waiting for Jenkins to complete the run"
        while [ "$RUNNING" = "true" ] && [ "$TIMEPASS" -le "$TIMEOUT" ]; do
            echo "Still running, wait for another 5 seconds before checking again, max timeout $TIMEOUT"
            echo "Jenkins Workflow URL: $WORKFLOW_URL"
            TIMEPASS=$(( TIMEPASS + 5 )) && echo time passed: $TIMEPASS
            sleep 5
            RUNNING=$(curl -s -XGET ${WORKFLOW_URL}api/json --user ${GITHUB_USER}:${GITHUB_TOKEN} | jq --raw-output .building)
        done

        if [ "$RUNNING" = "true" ]; then
            echo "Timed out"
            RESULT="TIMEOUT"
        else
            echo "Completed the run, checking the results now......"
            RESULT=$(curl -s -XGET ${WORKFLOW_URL}api/json --user ${GITHUB_USER}:${GITHUB_TOKEN} | jq --raw-output .result)
        fi
    fi
fi

echo "Please check jenkins url for logs: $WORKFLOW_URL"
echo "Result: $RESULT"
if [ "$RESULT" != "SUCCESS" ]; then
    exit 1
fi
