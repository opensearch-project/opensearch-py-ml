#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <endpoint-name>"
    exit 1
fi

ENDPOINT_NAME=$1
REGION="us-east-1"

# Test payloads for embedding models
QUERY_PAYLOAD='{
    "texts": ["how much protein should a female eat"],
    "content_type": "query"
}'

PASSAGE_PAYLOAD='{
    "texts": ["As a general guideline, the CDC'\''s average requirement of protein for women ages 19 to 70 is 46 grams per day."],
    "content_type": "passage"
}'

BATCH_PAYLOAD='{
    "texts": ["how much protein should a female eat", "what are the benefits of exercise"],
    "content_type": "query"
}'

# OpenSearch connector format
CONNECTOR_PAYLOAD='{
    "parameters": {
        "texts": ["sample query text"],
        "content_type": "query"
    }
}'

echo "Testing embedding endpoint: $ENDPOINT_NAME"

# Test query request
echo "Testing query request..."
echo "$QUERY_PAYLOAD" > query_payload.json
aws sagemaker-runtime invoke-endpoint \
    --region $REGION \
    --endpoint-name $ENDPOINT_NAME \
    --content-type application/json \
    --body fileb://query_payload.json \
    query_response.json

# Test passage request  
echo "Testing passage request..."
echo "$PASSAGE_PAYLOAD" > passage_payload.json
aws sagemaker-runtime invoke-endpoint \
    --region $REGION \
    --endpoint-name $ENDPOINT_NAME \
    --content-type application/json \
    --body fileb://passage_payload.json \
    passage_response.json

# Test batch request
echo "Testing batch request..."
echo "$BATCH_PAYLOAD" > batch_payload.json
aws sagemaker-runtime invoke-endpoint \
    --region $REGION \
    --endpoint-name $ENDPOINT_NAME \
    --content-type application/json \
    --body fileb://batch_payload.json \
    batch_response.json

# Test OpenSearch connector format
echo "Testing OpenSearch connector format..."
echo "$CONNECTOR_PAYLOAD" > connector_payload.json
aws sagemaker-runtime invoke-endpoint \
    --region $REGION \
    --endpoint-name $ENDPOINT_NAME \
    --content-type application/json \
    --body fileb://connector_payload.json \
    connector_response.json

if [ $? -eq 0 ]; then
    echo "✓ All requests successful!"
    echo "✓ Query response saved to query_response.json"
    echo "✓ Passage response saved to passage_response.json"
    echo "✓ Batch response saved to batch_response.json"
    echo "✓ Connector response saved to connector_response.json"
    
    # Show response sizes and sample embeddings
    QUERY_SIZE=$(wc -c < query_response.json)
    PASSAGE_SIZE=$(wc -c < passage_response.json)
    BATCH_SIZE=$(wc -c < batch_response.json)
    CONNECTOR_SIZE=$(wc -c < connector_response.json)
    
    echo "✓ Query response size: $QUERY_SIZE bytes"
    echo "✓ Passage response size: $PASSAGE_SIZE bytes"
    echo "✓ Batch response size: $BATCH_SIZE bytes"
    echo "✓ Connector response size: $CONNECTOR_SIZE bytes"
    
    # Show first few embedding values
    echo "✓ Sample query embedding (first 5 values):"
    head -c 100 query_response.json | jq '.[0:5]' 2>/dev/null || echo "   (raw response preview)"
    
    # Clean up payload files only, keep response files
    rm -f query_payload.json passage_payload.json batch_payload.json connector_payload.json
else
    echo "✗ Endpoint invocation failed!"
    exit 1
fi
