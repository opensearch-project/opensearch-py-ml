# Common Model Deployment

Shared deployment infrastructure for SageMaker endpoints across different model types.

## Structure

```
examples/
├── common/
│   ├── deploy.py              # Shared deployment script
│   └── README.md             # This file
├── semantic_highlighting/     # Highlighting models
│   ├── api_types.py          # Highlighting-specific types
│   ├── modernbert/
│   └── opensearch-semantic-highlighter/
└── embedding_models/          # Embedding models
    ├── api_types.py          # Embedding-specific types
    ├── validate.sh           # Validation script
    └── asymmetric_e5/
```

## Usage

### Deploy a Model

```bash
cd common
python3 deploy.py --model <model_name> [options]
```

The deployment script will:
1. Download the model from HuggingFace
2. Create a model package with inference code
3. Deploy to SageMaker endpoint
4. **Output the endpoint name** for validation

### Validate Deployment

After deployment, you'll see output like:
```
Endpoint deployed successfully: asymmetric-e5-20251113-210834-866f6617
```

Use this endpoint name to validate:

```bash
# For embedding models ONLY
cd ../embedding_models
./validate.sh asymmetric-e5-20251113-210834-866f6617

# For semantic highlighting models
# (no validation script yet - test manually via AWS console or CLI)
```

**Why specify endpoint name?**
- SageMaker generates unique endpoint names with timestamps
- Multiple deployments can exist simultaneously  
- Allows testing specific endpoint versions
- Prevents accidental validation of wrong endpoints

**Note:** The `validate.sh` script is specifically designed for embedding models and tests embedding-specific payloads (query/passage embeddings, OpenSearch connector format). Semantic highlighting models require different validation payloads.

### Available Models

**Semantic Highlighting:**
- `opensearch-semantic-highlighter`
- `modernbert`

**Embedding Models:**
- `asymmetric_e5`

### Options

- `--model`: Model to deploy (required)
- `--instance-type`: SageMaker instance type (default: ml.g5.xlarge)
- `--instance-count`: Number of instances (default: 1)

### Examples

```bash
# Deploy asymmetric E5 embedding model
python3 deploy.py --model asymmetric_e5 --instance-type ml.m5.large

# Deploy semantic highlighter
python3 deploy.py --model opensearch-semantic-highlighter
```

## Environment Variables

- `AWS_REGION`: AWS region (default: us-east-1)
- `INSTANCE_TYPE`: Default instance type
- `INSTANCE_COUNT`: Default instance count

## Finding Existing Endpoints

To list existing endpoints:
```bash
aws sagemaker list-endpoints --region us-east-1
```

## API Formats

### Embedding Models (asymmetric_e5)

**Request:**
```json
{
    "texts": ["how much protein should a female eat"],
    "content_type": "query"
}
```

**Response:**
```json
[[0.21125227, -0.19419950, ...]]
```

### Semantic Highlighting Models

**Request:**
```json
{
    "question": "What is the treatment?",
    "context": "Traditional treatments include cholinesterase inhibitors."
}
```

**Response:**
```json
{
    "highlights": [{"start": 0, "end": 50}],
    "processing_time_ms": 22.4,
    "device": "cuda"
}
```

## Adding New Models

1. Create model directory under appropriate task type
2. Add `inference.py` and `requirements.txt`
3. Update `MODEL_CONFIGS` in `deploy.py`
4. Ensure proper `api_types.py` exists for the task type
