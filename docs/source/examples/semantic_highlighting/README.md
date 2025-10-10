# Semantic Highlighting Remote Models Deployment

This directory contains deployment script examples for semantic highlighting models on AWS SageMaker.

## Models

- **opensearch-semantic-highlighter**: OpenSearch semantic highlighter v1 model, see huggingface [here](https://huggingface.co/opensearch-project/opensearch-semantic-highlighter-v1)
- **modernbert**: ModernBERT-base model for semantic highlighting, see huggingface [here](https://huggingface.co/answerdotai/ModernBERT-base)

## Usage

### Prerequisites

Install deployment dependencies:
```bash
pip install -r requirements.txt
```

### Deploy Models

Deploy ModernBERT model:
```bash
python deploy.py --model modernbert
```

Deploy OpenSearch Semantic Highlighter:
```bash
python deploy.py --model opensearch-semantic-highlighter
```

### Custom Configuration

Specify instance type and count:
```bash
python deploy.py --model modernbert --instance-type ml.g5.2xlarge --instance-count 2
```

### Environment Variables

- `INSTANCE_TYPE`: Default SageMaker instance type (default: ml.g5.xlarge)
- `INSTANCE_COUNT`: Number of instances (default: 1)
- `AWS_REGION`: AWS region (default: us-east-1)

## Inference API

### Request Format

Both models accept JSON requests with the following structure:

```json
{
  "question": "What is the treatment for Alzheimer's disease?",
  "context": "Alzheimers disease is a progressive neurodegenerative disorder. Traditional treatments include cholinesterase inhibitors and memantine. Recent clinical trials investigating monoclonal antibodies have shown promise."
}
```

### Response Format

Both models return the same format:

```json
{
  "highlights": [
    {"start": 85, "end": 165},
    {"start": 166, "end": 280}
  ],
  "processing_time_ms": 22.4,
  "device": "cuda"
}
```
