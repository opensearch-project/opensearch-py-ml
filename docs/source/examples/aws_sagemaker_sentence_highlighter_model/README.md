# Sentence Highlighting Model Deployment

This directory contains scripts for deploying the OpenSearch Sentence Highlighting model to AWS SageMaker. The deployment process automatically downloads the model from OpenSearch artifacts, packages it, and deploys it as a SageMaker endpoint.

## Prerequisites

1. AWS Account with appropriate permissions:
   - **IAM Permissions**:
     - `iam:CreateRole`
     - `iam:GetRole`
     - `iam:AttachRolePolicy`
   - **SageMaker Permissions**:
     - `sagemaker:CreateModel`
     - `sagemaker:CreateEndpoint`
     - `sagemaker:CreateEndpointConfig`
   - **S3 Permissions**:
     - `s3:PutObject`
     - `s3:GetObject`
     - `s3:CreateBucket` (for default bucket creation)

2. AWS credentials configured locally:
   ```bash
   # Configure using AWS CLI
   aws configure
   ```

3. Python 3.10 or higher
4. Required Python packages (installed via requirements.txt)

## Files

**Project Path**: `docs/source/examples/aws_sagemaker_sentence_highlighter_model/`

- `deploy.py`: Main deployment script
- `inference.py`: Model inference code for SageMaker
- `requirements.txt`: Python package dependencies

## Quick Start

1. Navigate to the deployment directory:
   ```bash
   cd docs/source/examples/aws_sagemaker_sentence_highlighter_model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the deployment script:
   ```bash
   python deploy.py
   ```

The script will:
1. Download the model from OpenSearch artifacts
2. Package the model with necessary dependencies
3. Create a SageMaker role if it doesn't exist
4. Upload the model to the default SageMaker S3 bucket
5. Deploy the model to a SageMaker endpoint

## API Usage

Once deployed, the model endpoint accepts POST requests with the following format:

### Input Format

```json
{
    "question": "What is the main topic discussed?",
    "context": "This is a long text document containing multiple sentences. The model will identify which sentences are most relevant to answering the question. It processes the entire context and returns the sentences that best help answer the provided question."
}
```

**Required fields:**
- `question` (string): The question you want to find relevant sentences for
- `context` (string): The text document containing multiple sentences to search through

### Output Format

```json
{
    "highlights": [
        {
            "start": 45,
            "end": 123,
            "text": "This sentence is relevant to the question.",
            "position": 2
        },
        {
            "start": 200,
            "end": 285,
            "text": "Another relevant sentence that helps answer the question.",
            "position": 5
        }
    ]
}
```

**Response fields:**
- `highlights` (array): List of highlighted sentences
  - `start` (integer): Character position where the sentence starts in the original context
  - `end` (integer): Character position where the sentence ends in the original context
  - `text` (string): The actual text of the highlighted sentence
  - `position` (integer): The sentence number in the original context (0-indexed)

### Example Usage

```python
import boto3
import json

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

# Prepare the input
payload = {
    "question": "What are the benefits of machine learning?",
    "context": "Machine learning is a powerful technology. It can help automate many tasks. The benefits include improved efficiency and accuracy. However, it requires good data quality. Machine learning models can make predictions on new data."
}

# Make the prediction
response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='application/json',
    Body=json.dumps(payload)
)

# Parse the response
result = json.loads(response['Body'].read().decode())
print(result)
```

## Environment Variables

You can customize the deployment using these environment variables:

### Optional Variables:
- `INSTANCE_TYPE`: SageMaker instance type (default: "ml.g5.xlarge")
  - **CPU instances**: "ml.m5.xlarge", "ml.m5.2xlarge", "ml.c5.xlarge"
  - **GPU instances**: "ml.g4dn.xlarge", "ml.g4dn.2xlarge", "ml.p3.2xlarge"
  - Example: `export INSTANCE_TYPE="ml.g4dn.xlarge"`

- `AWS_PROFILE`: AWS credentials profile (default: "default")
  - Example: `export AWS_PROFILE="my-profile"`

- `AWS_REGION`: AWS region (default: from AWS configuration)
  - Example: `export AWS_REGION="us-west-2"`

## GPU Deployment

To deploy with GPU acceleration for faster inference:

### Deploy with GPU:
```bash
# Set GPU instance type
export INSTANCE_TYPE="ml.g5.xlarge"

# Deploy
python deploy.py
```

### GPU Verification:
The inference logs will show GPU usage:
```
Environment Information:
CUDA Available: True
CUDA Device Name: NVIDIA T4
Using GPU for inference
Model device detected: cuda:0 - Moving tensors to this device for inference
```

Example configuration:
```bash
# Optional: Set custom instance type and region
export AWS_REGION="us-west-2"
export INSTANCE_TYPE="ml.g5.xlarge"

# Then run the deployment script
python deploy.py
```

Note: The script uses the default SageMaker bucket for your account (`sagemaker-{region}-{account_id}`), which is automatically created if it doesn't exist.

## Monitoring

The script provides detailed logging of the deployment process. Check the logs for:
- Download progress
- Packaging status
- S3 upload status
- Deployment progress
- Endpoint creation status

## Cleanup

To avoid unnecessary charges, delete the endpoint when not needed:
```python
import boto3
sagemaker = boto3.client('sagemaker')
with open('endpoint_name.txt', 'r') as f:
    endpoint_name = f.read().strip()
sagemaker.delete_endpoint(EndpointName=endpoint_name)
```

## Troubleshooting

Common issues and solutions:

1. **AWS Credentials**: Ensure AWS credentials are properly configured
2. **Memory Issues**: If packaging fails, ensure sufficient disk space
3. **Network Issues**: Check network connection if download fails
4. **Permission Issues**: Verify AWS IAM permissions

## Support

For issues and questions:
- Check OpenSearch documentation
- Submit issues to the OpenSearch GitHub repository
- Contact the OpenSearch community 