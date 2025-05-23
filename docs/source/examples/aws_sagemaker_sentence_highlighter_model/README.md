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

## Environment Variables

You can customize the deployment using these environment variables:

### Optional Variables:
- `INSTANCE_TYPE`: SageMaker instance type (default: "ml.g4dn.xlarge")
  - Example: `export INSTANCE_TYPE="ml.g4dn.xlarge"`

- `AWS_PROFILE`: AWS credentials profile (default: "default")
  - Example: `export AWS_PROFILE="my-profile"`

- `AWS_REGION`: AWS region (default: from AWS configuration)
  - Example: `export AWS_REGION="us-west-2"`

Example configuration:
```bash
# Optional: Set custom instance type and region
export AWS_REGION="us-west-2"
export INSTANCE_TYPE="ml.g4dn.xlarge"

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