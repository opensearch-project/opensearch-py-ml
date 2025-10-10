#!/usr/bin/env python3
"""
Deployment script for semantic highlighting models
"""

import os
import shutil
import tarfile
import boto3
import sagemaker
import json
import time
import logging
import tempfile
import uuid
import argparse
from datetime import datetime
from sagemaker.pytorch import PyTorchModel
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    'opensearch-semantic-highlighter': {
        'model_name': 'opensearch-project/opensearch-semantic-highlighter-v1',
        'endpoint_prefix': 'opensearch-semantic-highlighter',
        's3_prefix': 'opensearch-semantic-highlighter'
    },
    'modernbert': {
        'model_name': 'answerdotai/ModernBERT-base',
        'endpoint_prefix': 'modernbert-highlighter',
        's3_prefix': 'modernbert-highlighter'
    }
}

# Configuration from environment variables
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.g5.xlarge")
INSTANCE_COUNT = int(os.environ.get("INSTANCE_COUNT", "1"))
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Generate deployment identifiers once for consistency
DEPLOYMENT_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
DEPLOYMENT_TIMESTAMP_UNDERSCORE = datetime.now().strftime("%Y%m%d_%H%M%S")
DEPLOYMENT_UUID = uuid.uuid4().hex[:8]
DEPLOYMENT_ID = f"{DEPLOYMENT_TIMESTAMP}-{DEPLOYMENT_UUID}"

def create_sagemaker_role():
    """Create SageMaker execution role with necessary permissions if it doesn't exist."""
    try:
        iam = boto3.client('iam')
        sts = boto3.client('sts')
        account_id = sts.get_caller_identity()["Account"]
        role_name = 'SageMakerExecutionRole'
        role_arn = f'arn:aws:iam::{account_id}:role/{role_name}'
        
        try:
            iam.get_role(RoleName=role_name)
            logger.info(f"Using existing role: {role_name}")
            return role_arn
        except iam.exceptions.NoSuchEntityException:
            logger.info(f"Creating new role: {role_name}")
            iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps({
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "sagemaker.amazonaws.com"},
                            "Action": "sts:AssumeRole"
                        }
                    ]
                })
            )
            
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
            )
            
            logger.info(f"Created role: {role_name}")
            return role_arn
            
    except Exception as e:
        logger.error(f"Error creating SageMaker role: {e}")
        raise

def prepare_model_files(model_key):
    """Download and prepare model files based on model configuration."""
    config = MODEL_CONFIGS[model_key]
    model_name = config['model_name']
    
    work_dir = tempfile.mkdtemp()
    os.makedirs(f"{work_dir}/model_files", exist_ok=True)
    
    try:
        logger.info(f"Using model: {model_name}")
        logger.info("Downloading model from HuggingFace...")
        
        safetensors_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        
    except Exception as e:
        logger.error(f"Failed to download model files from {model_name}: {e}")
        raise RuntimeError(f"Model download failed: {e}")
    
    # Copy safetensors directly
    shutil.copy(safetensors_path, f"{work_dir}/model_files/model.safetensors")
    logger.info("Copied model files (safetensors format)")
    
    shutil.copy(config_path, f"{work_dir}/model_files/config.json")
    
    return work_dir

def create_model_tar(work_dir, model_key):
    """Create model.tar.gz with model files and inference code."""
    os.makedirs(f"{work_dir}/code", exist_ok=True)
    
    # Copy model-specific inference code
    inference_src = f"{model_key}/inference.py"
    requirements_src = f"{model_key}/requirements.txt"
    api_types_src = "api_types.py"
    
    if not os.path.exists(inference_src):
        raise FileNotFoundError(f"Inference code not found: {inference_src}")
    
    shutil.copy(inference_src, f"{work_dir}/code/inference.py")
    if os.path.exists(requirements_src):
        shutil.copy(requirements_src, f"{work_dir}/code/requirements.txt")
    
    # Copy shared API types
    if os.path.exists(api_types_src):
        shutil.copy(api_types_src, f"{work_dir}/code/api_types.py")
    
    model_path = f"{work_dir}/model.tar.gz"
    with tarfile.open(model_path, "w:gz") as tar:
        tar.add(f"{work_dir}/model_files", arcname=".")
        tar.add(f"{work_dir}/code", arcname="code")
    
    logger.info(f"Created model archive: {model_path}")
    return model_path

def deploy_model(model_path, model_key, instance_type, instance_count):
    """Deploy model to SageMaker endpoint."""
    config = MODEL_CONFIGS[model_key]
    
    role_arn = create_sagemaker_role()
    sess = sagemaker.Session()
    
    logger.info("Uploading to S3...")
    s3_key_prefix = f"{config['s3_prefix']}/{DEPLOYMENT_TIMESTAMP}"
    s3_path = sess.upload_data(model_path, key_prefix=s3_key_prefix)
    logger.info(f"Model uploaded to: {s3_path}")
    
    model = PyTorchModel(
        model_data=s3_path,
        role=role_arn,
        entry_point="inference.py",
        framework_version="2.6",
        py_version="py312",
        sagemaker_session=sess,
    )
    
    endpoint_name = f"{config['endpoint_prefix']}-{DEPLOYMENT_ID}"
    
    print(f"Deploying endpoint: {endpoint_name}")
    print(f"Using instance type: {instance_type}")
    print(f"Instance count: {instance_count}")
    
    predictor = model.deploy(
        initial_instance_count=instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    
    print(f"Endpoint deployed successfully: {endpoint_name}")
    print(f"Endpoint URL: {predictor.endpoint_name}")
    
    return predictor

def main():
    parser = argparse.ArgumentParser(description='Deploy semantic highlighting models')
    parser.add_argument('--model', required=True, choices=list(MODEL_CONFIGS.keys()),
                       help='Model to deploy')
    parser.add_argument('--instance-type', default=INSTANCE_TYPE,
                       help='SageMaker instance type')
    parser.add_argument('--instance-count', type=int, default=INSTANCE_COUNT,
                       help='Number of instances')
    
    args = parser.parse_args()
    
    # Use args directly instead of modifying globals
    instance_type = args.instance_type
    instance_count = args.instance_count
    
    try:
        logger.info(f"Starting deployment for model: {args.model}")
        
        # Prepare model files
        work_dir = prepare_model_files(args.model)
        
        # Create model archive
        model_path = create_model_tar(work_dir, args.model)
        
        # Deploy to SageMaker
        predictor = deploy_model(model_path, args.model, instance_type, instance_count)
        
        # Cleanup
        shutil.rmtree(work_dir)
        
        logger.info("Deployment completed successfully!")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise

if __name__ == "__main__":
    main()
