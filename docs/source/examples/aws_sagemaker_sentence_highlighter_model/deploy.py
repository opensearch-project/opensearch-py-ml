import os
import time
import logging
import boto3
import sagemaker
import shutil
import tarfile
import requests
import zipfile
from datetime import datetime
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get configuration from environment variables
INSTANCE_TYPE = os.getenv('INSTANCE_TYPE', 'ml.g4dn.xlarge')

def get_endpoint_name():
    """Generate a unique endpoint name with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"semantic-highlighter-{timestamp}"

def download_and_extract_model(url, extract_dir):
    """Download zip file and extract model"""
    try:
        # Create a temporary directory for the zip file
        temp_dir = "temp_download"
        os.makedirs(temp_dir, exist_ok=True)
        zip_path = os.path.join(temp_dir, "model.zip")
        
        # Download zip file
        logger.info(f"Downloading model zip from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Model zip downloaded successfully")
        
        # Extract zip file
        logger.info(f"Extracting zip file to {extract_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the .pt file in the extracted contents
        pt_files = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.endswith('.pt'):
                    pt_files.append(os.path.join(root, file))
        
        if not pt_files:
            raise FileNotFoundError("No .pt file found in the extracted contents")
        
        model_path = pt_files[0]  # Use the first .pt file found
        logger.info(f"Found model file at: {model_path}")
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        return model_path
        
    except Exception as e:
        logger.error(f"Error downloading and extracting model: {str(e)}")
        raise

def prepare_model_package():
    """Prepare model package for deployment"""
    try:
        logger.info("Preparing model package...")
        model_dir = "model"
        
        # Clean up existing model directory
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        # model from OpenSearch pre-trained model hub
        model_url = "https://artifacts.opensearch.org/models/ml-models/amazon/sentence-highlighting/opensearch-semantic-highlighter-v1/1.0.0/torch_script/sentence-highlighting_opensearch-semantic-highlighter-v1-1.0.0-torch_script.zip"
        
        # Download and extract model
        model_path = download_and_extract_model(model_url, model_dir)
        model_filename = os.path.basename(model_path)
        
        # Create code directory for inference script
        code_dir = os.path.join(model_dir, "code")
        os.makedirs(code_dir, exist_ok=True)
        
        # Copy inference script to code directory
        if not os.path.exists("inference.py"):
            raise FileNotFoundError("inference.py not found")
        shutil.copy("inference.py", code_dir)
        
        # Copy requirements.txt to code directory
        if not os.path.exists("requirements.txt"):
            raise FileNotFoundError("requirements.txt not found")
        shutil.copy("requirements.txt", code_dir)
        
        # Create model.tar.gz
        if os.path.exists("model.tar.gz"):
            os.remove("model.tar.gz")
            
        logger.info("Creating model.tar.gz...")
        with tarfile.open("model.tar.gz", "w:gz") as tar:
            tar.add(model_path, arcname=model_filename)
            tar.add(code_dir, arcname="code")
        
        logger.info("Model package prepared successfully")
        
    except Exception as e:
        logger.error(f"Error preparing model package: {str(e)}")
        raise

def create_sagemaker_role():
    """Create a new SageMaker execution role with necessary permissions."""
    try:
        logger.info("Creating new SageMaker role...")
        iam = boto3.client('iam')
        role_name = 'SageMakerExecutionRole'
        
        # Create the role
        try:
            role = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument='''{
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "sagemaker.amazonaws.com"
                            },
                            "Action": "sts:AssumeRole"
                        }
                    ]
                }'''
            )
            logger.info(f"Created new role: {role_name}")
        except iam.exceptions.EntityAlreadyExistsException:
            logger.info(f"Role {role_name} already exists")
            return f'arn:aws:iam::{boto3.client("sts").get_caller_identity()["Account"]}:role/{role_name}'
        
        # Attach necessary policies
        policies = [
            'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess'
        ]
        
        for policy in policies:
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy
            )
            logger.info(f"Attached policy {policy} to role {role_name}")
        
        # Get the role ARN
        role_arn = f'arn:aws:iam::{boto3.client("sts").get_caller_identity()["Account"]}:role/{role_name}'
        logger.info(f"Created role with ARN: {role_arn}")
        return role_arn
        
    except Exception as e:
        logger.error(f"Failed to create SageMaker role: {str(e)}")
        raise

def deploy_model():
    try:
        # Prepare model package first
        prepare_model_package()
        
        # Generate unique endpoint name
        endpoint_name = get_endpoint_name()
        logger.info(f"Using endpoint name: {endpoint_name}")
        
        # Initialize SageMaker session
        logger.info("Initializing SageMaker session...")
        session = sagemaker.Session()
        
        # Create and get execution role
        role = create_sagemaker_role()
        logger.info(f"Using execution role: {role}")
        
        # Get default bucket (creates it if it doesn't exist)
        bucket = session.default_bucket()
        logger.info(f"Using default SageMaker bucket: {bucket}")
        
        # Upload model to S3
        s3_prefix = 'semantic-highlighter'
        logger.info(f"Uploading model to S3: {bucket}/{s3_prefix}")
        
        try:
            s3_path = session.upload_data('model.tar.gz', bucket=bucket, key_prefix=s3_prefix)
            logger.info(f"Model uploaded successfully to {s3_path}")
        except Exception as e:
            logger.error(f"Failed to upload model to S3: {str(e)}")
            raise
        
        # Create SageMaker model
        logger.info("Creating SageMaker model...")
        model = PyTorchModel(
            model_data=s3_path,
            role=role,
            framework_version='2.5',
            py_version='py311',
            entry_point='inference.py',
            source_dir='.',
            sagemaker_session=session
        )
        logger.info("SageMaker model created successfully")
        
        # Deploy endpoint
        logger.info("Starting endpoint deployment...")
        logger.info("This may take several minutes...")
        start_time = time.time()
        
        try:
            predictor = model.deploy(
                initial_instance_count=1,
                instance_type=INSTANCE_TYPE,
                endpoint_name=endpoint_name,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer()
            )
            
            end_time = time.time()
            logger.info(f"Endpoint deployed successfully in {end_time - start_time:.2f} seconds")
            logger.info(f"Endpoint name: {endpoint_name}")
            
            return predictor
        except Exception as e:
            logger.error(f"Deployment failed with error: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    deploy_model()
