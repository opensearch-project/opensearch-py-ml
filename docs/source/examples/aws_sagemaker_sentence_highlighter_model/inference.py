import json
import os
import logging
import nltk
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_nltk_data():
    """Ensure NLTK data (punkt) is available for sentence tokenization."""
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK punkt tokenizer data found.")
    except LookupError:
        logger.info("NLTK punkt data not found. Downloading...")
        # When running on SageMaker, /tmp/ is a writable directory.
        nltk_data_path = '/tmp/nltk_data'
        if not os.path.exists(nltk_data_path):
            os.makedirs(nltk_data_path)
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
        nltk.data.path.append(nltk_data_path)
        logger.info(f"NLTK punkt data downloaded and added to path: {nltk_data_path}")

def model_fn(model_dir):
    """
    Load the model for inference.
    
    Args:
        model_dir: Directory containing model artifacts
    """
    logger.info("Loading model...")
    
    # Load model and tokenizer
    model_path = os.path.join(model_dir, "model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return {"model": model, "tokenizer": tokenizer}

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.
    
    Args:
        request_body: The request body
        request_content_type: The content type of the request
    """
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """
    Apply model to the input data.
    
    Args:
        input_data: The input data
        model_dict: Dictionary containing model and tokenizer
    """
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    
    # Prepare inputs
    text = input_data["text"]
    question = input_data.get("question", "")
    
    # Tokenize
    inputs = tokenizer(
        text,
        question,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.sigmoid(outputs.logits)
    
    # Convert to list for JSON serialization
    scores = scores.cpu().numpy().tolist()
    
    return {"scores": scores}

def output_fn(prediction_output, response_content_type):
    """
    Serialize and prepare the prediction output.
    
    Args:
        prediction_output: The prediction output
        response_content_type: The content type of the response
    """
    if response_content_type == "application/json":
        return json.dumps(prediction_output)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")

# Example for local testing (not used by SageMaker directly but helpful for dev)
if __name__ == '__main__':
    # Create a dummy model_dir with a dummy model for local testing structure
    if not os.path.exists("dummy_model_dir/code"):
        os.makedirs("dummy_model_dir/code")
    # Create a minimal dummy TorchScript model if one doesn't exist
    dummy_model_file = "dummy_model_dir/opensearch-semantic-highlighter-v1.pt"
    if not os.path.exists(dummy_model_file):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super(DummyModel, self).__init__()
                self.linear = torch.nn.Linear(10, 1) # Dummy layer
            def forward(self, input_ids, attention_mask, token_type_ids, sentence_ids):
                # This dummy model needs to return something plausible, e.g., an empty list or a fixed index
                # For testing, let's say it highlights the first sentence index found (0)
                # This is highly dependent on what your actual model returns
                # and how `sentence_ids` is structured and used.
                logger.info("DummyModel forward called")
                return torch.tensor([0]) # Example: highlight sentence 0
        scripted_model = torch.jit.script(DummyModel())
        torch.jit.save(scripted_model, dummy_model_file)
        logger.info(f"Created dummy TorchScript model at {dummy_model_file}")

    try:
        logger.info("--- Local Test: Loading Model ---")
        model_artifacts_local = model_fn("dummy_model_dir")
        logger.info("Model loaded locally.")

        logger.info("--- Local Test: Input Processing ---")
        sample_request_body = json.dumps({
            "question": "What is OpenSearch?",
            "context": "OpenSearch is a community-driven, open source search and analytics suite. It consists of a search engine daemon, OpenSearch, and a visualization and user interface, OpenSearch Dashboards."
        })
        input_data_local = input_fn(sample_request_body, 'application/json')
        logger.info(f"Input processed locally: {input_data_local}")

        logger.info("--- Local Test: Prediction ---")
        prediction_local = predict_fn(input_data_local, model_artifacts_local)
        logger.info(f"Prediction result locally: {prediction_local}")

        logger.info("--- Local Test: Output Formatting ---")
        output_json_local = output_fn(prediction_local, 'application/json')
        logger.info(f"Formatted output locally: {output_json_local}")
        print("\nFinal Local Test Output JSON:\n", output_json_local)

    except Exception as e:
        logger.error(f"Local test failed: {e}", exc_info=True)
    finally:
        # Clean up dummy model directory
        if os.path.exists("dummy_model_dir"):
            shutil.rmtree("dummy_model_dir")
            logger.info("Cleaned up dummy_model_dir.")
