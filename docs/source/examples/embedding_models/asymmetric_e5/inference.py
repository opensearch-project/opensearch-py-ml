import os
import sys
import json
import time
import logging
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from api_types import EmbeddingRequest, BatchEmbeddingRequest, EmbeddingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

def model_fn(model_dir):
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(DEVICE)
    return {"model": model, "tokenizer": tokenizer}

def input_fn(request_body, request_content_type):
    """Parse input and return texts for embedding"""
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")
    
    input_data = json.loads(request_body)
    
    # Handle OpenSearch connector format
    if "parameters" in input_data:
        params = input_data["parameters"]
        texts = params.get("texts", [])
        content_type = params.get("content_type")
    else:
        texts = input_data.get("texts", [])
        content_type = input_data.get("content_type")
    
    # Add content type prefix if specified
    if content_type:
        texts = [f"{content_type}: {text}" for text in texts]
    
    return texts

def predict_fn(input_data, model_dict):
    """Generate embeddings"""
    start_time = time.time()
    
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    
    inputs = tokenizer(input_data, padding=True, truncation=True, 
                      return_tensors="pt", max_length=512).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "embeddings": embeddings.cpu().numpy(),
        "processing_time_ms": processing_time,
        "device": str(DEVICE)
    }

def output_fn(prediction, content_type):
    """Format output for OpenSearch compatibility"""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    
    embeddings = prediction["embeddings"]
    
    # Return simple array format for OpenSearch
    if len(embeddings.shape) == 2:  # Batch
        result = [embedding.tolist() for embedding in embeddings]
    else:  # Single
        result = embeddings.tolist()
    
    return json.dumps(result)
