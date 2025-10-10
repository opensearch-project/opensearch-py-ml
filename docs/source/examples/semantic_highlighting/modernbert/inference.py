import os
import sys
import json
import time
import logging
import torch
import re
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from typing import List
from api_types import HighlightSpan, InferenceResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# Global model and tokenizer
model = None
tokenizer = None


def model_fn(model_dir):
    """Load ModernBERT model"""
    global model, tokenizer
    
    logger.info("Loading ModernBERT model...")
    
    # Load from model_dir (contains downloaded HuggingFace model)
    model_path = os.path.join(model_dir, "model_files") if os.path.exists(os.path.join(model_dir, "model_files")) else model_dir
    
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model.to(DEVICE)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model


def get_embedding(text):
    """Get sentence embedding from ModernBERT"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding


def split_sentences(text):
    """Split text into sentences with character positions"""
    # Split by periods (both English and Chinese)
    sentences = []
    current_pos = 0
    
    for match in re.finditer(r'[^.。]+[.。]?', text):
        sentence = match.group().strip()
        if sentence:
            start = text.find(sentence, current_pos)
            end = start + len(sentence)
            sentences.append({
                'text': sentence,
                'start': start,
                'end': end
            })
            current_pos = end
    
    return sentences


def _normalize_token(token: str) -> str:
    token = token.lower()
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _tokenize_for_overlap(text: str):
    return [_normalize_token(tok) for tok in re.findall(r"[A-Za-z]+", text)]


def highlight_sentences(question, context, min_score=0.65) -> List[HighlightSpan]:
    """
    Highlight relevant sentences using ModernBERT with normalized scoring.
    
    Returns character-level positions of highlighted sentences.
    """
    # Split context into sentences
    sentences = split_sentences(context)
    
    if not sentences:
        return []
    
    # Get query embedding
    query_emb = get_embedding(question)
    
    # Batch process all sentences at once
    sentence_texts = [s['text'] for s in sentences]
    inputs = tokenizer(sentence_texts, return_tensors="pt", truncation=True, 
                      max_length=8192, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        sentence_embs = outputs.last_hidden_state.mean(dim=1)
    
    # Calculate cosine similarity
    scores = torch.nn.functional.cosine_similarity(
        query_emb.unsqueeze(0),
        sentence_embs,
        dim=-1
    )
    
    min_val = scores.min()
    max_val = scores.max()
    if torch.isclose(max_val, min_val):
        normalized = torch.zeros_like(scores)
    else:
        normalized = (scores - min_val) / (max_val - min_val)

    query_terms = set(_tokenize_for_overlap(question))

    highlights = []
    for i in range(len(sentences)):
        norm_score = normalized[i].item()
        if norm_score < min_score:
            continue

        sentence_terms = set(_tokenize_for_overlap(sentences[i]['text']))
        if query_terms and not query_terms.intersection(sentence_terms):
            continue

        highlights.append({
            'start': sentences[i]['start'],
            'end': sentences[i]['end']
        })

    return [{'start': h['start'], 'end': h['end']} for h in highlights]


def input_fn(request_body: str, request_content_type: str) -> dict:
    """Parse input request"""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(data, model) -> InferenceResponse:
    """Run inference"""
    start_time = time.time()

    # Batch with individual questions (if inputs has content)
    if "inputs" in data and data["inputs"]:
        inputs = data["inputs"]
        all_highlights = []
        
        for item in inputs:
            question = item.get("question", "")
            context = item.get("context", "")
            highlights = highlight_sentences(question, context)
            all_highlights.append(highlights)
        
        processing_time = (time.time() - start_time) * 1000

        response = {
            "highlights": all_highlights,
            "processing_time_ms": round(processing_time, 2),
            "device": str(DEVICE)
        }
    # Single document (if question/context have content)
    elif ("question" in data and data["question"] and
          "context" in data and data["context"] and
          "contexts" not in data):
        # Single inference
        question = data['question']
        context = data['context']
        
        highlights = highlight_sentences(question, context)
        processing_time = (time.time() - start_time) * 1000

        response = {
            "highlights": highlights,
            "processing_time_ms": round(processing_time, 2),
            "device": str(DEVICE)
        }
    else:
        raise ValueError(
            "Invalid input format. Expected one of:\n"
            "1. {'inputs': [{'question': '...', 'context': '...'}, ...]}\n"
            "2. {'question': '...', 'context': '...'}"
        )

    return response


def output_fn(prediction: InferenceResponse, response_content_type: str) -> str:
    """Format output response"""
    if response_content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
