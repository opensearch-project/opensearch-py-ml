import os
import sys
import json
import time
import logging
import torch
import numpy as np
import nltk
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from typing import List
from api_types import HighlightSpan, InferenceResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

def ensure_nltk_data():
    """Ensure NLTK data is available"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        logger.info("NLTK data already available")
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            logger.info("Successfully downloaded NLTK data")
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {str(e)}")
            raise

ensure_nltk_data()

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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model


def get_embedding(text):
    """Get pooled embedding for a single text snippet"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding


def split_sentences(text):
    """Split text into sentences with character positions using NLTK"""
    doc_sents = nltk.sent_tokenize(text)
    
    sentences = []
    current_pos = 0
    
    for sentence in doc_sents:
        start = text.find(sentence, current_pos)
        if start != -1:
            end = start + len(sentence)
            sentences.append({
                'text': sentence,
                'start': start,
                'end': end
            })
            current_pos = end
    
    return sentences


def extract_context_embeddings(sentences, context_embeddings, offset_mapping):
    """Extract context-aware embeddings for sentences using offset mapping"""
    context_embs = []
    for sent in sentences:
        sent_start, sent_end = sent['start'], sent['end']
        token_mask = (
            (offset_mapping[:, 0] >= sent_start)
            & (offset_mapping[:, 1] <= sent_end)
            & (offset_mapping[:, 1] > offset_mapping[:, 0])
        )

        if token_mask.sum() > 0:
            context_embs.append(context_embeddings[token_mask].mean(dim=0))
        else:
            context_embs.append(torch.zeros_like(context_embeddings[0]))

    return torch.stack(context_embs)


def compute_highlights(question_emb, sentence_embs, context_embs, sentences, min_score=0.8, top_k_ratio=0.15):
    """Compute highlights from embeddings"""
    combined_embs = 0.7 * sentence_embs + 0.3 * context_embs

    scores = torch.nn.functional.cosine_similarity(
        question_emb.unsqueeze(0),
        combined_embs,
        dim=-1
    )

    scores_np = scores.detach().cpu().numpy()
    k = max(1, int(len(scores_np) * top_k_ratio))
    topk_floor = np.partition(scores_np, -k)[-k]
    threshold = max(min_score, topk_floor)

    highlights = []
    for i, score in enumerate(scores_np):
        if score >= threshold:
            highlights.append({
                'start': sentences[i]['start'],
                'end': sentences[i]['end']
            })

    return highlights


def highlight_sentences(
    question,
    context,
    min_score: float = 0.8,
    top_k_ratio: float = 0.15,
) -> List[HighlightSpan]:
    """
    Highlight relevant sentences using ModernBERT with context-aware encoding.
    
    Returns character-level positions of highlighted sentences.
    """
    sentences = split_sentences(context)
    
    if not sentences:
        return []
    
    # Get query embedding
    query_emb = get_embedding(question)
    
    # Encode sentences independently to retain absolute similarity magnitudes
    sentence_texts = [sent['text'] for sent in sentences]
    sent_inputs = tokenizer(
        sentence_texts,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        sent_outputs = model(**sent_inputs)
        sentence_embs = sent_outputs.last_hidden_state.mean(dim=1)

    # Encode full context once for relative cues
    ctx_inputs = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
        return_offsets_mapping=True,
    ).to(DEVICE)

    offset_mapping = ctx_inputs.pop("offset_mapping")[0].cpu()
    with torch.no_grad():
        ctx_outputs = model(**ctx_inputs)
        full_embeddings = ctx_outputs.last_hidden_state[0]

    context_embs = extract_context_embeddings(sentences, full_embeddings, offset_mapping)
    
    return compute_highlights(query_emb, sentence_embs, context_embs, sentences, min_score, top_k_ratio)


def highlight_sentences_batch(
    questions: List[str],
    contexts: List[str],
    min_score: float = 0.8,
    top_k_ratio: float = 0.15,
) -> List[List[HighlightSpan]]:
    """
    Batch process multiple documents with GPU parallelization.
    
    Returns list of highlights for each document.
    """
    # Step 1: Batch encode all questions
    question_inputs = tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192
    ).to(DEVICE)
    
    with torch.no_grad():
        question_outputs = model(**question_inputs)
        question_embs = question_outputs.last_hidden_state.mean(dim=1)
    
    # Step 2: Prepare all sentences from all documents
    all_doc_data = []
    all_sentences_flat = []
    doc_sentence_counts = []
    
    for context in contexts:
        sentences = split_sentences(context)
        all_doc_data.append({'context': context, 'sentences': sentences})
        all_sentences_flat.extend([s['text'] for s in sentences])
        doc_sentence_counts.append(len(sentences))
    
    # Step 3: Batch encode all sentences from all documents
    if all_sentences_flat:
        sent_inputs = tokenizer(
            all_sentences_flat,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        ).to(DEVICE)
        
        with torch.no_grad():
            sent_outputs = model(**sent_inputs)
            all_sentence_embs = sent_outputs.last_hidden_state.mean(dim=1)
    else:
        all_sentence_embs = torch.tensor([])
    
    # Step 4: Batch encode all full contexts
    ctx_inputs = tokenizer(
        contexts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192,
        return_offsets_mapping=True
    ).to(DEVICE)
    
    offset_mappings = ctx_inputs.pop("offset_mapping").cpu()
    with torch.no_grad():
        ctx_outputs = model(**ctx_inputs)
        all_context_embs = ctx_outputs.last_hidden_state
    
    # Step 5: Compute highlights for each document
    all_highlights = []
    sent_idx = 0
    
    for doc_idx, doc_data in enumerate(all_doc_data):
        num_sents = doc_sentence_counts[doc_idx]
        
        if num_sents == 0:
            all_highlights.append([])
            continue
        
        # Extract embeddings for this document
        sentence_embs = all_sentence_embs[sent_idx:sent_idx + num_sents]
        context_embs = extract_context_embeddings(
            doc_data['sentences'],
            all_context_embs[doc_idx],
            offset_mappings[doc_idx]
        )
        
        # Compute highlights
        highlights = compute_highlights(
            question_embs[doc_idx],
            sentence_embs,
            context_embs,
            doc_data['sentences'],
            min_score,
            top_k_ratio
        )
        
        all_highlights.append(highlights)
        sent_idx += num_sents
    
    return all_highlights


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
        questions = [item.get("question", "") for item in inputs]
        contexts = [item.get("context", "") for item in inputs]
        
        # True batch processing with GPU parallelization
        all_highlights = highlight_sentences_batch(questions, contexts)
        
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
