import os
import sys
import json
import time
import logging
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    BertModel
)
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from functools import partial
from datasets import Dataset
import nltk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from api_types import HighlightSpan, InferenceResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 256
MAX_LENGTH = 510
STRIDE = 128
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


class BertTaggerForSentenceExtractionWithBackoff(BertPreTrainedModel):
    """Sentence-level BERT classifier with confidence-backoff rule"""
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sentence_ids=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = self.dropout(outputs[0])
        
        def _get_agg_output(ids, seq_out):
            """Aggregate token embeddings to sentence level"""
            max_sentences = torch.max(ids) + 1
            d_model = seq_out.size(-1)
            
            agg_out, global_offsets, num_sents = [], [], []
            for i, sen_ids in enumerate(ids):
                out, local_ids = [], sen_ids.clone()
                mask = local_ids != -100
                offset = local_ids[mask].min()
                global_offsets.append(offset)
                local_ids[mask] -= offset
                n_sent = local_ids.max() + 1
                num_sents.append(n_sent)
                
                for j in range(int(n_sent)):
                    out.append(seq_out[i, local_ids == j].mean(dim=-2, keepdim=True))
                
                if max_sentences - n_sent:
                    padding = torch.zeros(
                        (int(max_sentences - n_sent), d_model), device=seq_out.device
                    )
                    out.append(padding)
                agg_out.append(torch.cat(out, dim=0))
            return torch.stack(agg_out), global_offsets, num_sents
        
        agg_output, offsets, num_sents_item = _get_agg_output(sentence_ids, sequence_output)
        
        logits = self.classifier(agg_output)
        probs = torch.softmax(logits, dim=-1)[:, :, 1]
        
        def _get_preds(pp, offs, num_s, threshold=0.5, alpha=0.05):
            """Apply threshold and backoff rule"""
            preds = []
            for p, off, ns in zip(pp, offs, num_s):
                if ns == 0:
                    preds.append(torch.tensor([], dtype=torch.long))
                    continue
                    
                rel_probs = p[:ns]
                hits = (rel_probs >= threshold).int()
                if hits.sum() == 0 and rel_probs.max().item() >= alpha:
                    hits[rel_probs.argmax()] = 1
                preds.append(torch.where(hits == 1)[0] + off)
            return preds
        
        return tuple(_get_preds(probs, offsets, num_sents_item))

_model = None
_tokenizer = None


def model_fn(model_dir):
    """Load model and tokenizer"""
    global _model, _tokenizer

    logger.info(f"Loading model from: {model_dir}")

    _tokenizer = AutoTokenizer.from_pretrained(
        "opensearch-project/opensearch-semantic-highlighter-v1",
        use_fast=True
    )

    model_path = os.path.join(model_dir, "model_files") if os.path.exists(os.path.join(model_dir, "model_files")) else model_dir

    logger.info(f"Loading model from: {model_path}")
    
    # Use from_pretrained to automatically handle both safetensors and pytorch_model.bin
    _model = BertTaggerForSentenceExtractionWithBackoff.from_pretrained(
        model_path,
        local_files_only=True
    )

    _model.to(DEVICE)
    _model.eval()

    logger.info("Model loading completed")
    return _model


def process_single_document(
    question: str,
    context: str,
    model: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Process a single document"""
    doc_sents = nltk.sent_tokenize(context)

    if not doc_sents:
        return {"highlights": [], "error": "No sentences found"}

    total_length = len(question.split()) + len(context.split())

    if total_length <= MAX_LENGTH:
        result = _process_single_chunk(
            question, context, doc_sents, model, device, threshold, alpha
        )
    else:
        result = _process_with_chunks(
            question, context, doc_sents, model, device, threshold, alpha
        )

    return result


def _process_single_chunk(
    question: str,
    context: str,
    doc_sents: List[str],
    model: nn.Module,
    device: torch.device,
    threshold: float,
    alpha: float
) -> Dict[str, Any]:
    """Process document in a single chunk"""

    sentence_ids = []
    words = []
    for sid, sent in enumerate(doc_sents):
        sent_words = sent.split()
        words.extend(sent_words)
        sentence_ids.extend([sid] * len(sent_words))

    example_dataset = Dataset.from_dict({
        "question": [[question]],
        "context": [words],
        "word_level_sentence_ids": [sentence_ids],
        "id": [0],
    })

    example_dataset = example_dataset.map(
        partial(
            prepare_input_features,
            _tokenizer,
            max_seq_length=MAX_LENGTH,
            stride=STRIDE,
        ),
        batched=True,
        remove_columns=example_dataset.column_names,
    )

    example = example_dataset[0]
    features = DataCollatorWithPadding(
        pad_kvs={
            "input_ids": 0,
            "token_type_ids": 0,
            "attention_mask": 0,
            "sentence_ids": -100,
        }
    )([example])

    features = {k: v.to(device) for k, v in features.items()}

    with torch.no_grad():
        sentence_preds = model(**features)

    highlights = []
    highlighted_sentences = []

    sentence_positions = []
    current_pos = 0
    for sent in doc_sents:
        start_pos = context.find(sent, current_pos)
        if start_pos == -1:
            start_pos = current_pos
        end_pos = start_pos + len(sent)
        sentence_positions.append((start_pos, end_pos))
        current_pos = end_pos

    if sentence_preds and len(sentence_preds[0]) > 0:
        for pred_idx in sentence_preds[0].cpu().tolist():
            if 0 <= pred_idx < len(doc_sents):
                highlighted_sentences.append(doc_sents[pred_idx])
                start_pos, end_pos = sentence_positions[pred_idx]
                highlights.append({
                    "text": doc_sents[pred_idx],
                    "start": start_pos,
                    "end": end_pos,
                    "position": pred_idx,
                    "score": 1.0
                })

    return {
        "highlights": highlights,
        "highlighted_sentences": highlighted_sentences,
        "num_chunks": 1,
    }


def _process_with_chunks(
    question: str,
    context: str,
    doc_sents: List[str],
    model: nn.Module,
    device: torch.device,
    threshold: float,
    alpha: float
) -> Dict[str, Any]:
    """Process document with multiple chunks"""

    # Pre-compute sentence positions once
    sentence_positions = []
    current_pos = 0
    for sent in doc_sents:
        start_pos = context.find(sent, current_pos)
        if start_pos == -1:
            start_pos = current_pos
        end_pos = start_pos + len(sent)
        sentence_positions.append((start_pos, end_pos))
        current_pos = end_pos

    chunks = []
    current_chunk = []
    current_length = len(question.split())

    for sent in doc_sents:
        sent_length = len(sent.split())
        if current_length + sent_length > MAX_LENGTH - 50 and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [current_chunk[-1]] if len(current_chunk) > 1 else []
            current_length = len(question.split()) + len(current_chunk[0].split()) if current_chunk else len(question.split())
        current_chunk.append(sent)
        current_length += sent_length

    if current_chunk:
        chunks.append(current_chunk)
    
    all_highlights = []
    seen_sentences = set()
    
    for chunk_idx, chunk_sents in enumerate(chunks):
        chunk_context = " ".join(chunk_sents)
        result = _process_single_chunk(
            question, chunk_context, chunk_sents, model, device, threshold, alpha
        )
        
        for highlight in result.get("highlights", []):
            sent_text = highlight["text"]
            if sent_text not in seen_sentences:
                seen_sentences.add(sent_text)
                for orig_idx, orig_sent in enumerate(doc_sents):
                    if orig_sent == sent_text:
                        start_pos, end_pos = sentence_positions[orig_idx]
                        all_highlights.append({
                            "text": orig_sent,
                            "start": start_pos,
                            "end": end_pos,
                            "position": orig_idx,
                            "score": 1.0
                        })
                        break
    
    return {
        "highlights": sorted(all_highlights, key=lambda x: x["position"]),
        "highlighted_sentences": [h["text"] for h in sorted(all_highlights, key=lambda x: x["position"])],
        "num_chunks": len(chunks)
    }


def process_batch_documents(
    questions: List[str],
    contexts: List[str],
    model: nn.Module,
    device: torch.device,
    batch_size: int = MAX_BATCH_SIZE
) -> List[Dict[str, Any]]:
    assert len(questions) == len(contexts), "Questions and contexts must align"
    total_docs = len(questions)
    results: List[Dict[str, Any]] = []

    # Separate short and long documents
    short_docs = []
    long_docs = []

    for i, (q, c) in enumerate(zip(questions, contexts)):
        total_length = len(q.split()) + len(c.split())
        if total_length <= MAX_LENGTH:
            short_docs.append((i, q, c))
        else:
            long_docs.append((i, q, c))

    # Process long documents individually (they need chunking)
    long_results = {}
    for orig_idx, q, c in long_docs:
        result = process_single_document(q, c, model, device)
        long_results[orig_idx] = result

    # Process short documents in batches
    short_results = {}
    if short_docs:
        short_results = _process_short_docs_batched(short_docs, model, device, batch_size)

    # Combine results in original order
    for i in range(total_docs):
        if i in short_results:
            results.append(short_results[i])
        elif i in long_results:
            results.append(long_results[i])
        else:
            results.append({"highlights": [], "highlighted_sentences": [], "num_chunks": 0})

    return results


def _process_short_docs_batched(
    short_docs: List[Tuple[int, str, str]],
    model: nn.Module,
    device: torch.device,
    batch_size: int
) -> Dict[int, Dict[str, Any]]:
    """Process short documents with batched tokenization"""
    results = {}

    for batch_start in range(0, len(short_docs), batch_size):
        batch_end = min(batch_start + batch_size, len(short_docs))
        batch = short_docs[batch_start:batch_end]

        # Batch NLTK sentence tokenization
        all_contexts = [c for _, _, c in batch]
        all_doc_sents = [nltk.sent_tokenize(c) for c in all_contexts]

        batch_data = []
        for idx, (orig_idx, q, c) in enumerate(batch):
            doc_sents = all_doc_sents[idx]

            # Calculate sentence positions using list comprehension
            sentence_positions = []
            curr_pos = 0
            for sent in doc_sents:
                start = c.find(sent, curr_pos)
                if start == -1:
                    start = curr_pos
                end = start + len(sent)
                sentence_positions.append((start, end))
                curr_pos = end

            # Vectorized word splitting and sentence ID mapping
            words = []
            sentence_ids = []
            for sid, sent in enumerate(doc_sents):
                ws = sent.split()
                words.extend(ws)
                sentence_ids.extend([sid] * len(ws))

            batch_data.append({
                'orig_idx': orig_idx,
                'question': q,
                'words': words,
                'sentence_ids': sentence_ids,
                'doc_sents': doc_sents,
                'sentence_positions': sentence_positions
            })

        # Prepare batch dataset
        questions_batch = [[item['question']] for item in batch_data]
        contexts_batch = [item['words'] for item in batch_data]
        sentence_ids_batch = [item['sentence_ids'] for item in batch_data]
        ids_batch = list(range(len(batch_data)))

        batch_dataset = Dataset.from_dict({
            "question": questions_batch,
            "context": contexts_batch,
            "word_level_sentence_ids": sentence_ids_batch,
            "id": ids_batch,
        })

        # Batch tokenization - this is the key optimization
        batch_dataset = batch_dataset.map(
            partial(
                prepare_input_features,
                _tokenizer,
                max_seq_length=MAX_LENGTH,
                stride=STRIDE,
            ),
            batched=True,
            remove_columns=batch_dataset.column_names,
        )

        # Prepare features for inference
        features_list = [batch_dataset[i] for i in range(len(batch_dataset))]
        collator = DataCollatorWithPadding(
            pad_kvs={
                "input_ids": 0,
                "token_type_ids": 0,
                "attention_mask": 0,
                "sentence_ids": -100,
            }
        )
        batch_inputs = collator(features_list)
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

        with torch.no_grad():
            preds = model(**batch_inputs)

        # Process results
        for i, (batch_item, sent_pred) in enumerate(zip(batch_data, preds)):
            orig_idx = batch_item['orig_idx']
            doc_sents = batch_item['doc_sents']
            sentence_positions = batch_item['sentence_positions']

            highlights = []
            highlighted_sents = []

            if sent_pred is not None and len(sent_pred) > 0:
                for idx in sent_pred.cpu().tolist():
                    if 0 <= idx < len(doc_sents):
                        start_pos, end_pos = sentence_positions[idx]
                        highlights.append({
                            "text": doc_sents[idx],
                            "start": start_pos,
                            "end": end_pos,
                            "position": idx,
                            "score": 1.0,
                        })
                        highlighted_sents.append(doc_sents[idx])

            results[orig_idx] = {
                "highlights": sorted(highlights, key=lambda x: x["position"]),
                "highlighted_sentences": highlighted_sents,
                "num_chunks": 1,
            }

    return results


@dataclass
class DataCollatorWithPadding:
    """Data collator with padding"""
    pad_kvs: Dict[str, Union[int, float]] = field(default_factory=dict)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        first = features[0]
        batch = {}
        
        for key, pad_value in self.pad_kvs.items():
            if key in first and first[key] is not None:
                batch[key] = pad_sequence(
                    [torch.tensor(f[key]) for f in features],
                    batch_first=True,
                    padding_value=pad_value,
                )
        
        return batch


def prepare_input_features(
    tokenizer, examples, max_seq_length=510, stride=128, padding=False
):
    """Prepare input features"""
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding=padding,
        is_split_into_words=True,
    )
    
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []
    tokenized_examples["word_ids"] = []
    tokenized_examples["sentence_ids"] = []
    
    for i, sample_index in enumerate(sample_mapping):
        word_ids = tokenized_examples.word_ids(i)
        word_level_sentence_ids = examples["word_level_sentence_ids"][sample_index]
        
        sequence_ids = tokenized_examples.sequence_ids(i)
        token_start_index = 0
        while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
            token_start_index += 1
        
        sentences_ids = [-100] * token_start_index
        for word_idx in word_ids[token_start_index:]:
            if word_idx is not None and word_idx < len(word_level_sentence_ids):
                sentences_ids.append(word_level_sentence_ids[word_idx])
            else:
                sentences_ids.append(-100)
        
        tokenized_examples["sentence_ids"].append(sentences_ids)
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["word_ids"].append(word_ids)
    
    for key in ("input_ids", "token_type_ids", "attention_mask", "sentence_ids"):
        tokenized_examples[key] = [seq[:max_seq_length] for seq in tokenized_examples[key]]
    
    return tokenized_examples


def input_fn(request_body: str, request_content_type: str) -> dict:
    """Parse input data"""
    if request_content_type != 'application/json':
        raise ValueError(f"Unsupported content type: {request_content_type}")

    input_data = json.loads(request_body)
    return input_data


def predict_fn(input_data, model) -> InferenceResponse:
    """Run prediction with batch processing"""
    
    start_time = time.time()

    # Batch with individual questions (if inputs has content)
    if "inputs" in input_data and input_data["inputs"]:
        inputs = input_data["inputs"]
        questions = [item["question"] for item in inputs]
        contexts = [item["context"] for item in inputs]

        results = process_batch_documents(
            questions, contexts, model, DEVICE, MAX_BATCH_SIZE
        )

        highlights_list = []
        for result in results:
            doc_highlights = [
                {"start": h["start"], "end": h["end"]}
                for h in result.get("highlights", [])
            ]
            highlights_list.append(doc_highlights)

        response = {
            "highlights": highlights_list,
            "processing_time_ms": (time.time() - start_time) * 1000,
            "device": str(DEVICE)
        }
    
    # Single document (if question/context have content)
    elif ("question" in input_data and input_data["question"] and
          "context" in input_data and input_data["context"] and
          "contexts" not in input_data):
        question = input_data['question']
        context = input_data['context']

        result = process_single_document(
            question, context, model, DEVICE
        )

        highlights = [
            {"start": h["start"], "end": h["end"]}
            for h in result.get("highlights", [])
        ]

        response = {
            "highlights": highlights,
            "processing_time_ms": (time.time() - start_time) * 1000,
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
    """Format output"""
    if response_content_type != 'application/json':
        raise ValueError(f"Unsupported content type: {response_content_type}")
    
    return json.dumps(prediction)
