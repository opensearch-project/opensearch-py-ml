import json
import os
import logging
import time
import nltk
import torch
import torch.nn.functional
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for core optimizations
GLOBAL_DEVICE = None
TOKENIZER_CACHE = None

logger.info("Optimized inference script started")

def ensure_nltk_data():
    """Ensure NLTK punkt data is available"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        logger.info("Successfully downloaded NLTK data")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
        raise

def setup_local_tokenizer():
    """Setup local tokenizer cache to avoid HuggingFace API calls"""
    global TOKENIZER_CACHE
    
    try:
        cache_dir = "/tmp/tokenizer_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        tokenizer_name = "bert-base-uncased"
        tokenizer_path = os.path.join(cache_dir, "bert-base-uncased")
        
        if not os.path.exists(tokenizer_path):
            logger.info("Downloading and caching tokenizer locally...")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                cache_dir=cache_dir,
                local_files_only=False
            )
            tokenizer.save_pretrained(tokenizer_path)
            logger.info(f"Tokenizer cached at: {tokenizer_path}")
        else:
            logger.info("Using existing cached tokenizer")
        
        # Load tokenizer from local cache
        TOKENIZER_CACHE = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True
        )
        logger.info("Local tokenizer setup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to setup local tokenizer: {str(e)}")
        logger.info("Falling back to online tokenizer")
        try:
            TOKENIZER_CACHE = AutoTokenizer.from_pretrained("bert-base-uncased")
            logger.info("Fallback tokenizer loaded successfully")
        except Exception as fallback_error:
            logger.error(f"Fallback tokenizer also failed: {str(fallback_error)}")
            TOKENIZER_CACHE = None

def model_fn(model_dir):
    """Load the model for inference"""
    global GLOBAL_DEVICE
    
    logger.info(f"Loading model from {model_dir}")
    
    # Setup device with GPU optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GLOBAL_DEVICE = device
    logger.info(f"Using device: {device}")
    
    # Ensure NLTK data is available
    ensure_nltk_data()
    
    # Setup local tokenizer cache
    setup_local_tokenizer()
    
    # Find and load the model file
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError(f"No .pt model file found in {model_dir}")
    
    model_path = os.path.join(model_dir, model_files[0])
    logger.info(f"Loading model from: {model_path}")
    
    # Load model with proper device mapping
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    
    # GPU memory optimization
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info(f"GPU memory after loading: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    
    logger.info("Model loaded successfully")
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    try:
        if request_content_type == 'application/json':
            input_data = json.loads(request_body)
            
            # Validate required fields
            if 'question' not in input_data:
                raise ValueError("Missing required field: 'question'")
            if 'context' not in input_data:
                raise ValueError("Missing required field: 'context'")
            
            return input_data
        raise ValueError(f"Unsupported content type: {request_content_type}")
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        raise

def get_tokenizer():
    """Get cached tokenizer instance"""
    global TOKENIZER_CACHE
    
    if TOKENIZER_CACHE is not None:
        return TOKENIZER_CACHE
    
    # Fallback: try to load tokenizer
    try:
        TOKENIZER_CACHE = AutoTokenizer.from_pretrained("bert-base-uncased")
        return TOKENIZER_CACHE
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        raise RuntimeError("Tokenizer not available")

def predict_fn(input_data, model):
    """Apply model to the input data with optimizations"""
    start_time = time.time()
    
    try:
        # Get input data
        question = input_data['question']
        context = input_data['context']
        
        # Use cached tokenizer for better performance
        tokenizer = get_tokenizer()
        
        # Determine device for tensor operations
        device = GLOBAL_DEVICE or next(model.parameters()).device
        
        # Step 1: Split context into sentences and assign sentence IDs
        sent_list = nltk.sent_tokenize(context)
        
        # Store original sentence positions
        sentence_positions = []
        current_pos = 0
        for sent in sent_list:
            start_pos = context.find(sent, current_pos)
            if start_pos == -1:  # If not found, use current position
                start_pos = current_pos
            end_pos = start_pos + len(sent)
            sentence_positions.append((start_pos, end_pos))
            current_pos = end_pos
        
        word_level_sentence_ids = []
        for i, sent in enumerate(sent_list):
            sent_words = sent.split(' ')
            word_level_sentence_ids.extend([i] * len(sent_words))
        
        # Step 2: Tokenize question and context
        question_words = question.split(' ')
        context_words = context.split(' ')
        
        tokenized_examples = tokenizer(
            question_words,
            context_words,
            truncation="only_second",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            padding=False,
            is_split_into_words=True,
        )
        
        # Step 3: Map word-level sentence IDs to token-level sentence IDs
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        n_chunks = len(sample_mapping)
        sentence_ids = []
        
        for i in range(n_chunks):
            # Find where the context starts
            sequence_ids = tokenized_examples.sequence_ids(i)
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            chunk_sentences_ids = [-100] * token_start_index
            
            # Map word-level IDs to token-level IDs
            word_ids = tokenized_examples.word_ids(i)
            for word_idx in word_ids[token_start_index:]:
                if word_idx is not None:
                    chunk_sentences_ids.append(word_level_sentence_ids[word_idx])
                else:
                    chunk_sentences_ids.append(-100)
            
            sentence_ids += [chunk_sentences_ids]
        
        tokenized_examples['sentence_ids'] = sentence_ids
        
        # Step 4: Run through model and get output
        highlight_sentences = []
        for i in range(n_chunks):
            input_ids = torch.LongTensor([tokenized_examples['input_ids'][i]]).to(device)
            attention_mask = torch.LongTensor([tokenized_examples['attention_mask'][i]]).to(device)
            token_type_ids = torch.LongTensor([tokenized_examples['token_type_ids'][i]]).to(device)
            sentence_ids = torch.LongTensor([tokenized_examples['sentence_ids'][i]]).to(device)
            
            with torch.no_grad():
                output = model(input_ids, attention_mask, token_type_ids, sentence_ids)
            
            for x in output:
                highlight_sentences.extend(x.tolist())
        
        # Step 5: Format output with positions
        highlighted_sentences = []
        for h in highlight_sentences:
            highlight_words = [c for c, s in zip(context_words, word_level_sentence_ids) if s == h]
            sentence = ' '.join(highlight_words)
            start_pos, end_pos = sentence_positions[h]
            highlighted_sentences.append({
                'text': sentence,
                'start': start_pos,
                'end': end_pos,
                'position': h
            })

        # Add basic performance info
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Processing completed in {processing_time:.1f}ms")
        
        return highlighted_sentences
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

def output_fn(prediction_output, response_content_type):
    """Serialize and prepare the prediction output"""
    try:
        if response_content_type == 'application/json':
            # Format the output in the desired structure
            formatted_output = {
                "highlights": []
            }
            
            # Add each highlighted sentence to the output
            for sentence in prediction_output:
                highlight = {
                    "start": sentence['start'],
                    "end": sentence['end'],
                    "text": sentence['text'],
                    "position": sentence['position']
                }
                formatted_output["highlights"].append(highlight)
            
            response = json.dumps(formatted_output)
            return response
        raise ValueError(f"Unsupported content type: {response_content_type}")
    except Exception as e:
        logger.error(f"Error preparing output: {str(e)}")
        raise

logger.info("Inference script loaded successfully")
