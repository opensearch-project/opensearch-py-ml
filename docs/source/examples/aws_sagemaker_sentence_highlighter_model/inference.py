import json
import os
import logging
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

logger.info("inference.py script started")

def ensure_nltk_data():
    """Ensure NLTK punkt data is available"""
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        logger.info("Successfully downloaded NLTK data")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
        raise

# Ensure NLTK data is available at startup
ensure_nltk_data()

def model_fn(model_dir):
    """Load the model for inference"""
    try:
        # Log environment information
        logger.info("Environment Information:")
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"Current CUDA Device: {torch.cuda.current_device()}")
            logger.info(f"CUDA Device Name: {torch.cuda.get_device_name()}")

        # Load model file
        model_path = os.path.join(model_dir, "opensearch-semantic-highlighter-v1.pt")
        logger.info(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Load the model
        model = torch.jit.load(model_path)
        logger.info("Model loaded successfully")
        model.eval()
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    
    Expected input format (JSON):
    {
        "question": "What is the main topic discussed?",
        "context": "This is a long text document containing multiple sentences. The model will identify which sentences are most relevant to answering the question."
    }
    
    Args:
        request_body: The request body as bytes
        request_content_type: Content type of the request (should be 'application/json')
    
    Returns:
        dict: Parsed input data containing 'question' and 'context' fields
    
    Raises:
        ValueError: If content type is not supported or required fields are missing
    """
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
        logger.error(f"Error processing input: {str(e)}", exc_info=True)
        raise

def predict_fn(input_data, model):
    """Apply model to the input data"""
    try:
        # Get input data
        question = input_data['question']
        context = input_data['context']
        
        # Determine device for tensor operations
        device = next(model.parameters()).device
        
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
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
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
            logger.info(f"Processing chunk {i+1}/{n_chunks}")
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

        return highlighted_sentences
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise

def output_fn(prediction_output, response_content_type):
    """
    Serialize and prepare the prediction output
    
    Output format (JSON):
    {
        "highlights": [
            {
                "start": 45,
                "end": 123,
                "text": "This sentence is relevant to the question.",
                "position": 2
            },
            ...
        ]
    }
    
    Args:
        prediction_output: List of highlighted sentences with metadata
        response_content_type: Content type for the response (should be 'application/json')
    
    Returns:
        str: JSON-formatted response
    
    Raises:
        ValueError: If content type is not supported
    """
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
        logger.error(f"Error preparing output: {str(e)}", exc_info=True)
        raise
