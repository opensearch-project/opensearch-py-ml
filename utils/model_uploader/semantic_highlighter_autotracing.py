# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""
Semantic Highlighter Auto-tracing Utility for OpenSearch

This module provides functionality for auto-tracing semantic highlighter models
for deployment in OpenSearch. It handles creating example inputs, tracing the model,
packaging it for upload, and optionally testing the deployment in a test environment.

Key components:
- Example input generation for model tracing
- Model tracing to TorchScript format
- Configuration file generation
- Optional deployment testing
- Preparing files for upload to OpenSearch model repository

This utility is used to prepare semantic highlighter models for efficient deployment
in OpenSearch ML Commons.
"""

import argparse
import json
import os
import sys
from typing import Optional

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(THIS_DIR, "../..")
sys.path.append(ROOT_DIR)

from functools import partial

import nltk
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from opensearch_py_ml.ml_commons import MLCommonClient
from opensearch_py_ml.ml_models import SemanticHighlighterModel
from tests import OPENSEARCH_TEST_CLIENT
from utils.model_uploader.autotracing_utils import (
    QUESTION_ANSWERING_ALGORITHM,
    TORCH_SCRIPT_FORMAT,
    autotracing_warning_filters,
    check_model_status,
    prepare_files_for_uploading,
    preview_model_config,
    register_and_deploy_model,
    store_description_variable,
    store_license_verified_variable,
    verify_license_by_hfapi,
)


def prepare_train_features(
    tokenizer, examples, max_seq_length=512, stride=128, padding=False
):
    """
    Prepare tokenized training features for the semantic highlighter model.

    This function tokenizes the input examples and extracts sentence-level information
    required for training and tracing the semantic highlighter model.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer to use for processing text
    examples : dict
        Dictionary containing questions, contexts, and sentence annotation data
    max_seq_length : int, default=512
        Maximum sequence length for tokenization
    stride : int, default=128
        Stride length for tokenization with overlap
    padding : bool, default=False
        Whether to pad sequences to max_seq_length

    Returns
    -------
    dict
        Dictionary containing tokenized and processed features including:
        - input_ids, attention_mask, token_type_ids: standard BERT inputs
        - sentence_ids: token-level sentence IDs
        - sentence_labels: binary labels for sentences
        - example_id: example identifiers
    """
    # jointly tokenize questions and context
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

    # Create data structures to hold processed features
    tokenized_examples["example_id"] = []
    tokenized_examples["word_ids"] = []
    tokenized_examples["sentence_ids"] = []
    tokenized_examples["answer_sentence_ids"] = []
    tokenized_examples["sentence_labels"] = []

    for i, sample_index in enumerate(sample_mapping):
        # Get word ids for current feature
        word_ids = tokenized_examples.word_ids(i)
        # Get marked answer sentences from original data
        answer_ids = set(np.where(examples["orig_sentence_labels"][sample_index])[0])
        # Get sentence mappings for each word
        word_level_sentence_ids = examples["word_level_sentence_ids"][sample_index]

        # Identify the context start position (after question tokens)
        sequence_ids = tokenized_examples.sequence_ids(i)
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # Map each token to its corresponding sentence id
        # Use -100 for special tokens and question tokens
        sentences_ids = [-100] * token_start_index
        for word_idx in word_ids[token_start_index:]:
            if word_idx is not None:
                sentences_ids.append(word_level_sentence_ids[word_idx])
            else:
                sentences_ids.append(-100)

        sentence_labels = [0] * (max(sentences_ids) + 1)
        answer_ids = set()
        for sentence_id in sentences_ids:
            if (
                sentence_id >= 0
                and examples["orig_sentence_labels"][sample_index][sentence_id] == 1
            ):
                sentence_labels[sentence_id] = 1
                answer_ids.add(sentence_id)

        tokenized_examples["sentence_ids"].append(sentences_ids)
        tokenized_examples["sentence_labels"].append(sentence_labels)
        tokenized_examples["answer_sentence_ids"].append(answer_ids)
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["word_ids"].append(word_ids)

    return tokenized_examples


def generate_tracing_dataset():
    """
    Generate a sample dataset for tracing the semantic highlighter model.

    This function creates a small example dataset with a question and context passage
    about OpenSearch highlighting, with sentence-level annotations. The dataset is
    processed with the tokenizer to generate inputs suitable for model tracing.

    The example contains:
    - A question about OpenSearch highlighting
    - A context passage with sentences about different highlighting methods
    - Sentence-level annotations marking relevant sentences

    Returns
    -------
    datasets.Dataset
        A processed dataset containing tokenized inputs ready for model tracing,
        including input_ids, attention_mask, token_type_ids, and sentence_ids.
    """
    # Define a question and corresponding passage about OpenSearch highlighting
    question = "When does OpenSearch use text reanalysis for highlighting?"
    passage = "To highlight the search terms, the highlighter needs the start and end character offsets of each term. The offsets mark the term's position in the original text. The highlighter can obtain the offsets from the following sources: Postings: When documents are indexed, OpenSearch creates an inverted search index—a core data structure used to search for documents. Postings represent the inverted search index and store the mapping of each analyzed term to the list of documents in which it occurs. If you set the index_options parameter to offsets when mapping a text field, OpenSearch adds each term's start and end character offsets to the inverted index. During highlighting, the highlighter reruns the original query directly on the postings to locate each term. Thus, storing offsets makes highlighting more efficient for large fields because it does not require reanalyzing the text. Storing term offsets requires additional disk space, but uses less disk space than storing term vectors. Text reanalysis: In the absence of both postings and term vectors, the highlighter reanalyzes text in order to highlight it. For every document and every field that needs highlighting, the highlighter creates a small in-memory index and reruns the original query through Lucene's query execution planner to access low-level match information for the current document. Reanalyzing the text works well in most use cases. However, this method is more memory and time intensive for large fields."

    # Split passage into words and assign sentence IDs to each word
    sentence_ids = []
    context = []
    passage_sents = nltk.sent_tokenize(passage)
    for sent_id, sent in enumerate(passage_sents):
        sent_words = sent.split(" ")
        context += sent_words
        sentence_ids += [sent_id] * len(sent_words)

    # Mark the relevant sentence (sentence 8 contains the answer)
    orig_sentence_labels = [0] * len(passage_sents)
    orig_sentence_labels[8] = 1

    # Create dataset with the question, context and sentence annotations
    trace_dataset = Dataset.from_dict(
        {
            "question": [[question]],
            "context": [context],
            "word_level_sentence_ids": [sentence_ids],
            "orig_sentence_labels": [orig_sentence_labels],
            "id": ["test"],
        }
    )

    # Initialize tokenizer and process the dataset
    base_model_id = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    preprocess_fn = partial(prepare_train_features, tokenizer)
    trace_dataset = trace_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=trace_dataset.column_names,
        desc="Preparing model inputs",
    )
    return trace_dataset


def main(
    model_id: str,
    model_version: str,
    tracing_format: str,
    model_description: Optional[str] = None,
    upload_prefix: Optional[str] = None,
    model_name: Optional[str] = None,
    skip_deployment: bool = False,
    device: Optional[str] = None,
) -> None:
    """
    Perform model auto-tracing and prepare files for uploading to OpenSearch model hub.

    This function handles the complete workflow for preparing a semantic highlighter model:
    1. Verify the model license (Apache-2.0)
    2. Generate example inputs for tracing
    3. Trace the model to TorchScript format
    4. Create configuration files
    5. Optionally test deployment in a local OpenSearch instance
    6. Prepare files for uploading to the model repository

    Parameters
    ----------
    model_id : str
        Model ID of the pretrained Hugging Face model to trace
    model_version : str
        Version number to assign to the traced model for registration
    tracing_format : str
        Model format for tracing (only "TORCH_SCRIPT" is supported for semantic highlighter)
    model_description : str, optional
        Custom description for the model in the config file
    upload_prefix : str, optional
        Path prefix for the uploaded model files
    model_name : str, optional
        Custom name for the model in the config file and when registered
    skip_deployment : bool, optional
        Whether to skip testing deployment in a local OpenSearch instance
    """
    print(f"""
    === Begin running semantic_highlighter_autotracing.py ===
    Model ID: {model_id}
    Model Version: {model_version}
    Tracing Format: {tracing_format}
    Model Description: {model_description if model_description is not None else 'N/A'}
    Upload Prefix: {upload_prefix if upload_prefix is not None else 'N/A'}
    Model Name: {model_name if model_name is not None else 'N/A'}
    Skip Deployment: {skip_deployment}
    Device Selection: Auto (prefer GPU if available)
    ==========================================
    """)

    # Semantic highlighter only supports TorchScript
    assert (
        tracing_format == TORCH_SCRIPT_FORMAT
    ), f"Semantic highlighter only supports {TORCH_SCRIPT_FORMAT}"

    # Initialize ML Commons client for testing if deployment is not skipped
    ml_client = None
    if not skip_deployment:
        print("--- Initializing MLCommonClient for deployment test ---")
        ml_client = MLCommonClient(OPENSEARCH_TEST_CLIENT)
    else:
        print("--- Skipping MLCommonClient initialization ---")

    # Verify license using Hugging Face API
    license_verified = False  # Default to false
    try:
        print(f"--- Verifying license for model {model_id} ---")
        license_verified = verify_license_by_hfapi(model_id)
        if license_verified:
            print("License verified as Apache-2.0.")
        else:
            print("License could not be verified as Apache-2.0 by Hugging Face API.")
    except Exception as e:
        print(f"Warning: License verification failed: {e}")
        print("Proceeding with license_verified=False.")

    # Initialize the semantic highlighter model
    print("--- Begin tracing semantic highlighter model ---")
    test_model = SemanticHighlighterModel(
        model_id=model_id, folder_path="semantic-highlighter/", overwrite=True
    )

    # Generate example inputs for tracing by creating a sample dataset
    trace_dataset = generate_tracing_dataset()
    example_inputs = {
        "input_ids": torch.tensor(trace_dataset[0]["input_ids"]).unsqueeze(
            0
        ),  # Add batch dimension
        "attention_mask": torch.tensor(trace_dataset[0]["attention_mask"]).unsqueeze(0),
        "token_type_ids": torch.tensor(trace_dataset[0]["token_type_ids"]).unsqueeze(0),
        "sentence_ids": torch.tensor(trace_dataset[0]["sentence_ids"]).unsqueeze(0),
    }

    # Log the shapes of input tensors
    print("Input shapes:")
    for k, v in example_inputs.items():
        print(f"{k}: {v.shape}")

    # Trace and save the model to TorchScript format
    torchscript_model_path = test_model.save_as_pt(
        example_inputs=example_inputs,
        model_id=model_id,
        model_name=None,  # Use default
        add_apache_license=True,
    )

    # Create model configuration file for OpenSearch ML Commons
    torchscript_model_config_path = test_model.make_model_config_json(
        model_name=model_name,
        version_number=model_version,
        model_format=TORCH_SCRIPT_FORMAT,
        description=model_description,
    )

    # Show the generated configuration
    preview_model_config(TORCH_SCRIPT_FORMAT, torchscript_model_config_path)

    # Test model deployment and inference if not skipped
    if not skip_deployment:
        print("--- Testing model deployment ---")
        model_id = register_and_deploy_model(
            ml_client,
            TORCH_SCRIPT_FORMAT,
            torchscript_model_path,
            torchscript_model_config_path,
        )

        # Verify model is deployed and ready
        check_model_status(
            ml_client,
            model_id,
            TORCH_SCRIPT_FORMAT,
            QUESTION_ANSWERING_ALGORITHM,
        )

        try:
            # Test inference with a sample question and context
            question = "What are the main side effects of aspirin and when should it not be used?"
            context = "Aspirin is a commonly used medication for pain relief and fever reduction. While effective, patients may experience stomach upset and bleeding as common side effects. In rare cases, some people may develop allergic reactions. Aspirin should not be given to children under 12 due to the risk of Reye's syndrome. Additionally, people with bleeding disorders, stomach ulcers, or those about to undergo surgery should avoid aspirin. Some patients taking blood thinners must also consult their doctor before using aspirin, as it can increase bleeding risk. For minor aches and fever, the typical adult dose is 325-650 mg every 4-6 hours."

            # Run inference using the deployed model
            output = ml_client.generate_question_answering(model_id, question, context)

            # Verify output format and contents
            assert output is not None, "No output received from model"
            assert (
                "inference_results" in output
            ), "Missing inference_results in response"
            assert len(output["inference_results"]) > 0, "No inference results found"

            # Log the output for verification
            print("\n=== Model Inference Output ===")
            print(json.dumps(output, indent=2))
            print("Successfully verified model inference output")

        except Exception as e:
            print(f"Warning: Question answering failed: {e}")

        # Clean up deployed resources
        print("--- Undeploying and cleaning up test model ---")
        if model_id:
            try:
                ml_client.undeploy_model(model_id)
                ml_client.delete_model(model_id)
            except Exception as e:
                print(f"Warning: Cleanup failed: {e}")

    # Prepare files for the OpenSearch Model Hub upload
    torchscript_dst_model_path, torchscript_dst_model_config_path = (
        prepare_files_for_uploading(
            model_id,
            model_version,
            TORCH_SCRIPT_FORMAT,
            torchscript_model_path,
            torchscript_model_config_path,
            upload_prefix,
            model_name,
        )
    )

    # Store verification results in environment variables for CI/CD
    store_license_verified_variable(license_verified)
    store_description_variable(torchscript_dst_model_config_path)

    print("\n=== Finished running semantic_highlighter_autotracing.py ===")


if __name__ == "__main__":
    # Configure warning filters to suppress unnecessary warnings during tracing
    autotracing_warning_filters()

    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_id",
        type=str,
        help="Model ID for auto-tracing and uploading",
    )
    parser.add_argument(
        "model_version",
        type=str,
        help="Model version number (e.g. 1.0.1)",
    )
    parser.add_argument(
        "tracing_format",
        choices=["TORCH_SCRIPT"],
        help="Model format for auto-tracing (only TORCH_SCRIPT supported)",
    )
    parser.add_argument(
        "-up",
        "--upload_prefix",
        type=str,
        nargs="?",
        default=None,
        help="Model customize path prefix for upload",
    )
    parser.add_argument(
        "-mn",
        "--model_name",
        type=str,
        nargs="?",
        default=None,
        help="Model customize name for upload",
    )
    parser.add_argument(
        "-md",
        "--model_description",
        type=str,
        nargs="?",
        default=None,
        const=None,
        help="Model description if you want to overwrite the default description",
    )
    parser.add_argument(
        "--skip-deployment",
        action="store_true",
        help="Skip the deployment and verification steps within the script.",
    )

    # Parse arguments and handle empty string values
    args = parser.parse_args()
    for arg in vars(args):
        value = getattr(args, arg)
        if isinstance(value, str) and value.strip() == "":
            setattr(args, arg, None)

    # Run the main function with parsed arguments
    main(
        args.model_id,
        args.model_version,
        args.tracing_format,
        args.model_description,
        args.upload_prefix,
        args.model_name,
        args.skip_deployment,
        None,  # Always use auto device selection
    )
