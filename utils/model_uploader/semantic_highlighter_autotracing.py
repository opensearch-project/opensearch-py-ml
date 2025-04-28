# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import argparse
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
    TORCH_SCRIPT_FORMAT,
    autotracing_warning_filters,
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

    tokenized_examples["example_id"] = []
    tokenized_examples["word_ids"] = []
    tokenized_examples["sentence_ids"] = []
    tokenized_examples["answer_sentence_ids"] = []
    tokenized_examples["sentence_labels"] = []

    for i, sample_index in enumerate(sample_mapping):
        word_ids = tokenized_examples.word_ids(i)
        answer_ids = set(np.where(examples["orig_sentence_labels"][sample_index])[0])
        word_level_sentence_ids = examples["word_level_sentence_ids"][sample_index]

        sequence_ids = tokenized_examples.sequence_ids(i)
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

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
    question = "When does OpenSearch use text reanalysis for highlighting?"
    passage = "To highlight the search terms, the highlighter needs the start and end character offsets of each term. The offsets mark the term's position in the original text. The highlighter can obtain the offsets from the following sources: Postings: When documents are indexed, OpenSearch creates an inverted search index—a core data structure used to search for documents. Postings represent the inverted search index and store the mapping of each analyzed term to the list of documents in which it occurs. If you set the index_options parameter to offsets when mapping a text field, OpenSearch adds each term's start and end character offsets to the inverted index. During highlighting, the highlighter reruns the original query directly on the postings to locate each term. Thus, storing offsets makes highlighting more efficient for large fields because it does not require reanalyzing the text. Storing term offsets requires additional disk space, but uses less disk space than storing term vectors. Text reanalysis: In the absence of both postings and term vectors, the highlighter reanalyzes text in order to highlight it. For every document and every field that needs highlighting, the highlighter creates a small in-memory index and reruns the original query through Lucene's query execution planner to access low-level match information for the current document. Reanalyzing the text works well in most use cases. However, this method is more memory and time intensive for large fields."

    sentence_ids = []
    context = []
    passage_sents = nltk.sent_tokenize(passage)
    for sent_id, sent in enumerate(passage_sents):
        sent_words = sent.split(" ")
        context += sent_words
        sentence_ids += [sent_id] * len(sent_words)

    orig_sentence_labels = [0] * len(passage_sents)
    orig_sentence_labels[8] = 1
    trace_dataset = Dataset.from_dict(
        {
            "question": [[question]],
            "context": [context],
            "word_level_sentence_ids": [sentence_ids],
            "orig_sentence_labels": [orig_sentence_labels],
            "id": ["test"],
        }
    )
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
) -> None:
    """
    Perform model auto-tracing and prepare files for uploading to OpenSearch model hub

    Parameters
    ----------
    model_id : str
        Model ID of the pretrained model
    model_version : str
        Version of the pretrained model for registration
    tracing_format : str
        Tracing format ("TORCH_SCRIPT" only for semantic highlighter)
    model_description : str, optional
        Model description input, by default None
    upload_prefix : str, optional
        Model upload prefix input, by default None
    model_name : str, optional
        Model customize name for upload, by default None
    skip_deployment : bool, optional
        Skip the deployment and verification steps, by default False
    """
    print(
        f"""
    === Begin running semantic_highlighter_autotracing.py ===
    Model ID: {model_id}
    Model Version: {model_version}
    Tracing Format: {tracing_format}
    Model Description: {model_description if model_description is not None else 'N/A'}
    Upload Prefix: {upload_prefix if upload_prefix is not None else 'N/A'}
    Model Name: {model_name if model_name is not None else 'N/A'}
    Skip Deployment: {skip_deployment}
    ==========================================
    """
    )

    # Semantic highlighter only supports TorchScript
    assert (
        tracing_format == TORCH_SCRIPT_FORMAT
    ), f"Semantic highlighter only supports {TORCH_SCRIPT_FORMAT}"

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

    print("--- Begin tracing semantic highlighter model ---")
    test_model = SemanticHighlighterModel(
        model_id=model_id, folder_path="semantic-highlighter/", overwrite=True
    )

    # Generate example inputs for tracing
    trace_dataset = generate_tracing_dataset()
    example_inputs = {
        "input_ids": torch.tensor(trace_dataset[0]["input_ids"]).unsqueeze(
            0
        ),  # Add batch dimension
        "attention_mask": torch.tensor(trace_dataset[0]["attention_mask"]).unsqueeze(0),
        "token_type_ids": torch.tensor(trace_dataset[0]["token_type_ids"]).unsqueeze(0),
        "sentence_ids": torch.tensor(trace_dataset[0]["sentence_ids"]).unsqueeze(0),
    }

    print("Input shapes:")
    for k, v in example_inputs.items():
        print(f"{k}: {v.shape}")

    # Trace and save model
    torchscript_model_path = test_model.save_as_pt(
        example_inputs=example_inputs,
        model_id=model_id,
        model_name=None,  # Use default
        add_apache_license=True,
    )

    # Create config file
    torchscript_model_config_path = test_model.make_model_config_json(
        model_name=model_name,
        version_number=model_version,
        model_format=TORCH_SCRIPT_FORMAT,
        description=model_description,
    )

    # Preview config
    preview_model_config(TORCH_SCRIPT_FORMAT, torchscript_model_config_path)

    if not skip_deployment:
        print("--- Testing model deployment ---")
        model_id = register_and_deploy_model(
            ml_client,
            TORCH_SCRIPT_FORMAT,
            torchscript_model_path,
            torchscript_model_config_path,
        )

        # Note: Add deployment test logic here if needed
        # For now, we just verify registration works

        print("--- Undeploying and cleaning up test model ---")
        if model_id:
            try:
                ml_client.undeploy_model(model_id)
                ml_client.delete_model(model_id)
            except Exception as e:
                print(f"Warning: Cleanup failed: {e}")

    # Prepare files for upload
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

    # Store verification results
    store_license_verified_variable(license_verified)
    store_description_variable(torchscript_dst_model_config_path)

    print("\n=== Finished running semantic_highlighter_autotracing.py ===")


if __name__ == "__main__":
    autotracing_warning_filters()

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

    args = parser.parse_args()
    for arg in vars(args):
        value = getattr(args, arg)
        if isinstance(value, str) and value.strip() == "":
            setattr(args, arg, None)

    main(
        args.model_id,
        args.model_version,
        args.tracing_format,
        args.model_description,
        args.upload_prefix,
        args.model_name,
        args.skip_deployment,
    )
