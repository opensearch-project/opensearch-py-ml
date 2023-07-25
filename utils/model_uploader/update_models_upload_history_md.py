# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

# This program is run by "Model Auto-tracing & Uploading" workflow
# (See model_uploader.yml) to update MODEL_UPLOAD_HISTORY.md & supported_models.json
# after uploading the model to our model hub.

import argparse
import json
import os
from typing import Dict, List, Optional

from mdutils.fileutils import MarkDownFile
from mdutils.tools.Table import Table

BOTH_FORMAT = "BOTH"
TORCH_SCRIPT_FORMAT = "TORCH_SCRIPT"
ONNX_FORMAT = "ONNX"

MD_FILENAME = "MODEL_UPLOAD_HISTORY.md"
JSON_FILENAME = "supported_models.json"
DIRNAME = "utils/model_uploader/upload_history"
MODEL_JSON_FILEPATH = os.path.join(DIRNAME, JSON_FILENAME)
KEYS = [
    "Upload Time",
    "Model Uploader",
    "Model ID",
    "Model Version",
    "Model Format",
    "Embedding Dimension",
    "Pooling Mode",
]
MD_HEADER = "# Pretrained Model Upload History\n\nThe model-serving framework supports a variety of open-source pretrained models that can assist with a range of machine learning (ML) search and analytics use cases. \n\n\n## Uploaded Pretrained Models\n\n\n### Sentence transformers\n\nSentence transformer models map sentences and paragraphs across a dimensional dense vector space. The number of vectors depends on the model. Use these models for use cases such as clustering and semantic search. \n\nThe following table shows sentence transformer model upload history.\n\n[//]: # (This may be the most platform independent comment)\n"


def create_model_json_obj(
    model_id: str,
    model_version: str,
    model_format: str,
    embedding_dimension: Optional[int] = None,
    pooling_mode: Optional[str] = None,
    model_uploader: Optional[str] = None,
    uploader_time: Optional[str] = None,
) -> Dict:
    """
    Create a model dict obj to be added to supported_models.json

    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param model_version: Version of the pretrained model for registration
    :type model_version: string
    :param model_format: Model format ("TORCH_SCRIPT" or "ONNX")
    :type model_format: string
    :param embedding_dimension: Embedding dimension input
    :type embedding_dimension: int
    :param pooling_mode: Pooling mode input ("CLS", "MEAN", "MAX", "MEAN_SQRT_LEN" or None)
    :type pooling_mode: string
    """
    model_obj = {
        "Model Uploader": "@" + model_uploader if model_uploader is not None else "-",
        "Upload Time": uploader_time if uploader_time is not None else "-",
        "Model ID": model_id,
        "Model Version": model_version,
        "Model Format": model_format,
        "Embedding Dimension": str(embedding_dimension)
        if embedding_dimension is not None
        else "Default",
        "Pooling Mode": pooling_mode if pooling_mode is not None else "Default",
    }
    return model_obj


def sort_models(models: List[Dict]) -> List[Dict]:
    """
    Sort models

    :param models: List of models to be sorted
    :type models: list[dict]
    """
    models = sorted(
        models,
        key=lambda d: (
            d["Upload Time"],
            d["Model Version"],
            d["Model ID"],
            d["Model Format"],
        ),
    )
    return models


def update_model_json_file(
    model_id: str,
    model_version: str,
    tracing_format: str,
    embedding_dimension: Optional[int] = None,
    pooling_mode: Optional[str] = None,
    model_uploader: Optional[str] = None,
    uploader_time: Optional[str] = None,
) -> None:
    """
    Update supported_models.json

    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param model_version: Version of the pretrained model for registration
    :type model_version: string
    :param tracing_format: Tracing format ("TORCH_SCRIPT", "ONNX", or "BOTH")
    :type tracing_format: string
    :param embedding_dimension: Embedding dimension input
    :type embedding_dimension: int
    :param pooling_mode: Pooling mode input ("CLS", "MEAN", "MAX", "MEAN_SQRT_LEN" or None)
    :type pooling_mode: string
    """
    models = []
    if os.path.exists(MODEL_JSON_FILEPATH):
        with open(MODEL_JSON_FILEPATH, "r") as f:
            models = json.load(f)
    else:
        os.makedirs(DIRNAME)

    if tracing_format == TORCH_SCRIPT_FORMAT or tracing_format == BOTH_FORMAT:
        model_obj = create_model_json_obj(
            model_id,
            model_version,
            TORCH_SCRIPT_FORMAT,
            embedding_dimension,
            pooling_mode,
            model_uploader,
            uploader_time,
        )
        models.append(model_obj)

    if tracing_format == ONNX_FORMAT or tracing_format == BOTH_FORMAT:
        model_obj = create_model_json_obj(
            model_id,
            model_version,
            ONNX_FORMAT,
            embedding_dimension,
            pooling_mode,
            model_uploader,
            uploader_time,
        )
        models.append(model_obj)

    models = [dict(t) for t in {tuple(m.items()) for m in models}]
    models = sort_models(models)
    with open(MODEL_JSON_FILEPATH, "w") as f:
        json.dump(models, f, indent=4)


def update_md_file():
    """
    Update MODEL_UPLOAD_HISTORY.md
    """
    models = []
    if os.path.exists(MODEL_JSON_FILEPATH):
        with open(MODEL_JSON_FILEPATH, "r") as f:
            models = json.load(f)
    models = sort_models(models)
    table_data = KEYS[:]
    for m in models:
        for k in KEYS:
            if k == "Model ID":
                table_data.append(f"`{m[k]}`")
            else:
                table_data.append(m[k])

    table = Table().create_table(
        columns=len(KEYS), rows=len(models) + 1, text=table_data, text_align="center"
    )

    mdFile = MarkDownFile(MD_FILENAME, dirname=DIRNAME)
    mdFile.rewrite_all_file(data=MD_HEADER + table)
    print(f"Finished updating {MD_FILENAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_id",
        type=str,
        help="Model ID for auto-tracing and uploading (e.g. sentence-transformers/msmarco-distilbert-base-tas-b)",
    )
    parser.add_argument(
        "model_version", type=str, help="Model version number (e.g. 1.0.1)"
    )
    parser.add_argument(
        "tracing_format",
        choices=["BOTH", "TORCH_SCRIPT", "ONNX"],
        help="Model format for auto-tracing",
    )
    parser.add_argument(
        "-ed",
        "--embedding_dimension",
        type=int,
        nargs="?",
        default=None,
        const=None,
        help="Embedding dimension of the model to use if it does not exist in original config.json",
    )

    parser.add_argument(
        "-pm",
        "--pooling_mode",
        type=str,
        nargs="?",
        default=None,
        const=None,
        choices=["CLS", "MEAN", "MAX", "MEAN_SQRT_LEN"],
        help="Pooling mode if it does not exist in original config.json",
    )

    parser.add_argument(
        "-u",
        "--model_uploader",
        type=str,
        nargs="?",
        default=None,
        const=None,
        help="Model Uploader",
    )

    parser.add_argument(
        "-t",
        "--upload_time",
        type=str,
        nargs="?",
        default=None,
        const=None,
        help="Upload Time",
    )
    args = parser.parse_args()

    update_model_json_file(
        args.model_id,
        args.model_version,
        args.tracing_format,
        args.embedding_dimension,
        args.pooling_mode,
        args.model_uploader,
        args.upload_time,
    )

    update_md_file()
