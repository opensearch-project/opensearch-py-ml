# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

import requests
import torch
from opensearchpy import OpenSearch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from opensearch_py_ml.ml_commons import ModelUploader
from opensearch_py_ml.ml_commons.ml_common_utils import (
    _generate_model_content_hash_value,
)


def _fix_tokenizer(max_len: int, path: Path):
    """
    Add truncation parameters to tokenizer file. Edits the file in place

    :param max_len: max number of tokens to truncate to
    :type max_len: int
    :param path: path to tokenizer file
    :type path: str
    """
    with open(Path(path) / "tokenizer.json", "r") as f:
        parsed = json.load(f)
    if "truncation" not in parsed or parsed["truncation"] is None:
        parsed["truncation"] = {
            "direction": "Right",
            "max_length": max_len,
            "strategy": "LongestFirst",
            "stride": 0,
        }
    with open(Path(path) / "tokenizer.json", "w") as f:
        json.dump(parsed, f, indent=2)


class CrossEncoderModel:
    """
    Class for configuring and uploading cross encoder models for opensearch
    """

    def __init__(
        self, hf_model_id: str, folder_path: str = None, overwrite: bool = False
    ) -> None:
        """
        Initialize a new CrossEncoder model from a huggingface id

        :param hf_model_id: huggingface id of the model to load
        :type hf_model_id: str
        :param folder_path: folder path to save the model
            default is /tmp/models/hf_model_id
        :type folder_path: str
        :param overwrite: whether to overwrite the existing model
        :type overwrite: bool
        :return: None
        """
        default_folder_path = Path(f"/tmp/models/{hf_model_id}")

        if folder_path is None:
            self._folder_path = default_folder_path
        else:
            self._folder_path = Path(folder_path)

        if self._folder_path.exists() and not overwrite:
            raise Exception(
                f"Folder {self._folder_path} already exists. To overwrite it, set `overwrite=True`."
            )

        self._hf_model_id = hf_model_id
        self._framework = None
        self._folder_path.mkdir(parents=True, exist_ok=True)

    def zip_model(self, framework: str = "pt") -> Path:
        """
        Compiles and zips the model to {self._folder_path}/model.zip

        :param framework: one of "pt", "onnx". The framework to zip the model as.
            default: "pt"
        :type framework: str
        :return: the path with the zipped model
        :rtype: Path
        """
        if framework == "pt":
            self._framework = "pt"
            return self._zip_model_pytorch()
        if framework == "onnx":
            self._framework = "onnx"
            return self._zip_model_onnx()
        raise Exception(
            f"Unrecognized framework {framework}. Accepted values are `pt`, `onnx`"
        )

    def _zip_model_pytorch(self) -> Path:
        """
        Compiles the model to TORCHSCRIPT format.
        """
        tk = AutoTokenizer.from_pretrained(self._hf_model_id)
        model = AutoModelForSequenceClassification.from_pretrained(self._hf_model_id)
        features = tk([["dummy sentence 1", "dummy sentence 2"]], return_tensors="pt")
        mname = Path(self._hf_model_id).name

        # bge models don't generate token type ids
        if mname.startswith("bge"):
            features["token_type_ids"] = torch.zeros_like(features["input_ids"])

        # compile
        compiled = torch.jit.trace(
            model,
            example_kwarg_inputs={
                "input_ids": features["input_ids"],
                "attention_mask": features["attention_mask"],
                "token_type_ids": features["token_type_ids"],
            },
            strict=False,
        )
        torch.jit.save(compiled, f"/tmp/{mname}.pt")

        # save tokenizer file
        tk_path = f"/tmp/{mname}-tokenizer"
        tk.save_pretrained(tk_path)
        _fix_tokenizer(tk.model_max_length, tk_path)

        # get apache license
        r = requests.get(
            "https://github.com/opensearch-project/opensearch-py-ml/raw/main/LICENSE"
        )
        with ZipFile(self._folder_path / "model.zip", "w") as f:
            f.write(f"/tmp/{mname}.pt", arcname=f"{mname}.pt")
            f.write(tk_path + "/tokenizer.json", arcname="tokenizer.json")
            f.writestr("LICENSE", r.content)

        # clean up temp files
        shutil.rmtree(f"/tmp/{mname}-tokenizer")
        os.remove(f"/tmp/{mname}.pt")
        return self._folder_path / "model.zip"

    def _zip_model_onnx(self):
        """
        Compiles the model to ONNX format.
        """
        tk = AutoTokenizer.from_pretrained(self._hf_model_id)
        model = AutoModelForSequenceClassification.from_pretrained(self._hf_model_id)
        features = tk([["dummy sentence 1", "dummy sentence 2"]], return_tensors="pt")
        mname = Path(self._hf_model_id).name

        # bge models don't generate token type ids
        if mname.startswith("bge"):
            features["token_type_ids"] = torch.zeros_like(features["input_ids"])

        # export to onnx
        onnx_model_path = f"/tmp/{mname}.onnx"
        torch.onnx.export(
            model=model,
            args=(
                features["input_ids"],
                features["attention_mask"],
                features["token_type_ids"],
            ),
            f=onnx_model_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "token_type_ids": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size"},
            },
            verbose=True,
        )

        # save tokenizer file
        tk_path = f"/tmp/{mname}-tokenizer"
        tk.save_pretrained(tk_path)
        _fix_tokenizer(tk.model_max_length, tk_path)

        # get apache license
        r = requests.get(
            "https://github.com/opensearch-project/opensearch-py-ml/raw/main/LICENSE"
        )
        with ZipFile(self._folder_path / "model.zip", "w") as f:
            f.write(onnx_model_path, arcname=f"{mname}.pt")
            f.write(tk_path + "/tokenizer.json", arcname="tokenizer.json")
            f.writestr("LICENSE", r.content)

        # clean up temp files
        shutil.rmtree(f"/tmp/{mname}-tokenizer")
        os.remove(onnx_model_path)
        return self._folder_path / "model.zip"

    def make_model_config_json(
        self,
        model_name: str = None,
        version_number: str = 1,
        description: str = None,
        all_config: str = None,
        model_type: str = None,
        verbose: bool = False,
    ):
        """
        Parse from config.json file of pre-trained hugging-face model to generate a ml-commons_model_config.json file.
        If all required fields are given by users, use the given parameters and will skip reading the config.json

        :param model_name:
            Optional, The name of the model. If None, default is model id, for example,
            'sentence-transformers/msmarco-distilbert-base-tas-b'
        :type model_name: string
        :param version_number:
            Optional, The version number of the model. Default is 1
        :type version_number: string
        :param description: Optional, the description of the model. If None, get description from the README.md
            file in the model folder.
        :type description: str
        :param all_config:
            Optional, the all_config of the model. If None, parse all contents from the config file of pre-trained
            hugging-face model
        :type all_config: dict
        :param model_type:
            Optional, the model_type of the model. If None, parse model_type from the config file of pre-trained
            hugging-face model
        :type model_type: string
        :param verbose:
            optional, use printing more logs. Default as false
        :type verbose: bool
        :return: model config file path. The file path where the model config file is being saved
        :rtype: string
        """
        if not (self._folder_path / "model.zip").exists():
            raise Exception("Generate the model zip before generating the config")
        hash_value = _generate_model_content_hash_value(
            str(self._folder_path / "model.zip")
        )
        if model_name is None:
            model_name = Path(self._hf_model_id).name
        if description is None:
            description = f"Cross Encoder Model {model_name}"
        if all_config is None:
            cfg = AutoConfig.from_pretrained(self._hf_model_id)
            all_config = cfg.to_json_string()
        if model_type is None:
            model_type = "bert"
        model_format = None
        if self._framework is not None:
            model_format = {"pt": "TORCH_SCRIPT", "onnx": "ONNX"}.get(self._framework)
        if model_format is None:
            raise Exception(
                "Model format either not found or not supported. Zip the model before generating the config"
            )
        model_config_content = {
            "name": model_name,
            "version": f"1.0.{version_number}",
            "description": description,
            "model_format": model_format,
            "function_name": "TEXT_SIMILARITY",
            "model_content_hash_value": hash_value,
            "model_config": {
                "model_type": model_type,
                "embedding_dimension": 1,
                "framework_type": "huggingface_transformers",
                "all_config": all_config,
            },
        }
        if verbose:
            print(json.dumps(model_config_content, indent=2))
        with open(self._folder_path / "config.json", "w") as f:
            json.dump(model_config_content, f)
        return self._folder_path / "config.json"

    def upload(
        self,
        client: OpenSearch,
        framework: str = "pt",
        model_group_id: str = "",
        verbose: bool = False,
    ):
        """
        Upload the model to OpenSearch

        :param client: OpenSearch client
        :type client: OpenSearch
        :param framework: either 'pt' or 'onnx'
        :type framework: str
        :param model_group_id: model group id to upload this model to
        :type model_group_id: str
        :param verbose: log a bunch or not
        :type verbose: bool
        """
        config_path = self._folder_path / "config.json"
        model_path = self._folder_path / "model.zip"
        gen_cfg = False
        if not model_path.exists() or self._framework != framework:
            gen_cfg = True
            self.zip_model(framework)
        if not config_path.exists() or gen_cfg:
            self.make_model_config_json()
        uploader = ModelUploader(client)
        uploader._register_model(
            str(model_path), str(config_path), model_group_id, verbose
        )