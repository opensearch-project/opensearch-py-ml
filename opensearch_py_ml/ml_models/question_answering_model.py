# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
import pickle
import platform
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import List
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import yaml
from accelerate import Accelerator, notebook_launcher
from mdutils.fileutils import MarkDownFile
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.models import Normalize, Pooling, Transformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers.convert_graph_to_onnx import convert

from opensearch_py_ml.ml_commons.ml_common_utils import (
    _generate_model_content_hash_value,
)

LICENSE_URL = "https://github.com/opensearch-project/opensearch-py-ml/raw/main/LICENSE"


class QuestionAnsweringModel:
    """
    Class for tracing the QuestionAnswering model.
    """
    # distilbert-base-cased-distilled-squad
    DEFAULT_MODEL_ID = "distilbert-base-cased-distilled-squad"
    SYNTHETIC_QUERY_FOLDER = "synthetic_queries"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        folder_path: str = None,
        overwrite: bool = False,
    ) -> None:
        """
        Initiate a question answering model class object. The model id will be used to download
        pretrained model from the hugging-face and served as the default name for model files, and the folder_path
        will be the default location to store files generated in the following functions

        :param model_id: Optional, the huggingface mode id to download the model,
            default model id: 'distilbert-base-cased-distilled-squad'
        :type model_id: string
        :param folder_path: Optional, the path of the folder to save output files, such as queries, pre-trained model,
            after-trained custom model and configuration files. if None, default as "/model_files/" under the current
            work directory
        :type folder_path: string
        :param overwrite: Optional,  choose to overwrite the folder at folder path. Default as false. When training
                    different sentence transformer models, it's recommended to give designated folder path every time.
                    Users can choose to overwrite = True to overwrite previous runs
        :type overwrite: bool
        :return: no return value expected
        :rtype: None
        """
        default_folder_path = os.path.join(
            os.getcwd(), "question_answering_model_files"
        )

        if folder_path is None:
            self.folder_path = default_folder_path
        else:
            self.folder_path = folder_path

        # Check if self.folder_path exists
        if os.path.exists(self.folder_path) and not overwrite:
            print(
                "To prevent overwriting, please enter a different folder path or delete the folder or enable "
                "overwrite = True "
            )
            raise Exception(
                str("The default folder path already exists at : " + self.folder_path)
            )

        self.model_id = model_id
        self.torch_script_zip_file_path = None
        self.onnx_zip_file_path = None
    
    
    def save_as_pt(
        self,
        sentences: [str],
        model_id="distilbert-base-cased-distilled-squad",
        model_name: str = None,
        save_json_folder_path: str = None,
        model_output_path: str = None,
        zip_file_name: str = None,
        add_apache_license: bool = False,
    ) -> str:
        """
        Download the model directly from huggingface, convert model to torch script format,
        zip the model file and its tokenizer.json file to prepare to upload to the Open Search cluster

        :param sentences:
            Required, for example  sentences = ['today is sunny']
        :type sentences: List of string [str]
        :param model_id:
            sentence transformer model id to download model from sentence transformers.
            default model_id = "distilbert-base-cased-distilled-squad"
        :type model_id: string
        :param model_name:
            Optional, model name to name the model file, e.g, "sample_model.pt". If None, default takes the
            model_id and add the extension with ".pt"
        :type model_name: string
        :param save_json_folder_path:
             Optional, path to save model json file, e.g, "home/save_pre_trained_model_json/"). If None, default as
             default_folder_path from the constructor
        :type save_json_folder_path: string
        :param model_output_path:
             Optional, path to save traced model zip file. If None, default as
             default_folder_path from the constructor
        :type model_output_path: string
        :param zip_file_name:
            Optional, file name for zip file. e.g, "sample_model.zip". If None, default takes the model_id
            and add the extension with ".zip"
        :type zip_file_name: string
        :param add_apache_license:
            Optional, whether to add a Apache-2.0 license file to model zip file
        :type add_apache_license: string
        :return: model zip file path. The file path where the zip file is being saved
        :rtype: string
        """

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
        model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

        if model_name is None:
            model_name = str(model_id.split("/")[-1] + ".pt")

        model_path = os.path.join(self.folder_path, model_name)

        if save_json_folder_path is None:
            save_json_folder_path = self.folder_path

        if model_output_path is None:
            model_output_path = self.folder_path

        if zip_file_name is None:
            zip_file_name = str(model_id.split("/")[-1] + ".zip")
        zip_file_path = os.path.join(model_output_path, zip_file_name)

        # handle when model_max_length is unproperly defined in model's tokenizer (e.g. "intfloat/e5-small-v2")
        # MODEL_MAX_SEQ_LENGTH = 512
        # if tokenizer.model_max_length > model.get_max_seq_length():
        #     tokenizer.model_max_length = model.get_max_seq_length()
        #     print(
        #         f"The model_max_length is not properly defined in tokenizer_config.json. Setting it to be {tokenizer.model_max_length}"
        #     )

        # save tokenizer.json in save_json_folder_name
        # max_position_embeddings
        tokenizer.save_pretrained(save_json_folder_path)
        
        # Find the tokenizer.json file path in cache: /Users/faradawn/.cache/huggingface/hub/models/...
        config_json_path = os.path.join(save_json_folder_path, "tokenizer_config.json")
        with open(config_json_path) as f:
            config_json = json.load(f)
            tokenizer_file_path = config_json["tokenizer_file"]

        # Open the tokenizer.json and replace the truncation field
        with open(tokenizer_file_path) as user_file:
            parsed_json = json.load(user_file)

        if "truncation" not in parsed_json or parsed_json["truncation"] is None:
            parsed_json["truncation"] = {
                "direction": "Right",
                "max_length": tokenizer.model_max_length,
                "strategy": "LongestFirst",
                "stride": 0,
            }

        tokenizer_file_path = os.path.join(save_json_folder_path, "tokenizer.json")
        with open(tokenizer_file_path, "w") as file:
            json.dump(parsed_json, file, indent=2)


        # convert to pt format will need to be in cpu,
        # set the device to cpu, convert its input_ids and attention_mask in cpu and save as .pt format
        device = torch.device("cpu")
        cpu_model = model.to(device)
        features = tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        compiled_model = torch.jit.trace(
            cpu_model,
            (features["input_ids"], features["attention_mask"]),
            strict=False
        )
        torch.jit.save(compiled_model, model_path)
        print("Traced model is saved to ", model_path)

        # zip model file along with tokenizer.json (and license file) as output
        with ZipFile(str(zip_file_path), "w") as zipObj:
            zipObj.write(
                model_path,
                arcname=str(model_name),
            )
            zipObj.write(
                os.path.join(save_json_folder_path, "tokenizer.json"),
                arcname="tokenizer.json",
            )
        if add_apache_license:
            self._add_apache_license_to_model_zip_file(zip_file_path)

        self.torch_script_zip_file_path = zip_file_path
        print("zip file is saved to ", zip_file_path, "\n")
        return zip_file_path
    
    def make_model_config_json(
        self,
        model_name: str = None,
        version_number: str = 1,
        model_format: str = "TORCH_SCRIPT",
        model_zip_file_path: str = None,
        embedding_dimension: int = None,
        pooling_mode: str = None,
        normalize_result: bool = None,
        description: str = None,
        all_config: str = None,
        model_type: str = None,
        verbose: bool = False,
    ) -> str:
        """
        Parse from config.json file of pre-trained hugging-face model to generate a ml-commons_model_config.json file.
        If all required fields are given by users, use the given parameters and will skip reading the config.json

        :param model_name:
            Optional, The name of the model. If None, default is model id, for example,
            'sentence-transformers/msmarco-distilbert-base-tas-b'
        :type model_name: string
        :param model_format:
            Optional, the format of the model. Default is "TORCH_SCRIPT".
        :type model_format: string
        :param model_zip_file_path:
            Optional, path to the model zip file. Default is the zip file path used in save_as_pt or save_as_onnx
            depending on model_format. This zip file is used to compute model_content_size_in_bytes and
            model_content_hash_value.
        :type model_zip_file_path: string
        :param version_number:
            Optional, The version number of the model. Default is 1
        :type version_number: string
        :param embedding_dimension: Optional, the embedding dimension of the model. If None, get embedding_dimension
            from the pre-trained hugging-face model object.
        :type embedding_dimension: int
        :param pooling_mode: Optional, the pooling mode of the model. If None, get pooling_mode
            from the pre-trained hugging-face model object.
        :type pooling_mode: string
        :param normalize_result: Optional, whether to normalize the result of the model. If None, check from the pre-trained
        hugging-face model object.
        :type normalize_result: bool
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
        folder_path = self.folder_path
        config_json_file_path = os.path.join(folder_path, "config.json")
        if model_name is None:
            model_name = self.model_id

        # if user input model_type/embedding_dimension/pooling_mode, it will skip this step.
        model = DistilBertForQuestionAnswering.from_pretrained(self.model_id)
        model.save_pretrained(self.folder_path)

        
        # fill the empty fields
        if (
            model_type is None
            or embedding_dimension is None
            or pooling_mode is None
            or normalize_result is None
        ):
            try:
                if embedding_dimension is None:
                    embedding_dimension = model.config.dim
                if model_type is None:
                    model_type = "distilbert"
                if pooling_mode is None:
                    pooling_mode = "CLS"
                if normalize_result is None:
                    normalize_result = False
                
                # for str_idx, module in model._modules.items():
                    # print(f"=== idx {str_idx}, module name {module.__class__.__name__}, module {module}")
                    # if model_type is None and isinstance(module, Transformer):
                    #     model_type = module.auto_model.__class__.__name__
                    #     model_type = model_type.lower().rstrip("model")
                    # elif pooling_mode is None and isinstance(module, Pooling):
                    #     pooling_mode = module.get_pooling_mode_str().upper()
                    # elif normalize_result is None and isinstance(module, Normalize):
                    #     normalize_result = True
                    # TODO: Support 'Dense' module
                
            except Exception as e:
                raise Exception(
                    f"Raised exception while getting model data from pre-trained hugging-face model object: {e}"
                )

        # fill the description
        if description is None:
            readme_file_path = os.path.join(self.folder_path, "README.md")
            if os.path.exists(readme_file_path):
                try:
                    if verbose:
                        print("reading README.md file")
                    description = self._get_model_description_from_readme_file(
                        readme_file_path
                    )
                except Exception as e:
                    print(f"Cannot scrape model description from README.md file: {e}")
                    description = self._generate_default_model_description(
                        embedding_dimension
                    )
            else:
                print("Using default model description")
                description = "This is a question-answering model: it provides answers to a question and context."

        # dump the config.json file
        if all_config is None:
            if not os.path.exists(config_json_file_path):
                raise Exception(
                    str(
                        "Cannot find config.json in"
                        + config_json_file_path
                        + ". Please check the config.son file in the path."
                    )
                )
            try:
                with open(config_json_file_path) as f:
                    if verbose:
                        print("reading config file from: " + config_json_file_path)
                    config_content = json.load(f)
                    if all_config is None:
                        all_config = config_content
            except IOError:
                print(
                    "Cannot open in config.json file at ",
                    config_json_file_path,
                    ". Please check the config.json ",
                    "file in the path.",
                )

        model_config_content = {
            "name": model_name,
            "version": version_number,
            "description": description,
            "model_format": model_format,
            "model_task_type": "QUESTION_ANSWERING",
            "model_config": {
                "model_type": model_type,
                "embedding_dimension": embedding_dimension,
                "framework_type": "sentence_transformers",
                "pooling_mode": pooling_mode,
                "normalize_result": normalize_result,
                "all_config": json.dumps(all_config),
            },
        }
        
        # get model size and hash value
        if model_zip_file_path is None:
            model_zip_file_path = (
                self.torch_script_zip_file_path
                if model_format == "TORCH_SCRIPT"
                else self.onnx_zip_file_path
            )

            # model_zip_file_path = '/Users/faradawn/CS/opensearch-py-ml/opensearch_py_ml/ml_models/question-model-folder/distilbert-base-cased-distilled-squad.zip'

            if model_zip_file_path is None:
                print(
                    "The model configuration JSON file currently lacks the 'model_content_size_in_bytes' and 'model_content_hash_value' fields. You can include these fields by specifying the 'model_zip_file_path' parameter. Failure to do so may result in the model registration process encountering issues."
                )
            else:
                model_config_content["model_content_size_in_bytes"] = os.stat(
                    model_zip_file_path
                ).st_size
                model_config_content[
                    "model_content_hash_value"
                ] = _generate_model_content_hash_value(model_zip_file_path)

        if verbose:
            print("generating ml-commons_model_config.json file...\n")
            print(json.dumps(model_config_content, indent=4))

        model_config_file_path = os.path.join(
            folder_path, "ml-commons_model_config.json"
        )
        os.makedirs(os.path.dirname(model_config_file_path), exist_ok=True)
        with open(model_config_file_path, "w") as file:
            json.dump(model_config_content, file, indent=4)
        print(
            "ml-commons_model_config.json file is saved at : ", model_config_file_path
        )

        return model_config_file_path
    
    def test_traced_model(self, model_path, question, context):
        """
        Load a model from TorchScript and run inference on the question and text.
        
        """
        traced_model = torch.jit.load(model_path)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
        inputs = tokenizer(question, context, return_tensors="pt")

        # Start inference
        with torch.no_grad():
            outputs = traced_model(**inputs)

        # Get the most likely start and end positions
        answer_start_index = torch.argmax(outputs["start_logits"], dim=-1).item()
        answer_end_index = torch.argmax(outputs["end_logits"], dim=-1).item()

        # Extract the answer tokens and convert back to text
        predict_answer_tokens = inputs['input_ids'][0, answer_start_index : answer_end_index + 1]
        answer = tokenizer.decode(predict_answer_tokens)

        return answer
