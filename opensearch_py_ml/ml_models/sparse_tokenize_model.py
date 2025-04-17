# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.
import json
import os
from zipfile import ZipFile

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from opensearch_py_ml.ml_commons.ml_common_utils import (
    SPARSE_TOKENIZE_FUNCTION_NAME,
    _generate_model_content_hash_value,
)
from opensearch_py_ml.ml_models.base_models import SparseModel


def _generate_default_model_description() -> str:
    """
    Generate default model description

    :return: Description of the model
    :rtype: string
    """
    print(
        "Using default description (You can overwrite this by specifying description parameter in "
        "make_model_config_json function"
    )
    description = "This is a neural sparse tokenizer model: It tokenize input sentence into tokens and assign pre-defined weight from IDF to each. It serves only in query."
    return description


class SparseTokenizeModel(SparseModel):
    """
    Class for  exporting and configuring the neural sparse tokenizer model.
    """

    DEFAULT_MODEL_ID = (
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    )

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        folder_path: str = None,
        overwrite: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(model_id, folder_path, overwrite)
        self.function_name = SPARSE_TOKENIZE_FUNCTION_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.special_token_ids = [
            self.tokenizer.vocab[token]
            for token in self.tokenizer.special_tokens_map.values()
        ]
        local_cached_path = hf_hub_download(repo_id=model_id, filename="idf.json")
        with open(local_cached_path) as f:
            self.idf = json.load(f)

        default_folder_path = os.path.join(
            os.getcwd(), "opensearch_neural_sparse_model_files"
        )
        if folder_path is None:
            self.folder_path = default_folder_path
        else:
            self.folder_path = folder_path

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
        model_id=DEFAULT_MODEL_ID,
        model_name: str = None,
        save_json_folder_path: str = None,
        model_output_path: str = None,
        zip_file_name: str = None,
        add_apache_license: bool = True,
    ) -> str:
        """
        Download sparse encoding model directly from huggingface, convert model to torch script format,
        zip the model file and its tokenizer.json file to prepare to upload to the OpenSearch cluster

        :param sentences:
            Required, for example  sentences = ['today is sunny']
        :type sentences: List of string [str]
        :param model_id:
             model id to download model from a sparse encoding model.
            default model_id = "opensearch-project/opensearch-neural-sparse-encoding-v1"
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
            Optional, whether to add Apache-2.0 license file to model zip file
        :type add_apache_license: string
        :return: model zip file path. The file path where the zip file is being saved
        :rtype: string
        """
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

        # save tokenizer.json in save_json_folder_name
        self.tokenizer.save_pretrained(save_json_folder_path)

        # save idf.json
        os.makedirs(model_path, exist_ok=True)
        idf_file_path = os.path.join(model_path, "idf.json")
        with open(idf_file_path, "w") as f:
            json.dump(self.idf, f)

        # zip model file along with self.tokenizer.json (and license file) as output
        with ZipFile(str(zip_file_path), "w") as zipObj:
            zipObj.write(
                idf_file_path,
                arcname="idf.json",
            )
            zipObj.write(
                os.path.join(save_json_folder_path, "tokenizer.json"),
                arcname="tokenizer.json",
            )
        if add_apache_license:
            super()._add_apache_license_to_model_zip_file(zip_file_path)

        self.torch_script_zip_file_path = zip_file_path
        print("zip file is saved to ", zip_file_path, "\n")
        return zip_file_path

    def save_as_onnx(self, *args, **kwargs):
        raise NotImplementedError

    def make_model_config_json(
        self,
        model_name: str = None,
        version_number: str = "1.0.0",
        model_format: str = "TORCH_SCRIPT",
        description: str = None,
        model_zip_file_path: str = None,
    ) -> str:
        folder_path = self.folder_path
        if model_name is None:
            model_name = self.model_id

        model_config_content = {
            "name": model_name,
            "version": version_number,
            "model_format": model_format,
            "function_name": SPARSE_TOKENIZE_FUNCTION_NAME,
        }
        if model_zip_file_path is None:
            model_zip_file_path = (
                self.torch_script_zip_file_path
                if model_format == "TORCH_SCRIPT"
                else self.onnx_zip_file_path
            )
            if model_zip_file_path is None:
                raise Exception(
                    "The model configuration JSON file currently lacks the 'model_content_size_in_bytes' and "
                    "'model_content_hash_value' fields. You can include these fields by specifying the "
                    "'model_zip_file_path' parameter. Failure to do so may result in the model registration process "
                    "encountering issues."
                )
            else:
                model_config_content["model_content_size_in_bytes"] = os.stat(
                    model_zip_file_path
                ).st_size
                model_config_content["model_content_hash_value"] = (
                    _generate_model_content_hash_value(model_zip_file_path)
                )
        if description is not None and description.strip() != "":
            model_config_content["description"] = description
        else:
            model_config_content["description"] = _generate_default_model_description()

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

    def process_sparse_encoding(self, queries):
        input_ids = self.tokenizer(queries, padding=True, truncation=True)["input_ids"]

        all_sparse_dicts = []
        for input_id in input_ids:
            sparse_dict = {
                self.tokenizer._convert_id_to_token(token_id): self.idf.get(
                    self.tokenizer._convert_id_to_token(token_id), 1
                )
                for token_id in input_id
                if token_id not in self.special_token_ids
            }
            all_sparse_dicts.append(sparse_dict)

        return all_sparse_dicts

    def save(self, path):
        self.tokenizer.save_pretrained(path)
        idf_file_path = os.path.join(path, "idf.json")
        with open(idf_file_path, "w") as f:
            json.dump(self.idf, f)
