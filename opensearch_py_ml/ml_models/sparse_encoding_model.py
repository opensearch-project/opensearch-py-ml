# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.
import json
import os
from zipfile import ZipFile

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from opensearch_py_ml.ml_commons.ml_common_utils import (
    SPARSE_ENCODING_FUNCTION_NAME,
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
    description = "This is a neural sparse model: It maps sentences & paragraphs to sparse vector space."
    return description


class SparseEncodingModel(SparseModel):
    """
    Class for  exporting and configuring the NeuralSparseV2Model model.
    """

    DEFAULT_MODEL_ID = "opensearch-project/opensearch-neural-sparse-encoding-v1"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        folder_path: str = None,
        overwrite: bool = False,
    ) -> None:

        super().__init__(model_id, folder_path, overwrite)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.backbone_model = AutoModelForMaskedLM.from_pretrained(model_id)
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
        Download sentence transformer model directly from huggingface, convert model to torch script format,
        zip the model file and its tokenizer.json file to prepare to upload to the Open Search cluster

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

        model = NeuralSparseModel(self.backbone_model, self.tokenizer)

        # save tokenizer.json in save_json_folder_name
        self.tokenizer.save_pretrained(save_json_folder_path)

        super()._fill_null_truncation_field(
            save_json_folder_path, self.tokenizer.model_max_length
        )

        # convert to pt format will need to be in cpu,
        # set the device to cpu, convert its input_ids and attention_mask in cpu and save as .pt format
        device = torch.device("cpu")
        cpu_model = model.to(device)

        features = self.tokenizer(
            sentences,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        ).to(device)

        compiled_model = torch.jit.trace(cpu_model, dict(features), strict=False)
        torch.jit.save(compiled_model, model_path)
        print("model file is saved to ", model_path)

        # zip model file along with self.tokenizer.json (and license file) as output
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
            "function_name": SPARSE_ENCODING_FUNCTION_NAME,
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
        if description is not None:
            model_config_content["description"] = description

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

    def get_backbone_model(self):
        if self.backbone_model is not None:
            return self.backbone_model
        else:
            return AutoModelForMaskedLM.from_pretrained(self.model_id)

    def get_model(self):
        return NeuralSparseModel(self.get_backbone_model(), self.get_tokenizer())

    def save(self, path):
        backbone_model = self.get_backbone_model()
        tokenizer = self.get_tokenizer()
        backbone_model.save_pretrained(path)
        tokenizer.save_pretrained(path)

    def post_process(self):
        pass

    def pre_process(self):
        pass

    def get_tokenizer(self):
        if self.tokenizer is not None:
            return self.tokenizer
        else:
            return AutoTokenizer.from_pretrained(self.model_id)

    def process_sparse_encoding(self, queries):
        return self.get_model().process_sparse_encoding(queries)

    def init_tokenizer(self, model_id=None):
        if model_id is None:
            model_id = self.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)


INPUT_ID_KEY = "input_ids"
ATTENTION_MASK_KEY = "attention_mask"
OUTPUT_KEY = "output"


class NeuralSparseModel(torch.nn.Module):
    """
    A PyTorch module for transforming input text to sparse vector representation using a pre-trained internal BERT model.
    This class encapsulates the BERT model and provides methods to process text queries into sparse vectors,
    which are easier to handle in sparse data scenarios such as information retrieval.
    """

    def __init__(self, backbone_model, tokenizer=None):
        super().__init__()
        self.backbone_model = backbone_model
        if tokenizer is not None:
            self.tokenizer = tokenizer
            self.special_token_ids = [
                tokenizer.vocab[token]
                for token in tokenizer.special_tokens_map.values()
            ]
            self.id_to_token = ["" for _ in range(len(tokenizer.vocab))]
            for token, idx in tokenizer.vocab.items():
                self.id_to_token[idx] = token

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = self.backbone_model(
            input_ids=input[INPUT_ID_KEY], attention_mask=input[ATTENTION_MASK_KEY]
        )[0]
        values, _ = torch.max(result * input[ATTENTION_MASK_KEY].unsqueeze(-1), dim=1)
        values = torch.log(1 + torch.relu(values))
        values[:, self.special_token_ids] = 0
        return {OUTPUT_KEY: values}

    def get_sparse_vector(self, feature):
        output = self.forward(feature)
        values = output[OUTPUT_KEY]
        return values

    def transform_sparse_vector_to_dict(self, sparse_vector):
        all_sparse_dicts = []
        for vector in sparse_vector:
            tokens = [
                self.id_to_token[i]
                for i in torch.nonzero(vector, as_tuple=True)[0].tolist()
            ]
            sparse_dict = {
                token: weight.item()
                for token, weight in zip(
                    tokens, vector[torch.nonzero(vector, as_tuple=True)]
                )
            }
            all_sparse_dicts.append(sparse_dict)
        return all_sparse_dicts

    def process_sparse_encoding(self, queries):
        features = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        sparse_vector = self.get_sparse_vector(features)
        sparse_dict = self.transform_sparse_vector_to_dict(sparse_vector)
        return sparse_dict
