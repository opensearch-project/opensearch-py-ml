# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
from typing import List
from zipfile import ZipFile

import torch
from torch import nn
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel

from opensearch_py_ml.ml_commons.ml_common_utils import (
    _generate_model_content_hash_value,
)
from opensearch_py_ml.ml_models.base_models import BaseUploadModel

DEFAULT_MODEL_ID = "opensearch-project/opensearch-semantic-highlighter-v1"


class TraceableBertTaggerForSentenceExtractionWithBackoff(BertPreTrainedModel):
    """
    A torch.jit-compatible version of the sentence highlighter model.
    For inference only. Classification logic includes backoff rule
    with minimum confidence alpha=0.05.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config).from_pretrained(
            "bert-base-uncased", torchscript=True
        )
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

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        @torch.jit.script_if_tracing
        def _get_agg_output(ids, sequence_output):
            max_sentences = torch.max(ids) + 1
            d_model = sequence_output.shape[-1]

            agg_output = []
            global_offset_per_item = []
            num_sentences_per_item = []
            for i, sen_ids in enumerate(ids):
                out = []
                local_sen_ids = sen_ids.clone()
                mask = local_sen_ids != -100
                offset = local_sen_ids[mask].min()
                global_offset_per_item.append(offset)
                local_sen_ids[mask] = local_sen_ids[mask] - offset
                num_sentences = torch.max(local_sen_ids) + 1
                num_sentences_per_item.append(num_sentences)

                for j in range(int(num_sentences)):
                    out.append(
                        sequence_output[i, local_sen_ids == j].mean(
                            dim=-2, keepdim=True
                        )
                    )
                padding = torch.zeros((int(max_sentences - num_sentences), d_model))
                out.append(padding)
                out = torch.cat(out, dim=0)
                agg_output.append(out)
            agg_output = torch.stack(agg_output)
            return (agg_output, global_offset_per_item, num_sentences_per_item)

        agg_output, global_offset_per_item, num_sentences_per_item = _get_agg_output(
            sentence_ids, sequence_output
        )
        logits = self.classifier(agg_output)
        probs = torch.softmax(logits, dim=-1)
        pos_probs = probs[:, :, 1]

        @torch.jit.script_if_tracing
        def _get_sentence_preds(
            pos_probs,
            global_offset_per_item: List[torch.Tensor],
            num_sentences_per_item: List[torch.Tensor],
            threshold: float = 0.5,
            alpha: float = 0.05,
        ):
            sentences_preds = []
            for pprob, offset, num_sentences in zip(
                pos_probs, global_offset_per_item, num_sentences_per_item
            ):
                relevant_probs = pprob[:num_sentences]
                relevant_preds = (relevant_probs >= threshold).int()

                if relevant_preds.sum() == 0:  # backoff logic
                    max_prob_idx = relevant_probs.argmax()
                    max_prob = relevant_probs[max_prob_idx].item()
                    if max_prob >= alpha:
                        relevant_preds[max_prob_idx] = 1

                (indices,) = torch.where(relevant_preds == 1)
                indices += offset
                sentences_preds += [indices]

            return sentences_preds

        sentences_preds = _get_sentence_preds(
            pos_probs, global_offset_per_item, num_sentences_per_item
        )

        return tuple(sentences_preds)


class SemanticHighlighterModel(BaseUploadModel):
    """
    A model class for the OpenSearch semantic highlighter that identifies relevant sentences in a document
    given a query.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        folder_path: str = None,
        overwrite: bool = False,
    ) -> None:
        """
        Initialize a SemanticHighlighterModel instance.

        Parameters
        ----------
        model_id : str, optional
            The model ID to use, by default "opensearch-project/opensearch-semantic-highlighter-v1"
        folder_path : str, optional
            Path to save model files, by default None
        overwrite : bool, optional
            Whether to overwrite existing files, by default False
        """
        if folder_path is None:
            folder_path = "semantic-highlighter/"

        super().__init__(
            model_id=model_id, folder_path=folder_path, overwrite=overwrite
        )
        self.torch_script_zip_file_path = None

    def save_as_pt(
        self,
        example_inputs: dict,
        model_id: str = DEFAULT_MODEL_ID,
        model_name: str = None,
        save_json_folder_path: str = None,
        model_output_path: str = None,
        zip_file_name: str = None,
        add_apache_license: bool = True,
    ) -> str:
        """
        Convert the semantic highlighter model to TorchScript format and prepare it for upload.

        Parameters
        ----------
        example_inputs : dict
            Example inputs for tracing with keys: input_ids, attention_mask, token_type_ids, sentence_ids
        model_id : str, optional
            Model ID to use, by default DEFAULT_MODEL_ID
        model_name : str, optional
            Name for the traced model file, by default None
        save_json_folder_path : str, optional
            Path to save config files, by default None
        model_output_path : str, optional
            Path to save the traced model, by default None
        zip_file_name : str, optional
            Name for the zip file, by default None
        add_apache_license : bool, optional
            Whether to add Apache license, by default True

        Returns
        -------
        str
            Path to the created zip file
        """
        if model_name is None:
            model_name = str(model_id.split("/")[-1] + ".pt")

        # Create output directories
        os.makedirs(self.folder_path, exist_ok=True)
        model_path = os.path.join(self.folder_path, model_name)

        if save_json_folder_path is None:
            save_json_folder_path = self.folder_path

        if model_output_path is None:
            model_output_path = self.folder_path

        if zip_file_name is None:
            zip_file_name = str(model_id.split("/")[-1] + ".zip")

        zip_file_path = os.path.join(model_output_path, zip_file_name)

        # Initialize and trace the model
        model = TraceableBertTaggerForSentenceExtractionWithBackoff.from_pretrained(
            model_id
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        device = torch.device("cpu")
        model = model.to(device)

        # Save tokenizer files
        tokenizer_path = os.path.join(self.folder_path, "tokenizer")
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Tokenizer files saved to {tokenizer_path}")

        # Move inputs to CPU
        for k, v in example_inputs.items():
            example_inputs[k] = v.to(device)

        # Trace the model
        traced_model = torch.jit.trace(
            model,
            (
                example_inputs["input_ids"],
                example_inputs["attention_mask"],
                example_inputs["token_type_ids"],
                example_inputs["sentence_ids"],
            ),
        )

        # Save the traced model
        torch.jit.save(traced_model, model_path)
        print(f"Model file saved to {model_path}")

        # Create zip file with model, tokenizer, and config
        with ZipFile(str(zip_file_path), "w") as zipObj:
            # Add model file
            zipObj.write(model_path, arcname=str(model_name))

            # Add tokenizer files directly to root
            for file in os.listdir(tokenizer_path):
                file_path = os.path.join(tokenizer_path, file)
                zipObj.write(file_path, arcname=file)

        if add_apache_license:
            super()._add_apache_license_to_model_zip_file(zip_file_path)

        self.torch_script_zip_file_path = zip_file_path
        print(f"Zip file saved to {zip_file_path}")
        return zip_file_path

    def save_as_onnx(
        self,
        example_inputs: dict,
        model_id: str = DEFAULT_MODEL_ID,
        model_name: str = None,
        save_json_folder_path: str = None,
        model_output_path: str = None,
        zip_file_name: str = None,
        add_apache_license: bool = True,
    ) -> str:
        """
        ONNX format is not supported for semantic highlighter models.
        This method is implemented to satisfy the BaseUploadModel interface.

        Raises
        ------
        NotImplementedError
            Always raises this error as ONNX format is not supported.
        """
        raise NotImplementedError(
            "ONNX format is not supported for semantic highlighter models"
        )

    def make_model_config_json(
        self,
        model_name: str = None,
        version_number: str = "1.0.0",
        model_format: str = "TORCH_SCRIPT",
        description: str = None,
        model_zip_file_path: str = None,
    ) -> str:
        """
        Create the model configuration file for OpenSearch.

        Parameters
        ----------
        model_name : str, optional
            Name of the model, by default None
        version_number : str, optional
            Version of the model, by default "1.0.0"
        model_format : str, optional
            Format of the model, by default "TORCH_SCRIPT"
        description : str, optional
            Model description, by default None
        model_zip_file_path : str, optional
            Path to the model zip file, by default None

        Returns
        -------
        str
            Path to the created config file
        """
        if model_name is None:
            model_name = self.model_id

        model_config_content = {
            "name": model_name,
            "version": version_number,
            "model_format": model_format,
            "function_name": "QUESTION_ANSWERING",
            "description": (
                description
                if description
                else "A semantic highlighter model that identifies relevant sentences in a document given a query."
            ),
            "model_config": {
                "model_type": "sentence_highlighting",
                "framework_type": "huggingface_transformers",
            },
        }

        if model_zip_file_path is None:
            model_zip_file_path = self.torch_script_zip_file_path

        if model_zip_file_path:
            model_config_content["model_content_size_in_bytes"] = os.stat(
                model_zip_file_path
            ).st_size
            model_config_content["model_content_hash_value"] = (
                _generate_model_content_hash_value(model_zip_file_path)
            )

        model_config_file_path = os.path.join(
            self.folder_path, "ml-commons_model_config.json"
        )
        os.makedirs(os.path.dirname(model_config_file_path), exist_ok=True)

        with open(model_config_file_path, "w") as file:
            json.dump(model_config_content, file, indent=4)

        print(f"Model config file saved at: {model_config_file_path}")
        return model_config_file_path
