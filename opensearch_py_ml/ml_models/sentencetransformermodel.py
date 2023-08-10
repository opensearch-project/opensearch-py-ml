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
import shutil
import subprocess
import time
from pathlib import Path
from typing import List
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from accelerate import Accelerator, notebook_launcher
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize, Pooling, Transformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainingArguments, get_linear_schedule_with_warmup
from transformers.convert_graph_to_onnx import convert


class SentenceTransformerModel:
    """
    Class for training, exporting and configuring the SentenceTransformers model.
    """

    DEFAULT_MODEL_ID = "sentence-transformers/msmarco-distilbert-base-tas-b"
    SYNTHETIC_QUERY_FOLDER = "synthetic_queries"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        folder_path: str = None,
        overwrite: bool = False,
    ) -> None:
        """
        Description: Initiate a sentence transformer model class object. The model id will be used to download
        pretrained model from the hugging-face and served as the default name for model files, and the folder_path
        will be the default location to store files generated in the following functions

        :param model_id: Optional, the huggingface mode id to download sentence transformer model,
            default model id: 'sentence-transformers/msmarco-distilbert-base-tas-b'
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
            os.getcwd(), "sentence_transformer_model_files"
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

    def train(
        self,
        read_path: str,
        overwrite: bool = False,
        output_model_name: str = None,
        zip_file_name: str = None,
        compute_environment: str = None,
        num_machines: int = 1,
        num_gpu: int = 0,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        batch_size: int = 32,
        verbose: bool = False,
        percentile: float = 95,
    ) -> None:
        """
        Read the synthetic queries and use it to fine tune/train (and save) a sentence transformer model.

        Parameters
        ----------
        :param read_path:
            required, path to the zipped file that contains generated queries, if None, raise exception.
            the zipped file should contain pickled file in list of dictionary format with key named as 'query',
            'probability' and 'passages'. For example: [{'query':q1,'probability': p1,'passages': pa1}, ...].
            'probability' is not required for training purpose
        :type read_path: string
        :param overwrite:
            optional, synthetic_queries/ folder in current directory is to store unzip queries files.
            Default to set overwrite as false and if the folder is not empty, raise exception to recommend users
            to either clean up folder or enable overwriting is True
        :type overwrite: bool
        :param output_model_name:
            the name of the trained custom model. If None, default as model_id + '.pt'
        :type output_model_name: string
        :param zip_file_name:
            Optional, file name for zip file. if None, default as model_id + '.zip'
        :type zip_file_name: string
        :param compute_environment:
            optional, compute environment type to run model, if None, default using `LOCAL_MACHINE`
        :type compute_environment: string
        :param num_machines:
            optional, number of machine to run model , if None, default using 1
        :type num_machines: int
        :param num_gpu:
            optional, number of gpus to run model , if None, default to 0. If number of gpus > 1, use HuggingFace
            accelerate to launch distributed training
        :param learning_rate:
            optional, learning rate to train model, default is 2e-5
        :type learning_rate: float
        :param num_epochs:
            optional, number of epochs to train model, default is 10
        :type num_epochs: int
        :param batch_size:
            optional, batch size for training, default is 32
        :type batch_size: int
        :param verbose:
            optional, use plotting to plot the training progress. Default as false
        :type verbose: bool
        :param percentile:
            we find the max length of {percentile}% of the documents. Default is 95%
            Since this length is measured in terms of words and not tokens we multiply it by 1.4 to approximate the fact
            that 1 word in the english vocabulary roughly translates to 1.3 to 1.5 tokens
        :type percentile: float

        Returns
        -------
        :return: no return value expected
        :rtype: None
        """

        query_df = self.read_queries(read_path, overwrite)

        train_examples = self.load_training_data(query_df)

        if num_gpu > 1:
            self.set_up_accelerate_config(
                compute_environment=compute_environment,
                num_machines=num_machines,
                num_processes=num_gpu,
                verbose=verbose,
            )

            if self.__is_notebook():
                # MPS needs to be only enabled for MACOS: https://pytorch.org/docs/master/notes/mps.html
                if platform.system() == "Darwin":
                    if not torch.backends.mps.is_available():
                        if not torch.backends.mps.is_built():
                            print(
                                "MPS not available because the current PyTorch install was not "
                                "built with MPS enabled."
                            )
                        else:
                            print(
                                "MPS not available because the current MacOS version is not 12.3+ "
                                "and/or you do not have an MPS-enabled device on this machine."
                            )
                        exit(1)  # Existing the script as the script will break anyway
                notebook_launcher(
                    self.train_model,
                    args=(
                        train_examples,
                        self.model_id,
                        output_model_name,
                        learning_rate,
                        num_epochs,
                        batch_size,
                        verbose,
                        num_gpu,
                        percentile,
                    ),
                    num_processes=num_gpu,
                )
            else:
                try:
                    subprocess.run(
                        [
                            "accelerate",
                            "launch",
                            self.train_model(
                                train_examples,
                                self.model_id,
                                output_model_name,
                                learning_rate,
                                num_epochs,
                                batch_size,
                                verbose,
                                num_gpu,
                                percentile,
                            ),
                        ],
                    )
                # TypeError: expected str, bytes or os.PathLike object, not TopLevelTracedModule happens after
                # running process.
                except TypeError:
                    self.zip_model(self.folder_path, output_model_name, zip_file_name)
                    return None

        else:  # Do not use accelerate when num_gpu is 1 or 0
            self.train_model(
                train_examples,
                self.model_id,
                output_model_name,
                learning_rate,
                num_epochs,
                batch_size,
                verbose,
                num_gpu,
                percentile,
            )

        self.zip_model(self.folder_path, output_model_name, zip_file_name)
        return None

    #    public step by step functions:
    def read_queries(self, read_path: str, overwrite: bool = False) -> pd.DataFrame:
        """
        Read the queries generated from the Synthetic Query Generator (SQG) model, unzip files to current directory
        within synthetic_queries/ folder, output as a dataframe

        :param read_path:
            required, path to the zipped file that contains generated queries
        :type read_path: string
        :param overwrite:
            optional, synthetic_queries/ folder in current directory is to store unzip queries files.
            Default to set overwrite as false and if the folder is not empty, raise exception to recommend users
            to either clean up folder or enable overwriting is True
        :type overwrite: bool
        :return: The dataframe of queries.
        :rtype: panda dataframe
        """
        # assign a local folder 'synthetic_queries/' to store the unzip file,
        # check if the folder contains sub-folders and files, remove and clean up the folder before unzip.
        # walk through the zip file and read the file paths into file_list
        unzip_path = os.path.join(self.folder_path, self.SYNTHETIC_QUERY_FOLDER)

        if os.path.exists(unzip_path):
            if len(os.listdir(unzip_path)) > 0:
                if overwrite:
                    for files in os.listdir(unzip_path):
                        sub_path = os.path.join(unzip_path, files)
                        if os.path.isfile(sub_path):
                            os.remove(sub_path)
                        else:
                            try:
                                shutil.rmtree(sub_path)
                            except OSError as err:
                                print(
                                    "Failed to delete files, please delete all files in "
                                    + str(unzip_path)
                                    + " "
                                    + str(err)
                                )
                else:
                    raise Exception(
                        "'synthetic_queries' folder is not empty, please clean up folder, or enable overwrite = "
                        + "True. Try again. Please check "
                        + unzip_path
                    )

        # appending all the file paths of synthetic query files in a list.
        file_list = []
        process = []
        with ZipFile(read_path, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

        for root, dirnames, filenames in os.walk(unzip_path):
            for filename in filenames:
                file_list.append(os.path.join(root, filename))

        # check empty zip file
        num_file = len(file_list)

        if num_file == 0:
            raise Exception(
                "Zipped file is empty. Please provide a zip file with synthetic queries."
            )

        for file_path in file_list:
            try:
                with open(file_path, "rb") as f:
                    print("Reading synthetic query file: " + file_path + "\n")
                    process.append(pickle.load(f))
            except IOError:
                print("Failed to open synthetic query file: " + file_path + "\n")

        # reading the files to get the probability, queries and passages
        prob = []
        query = []
        passages = []
        for j in range(0, num_file):
            for dict_str in process[j]:
                if "query" in dict_str.keys() and "passage" in dict_str.keys():
                    query.append(dict_str["query"])
                    passages.append(dict_str["passage"])
                    if "probability" in dict_str.keys():
                        prob.append(dict_str["probability"])  # language modeling score
                    else:
                        prob.append(
                            "-1"
                        )  # "-1" will serve as a label saying that the probability does not exist.

        df = pd.DataFrame(
            list(zip(prob, query, passages)), columns=["prob", "query", "passages"]
        )

        # dropping duplicate queries
        df = df.drop_duplicates(subset=["query"])
        # for removing the "QRY:" token if they exist in passages
        df["passages"] = df.apply(lambda x: self.__qryrem(x), axis=1)
        # shuffle data within dataframe
        df = df.sample(frac=1)
        return df

    def load_training_data(self, query_df) -> List[List[str]]:
        """
        Create input data for training the model

        :param query_df:
            required for loading training data
        :type query_df: pd.DataFrame
        :return: the list of train examples.
        :rtype: list
        """

        train_examples = []
        print("Loading training examples... \n")
        queries = list(query_df["query"])
        passages = list(query_df["passages"])
        for i in tqdm(range(len(query_df)), total=len(query_df)):
            train_examples.append([queries[i], passages[i]])
        return train_examples

    def train_model(
        self,
        train_examples: List[List[str]],
        model_id: str = None,
        output_model_name: str = None,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        batch_size: int = 32,
        verbose: bool = False,
        num_gpu: int = 0,
        percentile: float = 95,
    ):
        """
        Description:
        Takes in training data and a sentence transformer url to train a custom semantic search model

        :param train_examples:
            required, input for the sentence transformer model training
        :type train_examples: List of strings in another list
        :param model_id:
            [optional] the url to download sentence transformer model, if None,
            default as 'sentence-transformers/msmarco-distilbert-base-tas-b'
        :type model_id: string
        :param output_model_name:
            optional,the name of the trained custom model. If None, default as model_id + '.pt'
        :type output_model_name: string
        :param learning_rate:
            optional, learning rate to train model, default is 2e-5
        :type learning_rate: float
        :param num_epochs:
            optional, number of epochs to train model, default is 10
        :type num_epochs: int
        :param batch_size:
            optional, batch size for training, default is 32
        :type batch_size: int
        :param verbose:
            optional, use plotting to plot the training progress and printing more logs. Default as false
        :type verbose: bool
        :param num_gpu:
            Number of gpu will be used for training. Default 0
        :type num_gpu: int
        :param percentile:
            To save memory while training we truncate all passages beyond a certain max_length.
            Most middle-sized transformers have a max length limit of 512 tokens. However, certain corpora can
            have shorter documents. We find the word length of all documents, sort them in increasing order and
            take the max length of {percentile}% of the documents. Default is 95%
        :type percentile: float
        :return: the torch script format trained model.
        :rtype: .pt file
        """

        if model_id is None:
            model_id = "sentence-transformers/msmarco-distilbert-base-tas-b"
        if output_model_name is None:
            output_model_name = str(self.model_id.split("/")[-1] + ".pt")

        # declare variables before assignment for training
        corp_len = []

        # Load a model from HuggingFace
        model = SentenceTransformer(model_id)

        # Calculate the length of passages
        for i in range(len(train_examples)):
            corp_len.append(len(train_examples[i][1].split(" ")))

        # In the following, we find the max length of 95% of the documents (when sorted by increasing word length).
        # Since this length is measured in terms of words and not tokens we multiply it by 1.4 to approximate the
        # fact that 1 word in the english vocabulary roughly translates to 1.3 to 1.5 tokens. For instance the word
        # butterfly will be split by most tokenizers into butter and fly, but the word sun will be probably kept as it
        # is. Note that this ratio will be higher if the corpus is jargon heavy and/or domain specific.

        corp_max_tok_len = int(np.percentile(corp_len, percentile) * 1.4)
        model.tokenizer.model_max_length = corp_max_tok_len
        model.max_seq_length = corp_max_tok_len

        # use accelerator for training
        if num_gpu > 1:
            # the default_args are required for initializing train_dataloader,
            # but output_dir is not used in this function.
            default_args = {
                "output_dir": "~/",
                "evaluation_strategy": "steps",
                "num_train_epochs": num_epochs,
                "log_level": "error",
                "report_to": "none",
            }
            training_args = TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=1,
                fp16=False,
                **default_args,
            )

            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=training_args.per_device_train_batch_size,  # Trains with this batch size.
            )

            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=min(10000, 0.05 * len(train_dataloader)),
                num_training_steps=num_epochs * len(train_dataloader),
            )

            accelerator = Accelerator()
            model, optimizer, train_dataloader, scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, scheduler
            )
            print("Device using for training: ", accelerator.device)
            model.to(accelerator.device)
            init_time = time.time()
            total_loss = []

            if accelerator.process_index == 0:
                print("Start training with accelerator...\n")

                print(f"The number of training epochs per process are {num_epochs}\n")

                print(
                    f"The total number of steps per training epoch are {len(train_dataloader)}\n"
                )

            # The following training loop trains the sentence transformer model using the standard contrastive loss
            # with in-batch negatives. The particular contrastive loss that we use is the Multi-class N-pair Loss (
            # eq. 6, Sohn, NeurIPS 2016). In addition, we symmetrize the loss with respect to queries and passages (
            # as also used in OpenAI's CLIP model). The performance improves with the number of in-batch negatives
            # but larger batch sizes can lead to out of memory issues, so please use the batch-size judiciously.

            for epoch in range(num_epochs):
                print(
                    "Training epoch "
                    + str(epoch)
                    + " in process "
                    + str(accelerator.process_index)
                    + "...\n"
                )
                for step, batch in tqdm(
                    enumerate(train_dataloader), total=len(train_dataloader)
                ):
                    batch_q = batch[0]
                    batch_p = batch[1]

                    model = accelerator.unwrap_model(model)
                    out_q = accelerator.unwrap_model(model).tokenize(batch_q)
                    for key in out_q.keys():
                        out_q[key] = out_q[key].to(model.device)

                    out_p = accelerator.unwrap_model(model).tokenize(batch_p)
                    for key in out_p.keys():
                        out_p[key] = out_p[key].to(model.device)

                    Y = model(out_q)["sentence_embedding"]
                    X = model(out_p)["sentence_embedding"]

                    XY = torch.exp(torch.matmul(X, Y.T) / (X.shape[1]) ** 0.5)
                    num = torch.diagonal(XY)

                    den0 = torch.sum(XY, dim=0)
                    den1 = torch.sum(XY, dim=1)

                    batch_loss = -torch.sum(torch.log(num / den0)) - torch.sum(
                        torch.log(num / den1)
                    )

                    accelerator.backward(batch_loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    total_loss.append(batch_loss.item())

                    if verbose is True and not step % 500 and step != 0:
                        plt.plot(total_loss[::100])
                        plt.show()
            accelerator.wait_for_everyone()

        # When number of GPU is less than 2, we don't need to accelerate
        else:
            # identify if running on gpu or cpu
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            total_steps = (len(train_examples) // batch_size) * num_epochs
            steps_size = len(train_examples) // batch_size
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=max(10000, total_steps * 0.05),
                num_training_steps=total_steps,
            )

            loss = []
            init_time = time.time()

            print("Start training without accelerator...\n")
            print(f"The number of training epoch are {num_epochs}\n")
            print(
                f"The total number of steps per training epoch are {len(train_examples) // batch_size}\n"
            )

            # The following training loop trains the sentence transformer model using the standard contrastive loss
            # with in-batch negatives. The particular contrastive loss that we use is the Multi-class N-pair Loss (
            # eq. 6, Sohn, NeurIPS 2016). In addition, we symmetrize the loss with respect to queries and passages (
            # as also used in OpenAI's CLIP model). The performance improves with the number of in-batch negatives
            # but larger batch sizes can lead to out of memory issues, so please use the batch-size judiciously.

            for epoch in range(num_epochs):
                random.shuffle(train_examples)
                print("Training epoch " + str(epoch) + "...\n")
                for j in tqdm(range(steps_size), total=steps_size):
                    batch_q = []
                    batch_p = []
                    for example in train_examples[
                        j * batch_size : (j + 1) * batch_size
                    ]:
                        batch_q.append(example[0])
                        batch_p.append(example[1])

                    out_q = model.tokenize(batch_q)
                    for key in out_q.keys():
                        out_q[key] = out_q[key].to(device)
                    Y = model(out_q)["sentence_embedding"]

                    out = model.tokenize(batch_p)
                    for key in out.keys():
                        out[key] = out[key].to(device)
                    X = model(out)["sentence_embedding"]

                    XY = torch.exp(torch.matmul(X, Y.T) / (X.shape[1]) ** 0.5)
                    num = torch.diagonal(XY)

                    den0 = torch.sum(XY, dim=0)
                    den1 = torch.sum(XY, dim=1)

                    train_loss = -torch.sum(torch.log(num / den0)) - torch.sum(
                        torch.log(num / den1)
                    )
                    loss.append(train_loss.item())
                    train_loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if verbose is True and not j % 500 and j != 0:
                        plt.plot(loss[::100])
                        plt.show()

        # saving the pytorch model and the tokenizers.json file is saving at this step
        model.save(self.folder_path)
        device = "cpu"
        cpu_model = model.to(device)
        print(f"Total training time: {time.time() - init_time}\n")

        for key in out_q.keys():
            out_q[key] = out_q[key].to(device)

        traced_cpu = torch.jit.trace(
            cpu_model,
            (
                {
                    "input_ids": out_q["input_ids"],
                    "attention_mask": out_q["attention_mask"],
                }
            ),
            strict=False,
        )
        if verbose:
            print("Preparing model to save...\n")
        torch.jit.save(traced_cpu, os.path.join(self.folder_path, output_model_name))
        print("Model saved to path: " + self.folder_path + "\n")
        return traced_cpu

    def zip_model(
        self,
        model_path: str = None,
        model_name: str = None,
        zip_file_name: str = None,
        verbose: bool = False,
    ) -> None:
        """
        Description:
        zip the model file and its tokenizer.json file to prepare to upload to the Open Search cluster

        :param model_path:
            Optional, path to find the model file, if None, default as concatenate model_id and
            '.pt' file in current path
        :type model_path: string
        :param model_name:
            the name of the trained custom model. If None, default as concatenate model_id and '.pt'
        :type model_name: string
        :param zip_file_name: str =None
            Optional, file name for zip file. if None, default as concatenate model_id and '.zip'
        :type zip_file_name: string
        :param verbose:
            optional, use to print more logs. Default as false
        :type verbose: bool
        :return: no return value expected
        :rtype: None
        """
        if model_name is None:
            model_name = str(self.model_id.split("/")[-1] + ".pt")

        if model_path is None:
            model_path = os.path.join(self.folder_path, str(model_name))
        else:
            model_path = os.path.join(model_path, str(model_name))

        if verbose:
            print("model path is: ", model_path)

        if zip_file_name is None:
            zip_file_name = str(model_name + ".zip")

        zip_file_name_without_extension = zip_file_name.split(".")[0]

        if verbose:
            print("Zip file name without extension: ", zip_file_name_without_extension)

        tokenizer_json_path = os.path.join(self.folder_path, "tokenizer.json")
        print("tokenizer_json_path: ", tokenizer_json_path)

        zip_file_path = os.path.join(self.folder_path, zip_file_name)

        if not os.path.exists(tokenizer_json_path):
            raise Exception(
                "Cannot find tokenizer.json file, please check at "
                + tokenizer_json_path
            )
        if not os.path.exists(model_path):
            raise Exception(
                "Cannot find model in the model path , please check at " + model_path
            )

        # Create a ZipFile Object
        with ZipFile(zip_file_path, "w") as zipObj:
            zipObj.write(model_path, zip_file_name_without_extension + "/" + model_name)
            zipObj.write(
                tokenizer_json_path,
                zip_file_name_without_extension + "/" + "tokenizer.json",
            )
        print("zip file is saved to " + zip_file_path + "\n")

    def _fill_null_truncation_field(
        self,
        save_json_folder_path: str,
        max_length: int,
    ) -> None:
        """
        Description:
        Fill truncation field in tokenizer.json when it is null

        :param save_json_folder_path:
             path to save model json file, e.g, "home/save_pre_trained_model_json/")
        :type save_json_folder_path: string
        :param max_length:
             maximum sequence length for model
        :type max_length: int
        :return: no return value expected
        :rtype: None
        """
        tokenizer_file_path = os.path.join(save_json_folder_path, "tokenizer.json")
        with open(tokenizer_file_path) as user_file:
            parsed_json = json.load(user_file)
        if "truncation" not in parsed_json or parsed_json["truncation"] is None:
            parsed_json["truncation"] = {
                "direction": "Right",
                "max_length": max_length,
                "strategy": "LongestFirst",
                "stride": 0,
            }
            with open(tokenizer_file_path, "w") as file:
                json.dump(parsed_json, file, indent=2)

    def save_as_pt(
        self,
        sentences: [str],
        model_id="sentence-transformers/msmarco-distilbert-base-tas-b",
        model_name: str = None,
        save_json_folder_path: str = None,
        model_output_path: str = None,
        zip_file_name: str = None,
    ) -> str:
        """
        download sentence transformer model directly from huggingface, convert model to torch script format,
        zip the model file and its tokenizer.json file to prepare to upload to the Open Search cluster

        :param sentences:
            Required, for example  sentences = ['today is sunny']
        :type sentences: List of string [str]
        :param model_id:
            sentence transformer model id to download model from sentence transformers.
            default model_id = "sentence-transformers/msmarco-distilbert-base-tas-b"
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
        :return: model zip file path. The file path where the zip file is being saved
        :rtype: string
        """

        model = SentenceTransformer(model_id)

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
        model.save(save_json_folder_path)
        self._fill_null_truncation_field(
            save_json_folder_path, model.tokenizer.model_max_length
        )

        # convert to pt format will need to be in cpu,
        # set the device to cpu, convert its input_ids and attention_mask in cpu and save as .pt format
        device = torch.device("cpu")
        cpu_model = model.to(device)
        features = cpu_model.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        compiled_model = torch.jit.trace(
            cpu_model,
            (
                {
                    "input_ids": features["input_ids"],
                    "attention_mask": features["attention_mask"],
                }
            ),
            strict=False,
        )
        torch.jit.save(compiled_model, model_path)
        print("model file is saved to ", model_path)

        # zip model file along with tokenizer.json as output
        with ZipFile(str(zip_file_path), "w") as zipObj:
            zipObj.write(
                model_path,
                arcname=str(model_name),
            )
            zipObj.write(
                os.path.join(save_json_folder_path, "tokenizer.json"),
                arcname="tokenizer.json",
            )
        print("zip file is saved to ", zip_file_path, "\n")
        return zip_file_path

    def save_as_onnx(
        self,
        model_id="sentence-transformers/msmarco-distilbert-base-tas-b",
        model_name: str = None,
        save_json_folder_path: str = None,
        model_output_path: str = None,
        zip_file_name: str = None,
    ) -> str:
        """
        download sentence transformer model directly from huggingface, convert model to onnx format,
        zip the model file and its tokenizer.json file to prepare to upload to the Open Search cluster

        :param model_id:
            sentence transformer model id to download model from sentence transformers.
            default model_id = "sentence-transformers/msmarco-distilbert-base-tas-b"
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
        :return: model zip file path. The file path where the zip file is being saved
        :rtype: string
        """

        model = SentenceTransformer(model_id)

        if model_name is None:
            model_name = str(model_id.split("/")[-1] + ".onnx")

        model_path = os.path.join(self.folder_path, "onnx", model_name)

        if save_json_folder_path is None:
            save_json_folder_path = self.folder_path

        if model_output_path is None:
            model_output_path = self.folder_path

        if zip_file_name is None:
            zip_file_name = str(model_id.split("/")[-1] + ".zip")

        zip_file_path = os.path.join(model_output_path, zip_file_name)

        # save tokenizer.json in output_path
        model.save(save_json_folder_path)
        self._fill_null_truncation_field(
            save_json_folder_path, model.tokenizer.model_max_length
        )

        convert(
            framework="pt",
            model=model_id,
            output=Path(model_path),
            opset=15,
        )

        # zip model file along with tokenizer.json as output
        with ZipFile(str(zip_file_path), "w") as zipObj:
            zipObj.write(
                model_path,
                arcname=str(model_name),
            )
            zipObj.write(
                os.path.join(save_json_folder_path, "tokenizer.json"),
                arcname="tokenizer.json",
            )
        print("zip file is saved to ", zip_file_path, "\n")
        return zip_file_path

    def set_up_accelerate_config(
        self,
        compute_environment: str = None,
        num_machines: int = 1,
        num_processes: int = None,
        verbose: bool = False,
    ) -> None:
        """
        get default config setting based on the number of GPU on the machine
        if users require other configs, users can run !acclerate config for more options

        :param compute_environment:
            optional, compute environment type to run model, if None, default using 'LOCAL_MACHINE'
        :type compute_environment: string
        :param num_machines:
            optional, number of machine to run model , if None, default using 1
        :type num_machines: int
        :param num_processes:
            optional, number of processes to run model, if None, default to check how many gpus are available and
            use all. if no gpu is available, use cpu
        :type num_processes: int
        :param verbose:
            optional, use printing more logs. Default as false
        :type verbose: bool
        :return: no return value expected
        :rtype: None
        """

        if compute_environment is None or compute_environment == 0:
            compute_environment = "LOCAL_MACHINE"
        else:
            subprocess.run("accelerate config")
            return

        hf_cache_home = os.path.expanduser(
            os.getenv(
                "HF_HOME",
                os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"),
            )
        )

        cache_dir = os.path.join(hf_cache_home, "accelerate")

        file_path = os.path.join(cache_dir, "default_config.yaml")
        use_cpu = False
        if verbose:
            print("generated config file: at " + file_path + "\n")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if num_processes is None:
            if torch.cuda.is_available():
                num_processes = torch.cuda.device_count()
            else:
                num_processes = 1
                use_cpu = True

        model_config_content = [
            {
                "compute_environment": compute_environment,
                "deepspeed_config": {
                    "gradient_accumulation_steps": 1,
                    "offload_optimizer_device": "none",
                    "offload_param_device": "none",
                    "zero3_init_flag": False,
                    "zero_stage": 2,
                },
                "distributed_type": "DEEPSPEED",
                "downcast_bf16": "no",
                "fsdp_config": {},
                "machine_rank": 0,
                "main_process_ip": None,
                "main_process_port": None,
                "main_training_function": "main",
                "mixed_precision": "no",
                "num_machines": num_machines,
                "num_processes": num_processes,
                "use_cpu": use_cpu,
            }
        ]

        if verbose:
            print("Printing model config content: \n")
            print(model_config_content)

        try:
            with open(file_path, "w") as file:
                if verbose:
                    print(
                        "generating config file for ml common upload: "
                        + file_path
                        + "\n"
                    )
                yaml.dump(model_config_content, file)
        except IOError:
            print(
                "Failed to open config file for ml common upload: " + file_path + "\n"
            )

    def make_model_config_json(
        self,
        model_name: str = None,
        version_number: str = 1,
        model_format: str = "TORCH_SCRIPT",
        embedding_dimension: int = None,
        pooling_mode: str = None,
        normalize_result: bool = None,
        all_config: str = None,
        model_type: str = None,
        verbose: bool = False,
    ) -> str:
        """
        parse from config.json file of pre-trained hugging-face model to generate a ml-commons_model_config.json file. If all required
        fields are given by users, use the given parameters and will skip reading the config.json
        :param model_name:
            Optional, The name of the model. If None, default is model id, for example,
            'sentence-transformers/msmarco-distilbert-base-tas-b'
        :type model_name: string
        :param model_format:
            Optional, The format of the model. Default is "TORCH_SCRIPT".
        :type model_format: string
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
        model = SentenceTransformer(self.model_id)
        if (
            model_type is None
            or embedding_dimension is None
            or pooling_mode is None
            or normalize_result is None
        ):
            try:
                if embedding_dimension is None:
                    embedding_dimension = model.get_sentence_embedding_dimension()

                for str_idx, module in model._modules.items():
                    if model_type is None and isinstance(module, Transformer):
                        model_type = module.auto_model.__class__.__name__
                        model_type = model_type.lower().rstrip("model")
                    elif pooling_mode is None and isinstance(module, Pooling):
                        pooling_mode = module.get_pooling_mode_str().upper()
                    elif normalize_result is None and isinstance(module, Normalize):
                        normalize_result = True
                    # TODO: Support 'Dense' module
                if normalize_result is None:
                    normalize_result = False
            except Exception as e:
                raise Exception(
                    f"Raised exception while getting model data from pre-trained hugging-face model object: {e}"
                )

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
            "model_format": model_format,
            "model_task_type": "TEXT_EMBEDDING",
            "model_config": {
                "model_type": model_type,
                "embedding_dimension": embedding_dimension,
                "framework_type": "sentence_transformers",
                "pooling_mode": pooling_mode,
                "normalize_result": normalize_result,
                "all_config": json.dumps(all_config),
            },
        }

        if verbose:
            print("generating ml-commons_model_config.json file...\n")
            print(model_config_content)

        model_config_file_path = os.path.join(
            folder_path, "ml-commons_model_config.json"
        )
        os.makedirs(os.path.dirname(model_config_file_path), exist_ok=True)
        with open(model_config_file_path, "w") as file:
            json.dump(model_config_content, file)
        print(
            "ml-commons_model_config.json file is saved at : ", model_config_file_path
        )

        return model_config_file_path

    # private methods
    def __qryrem(self, x):
        # for removing the "QRY:" token if they exist in passages
        if "QRY:" in x.passages and "<|startoftext|>" in x.passages:
            y = x.passages.split(" QRY:")[0].split("<|startoftext|>")
            return y[1]
        else:
            return x

    def __is_notebook(self) -> bool:
        try:
            shell = get_ipython().__class__.__name__
            # Jupyter notebook or qtconsole
            if shell == "ZMQInteractiveShell":
                return True
                # Terminal running IPython
            elif shell == "TerminalInteractiveShell":
                return False
            else:
                return False
        except NameError:
            # Probably standard Python interpreter
            return False
