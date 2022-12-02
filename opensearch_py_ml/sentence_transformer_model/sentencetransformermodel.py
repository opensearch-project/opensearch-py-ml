# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
import pickle
import random
import shutil
import subprocess
import time
from typing import List
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from accelerate import Accelerator, notebook_launcher
from sentence_transformers import InputExample, SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainingArguments, get_linear_schedule_with_warmup


class SentenceTransformerModel:
    def __init__(
        self, model_id: str = None, folder_path: str = None, overwrite: bool = False
    ) -> None:
        """
        Description:
        Initiate a sentence transformer model object. The model id will be used tp download pretrained model from the
        hugging-face and served as the default name for model files, and the folder_path will be the default location
        to store files generated in the following functions.

        Parameters
        ----------
        model_id: str = None
             Optional, the huggingface mode id to download sentence transformer model,
             if None, default as 'sentence-transformers/msmarco-distilbert-base-tas-b'
        folder_path: str = None
             Optional, the path of the folder to save output files, such as queries, pre-trained model, after-trained
             custom model and configuration files. if None, default as "/model_files/" under the current work directory.
        overwrite: bool = False
             Optional,  choose to overwrite the folder at folder path. Default as false. So when training different
             sentence transformer models,it's recommended to give designated folder path per model training. But if the
             training process get interrupted in between, users can choose overwrite = True to restart the process.
        Returns
        -------
        None
        """
        default_folder_path = os.path.join(os.getcwd() + "/model_files")

        # check folder exist in default_folder_path
        if os.path.exists(default_folder_path) and not overwrite:
            print(
                "To prevent overwritten, please enter a different folder path or delete the 'model_files' folder or enable overwrite = True"
            )
            raise Exception(
                str(
                    "The default folder path already exists at : " + default_folder_path
                )
            )

        if folder_path is None:
            self.folder_path = default_folder_path
        else:
            self.folder_path = folder_path
        if model_id is None:
            self.model_id = "sentence-transformers/msmarco-distilbert-base-tas-b"
        else:
            self.model_id = model_id

    def train(
        self,
        read_path: str,
        overwrite: bool = False,
        output_model_name: str = None,
        output_model_path: str = None,
        zip_file_name: str = None,
        use_accelerate: bool = False,
        compute_environment: str = None,
        num_machines: int = 1,
        num_processes: int = None,
        learning_rate: float = 2e-5,
        num_epochs: int = 20,
        verbose: bool = False,
    ) -> None:
        """
        Read the synthetic queries and use it to fine tune/train (and save) a sentence transformer model.

        Parameters
        ----------
        read_path: str
            required, path to the zipped file that contains generated queries, if None, raise exception.
            the zipped file should contain pickled file in list of dictionary format with key named as 'query',
            'probability' and 'passages'. For example: [{'query':q1,'probability': p1,'passages': pa1}, ...].
            'probability' is not required for training purpose.
        overwrite: bool
            optional, synthetic_queries/ folder in current directory is to store unzip queries files.
            Default to set overwrite as false and if the folder is not empty, raise exception to recommend users
            to either clean up folder or enable overwriting is True.
        output_model_path: str=None
            the path to store trained custom model. If None, default as current folder path
        output_model_name: str=None
            the name of the trained custom model. If None, default as model_id + '.pt'
        zip_file_name: str =None
            Optional, file name for zip file. if None, default as model_id + '.zip'
        use_accelerate: bool = False,
            Optional, use accelerate to fine tune model. Default as false to not use accelerator to fine tune model.
            If there are multiple gpus available in the machine, it's recommended to use accelerate with num_processor>1
            to speeed up the training progress. If use accelerator to train model, run auto setup accelerate confi and
            launch train_model function with the number of processors provided by users if NOT use accelerator,
            trigger train_model function with default setting
        compute_environment: str
            optional, compute environment type to run model, if None, default using `LOCAL_MACHINE`
        num_machines: int
            optional, number of machine to run model , if None, default using 1
        num_processes: int
            optional, number of processors to run model , if None, default using 1
        learning_rate: float
            optional, learning rate to train model, default is 2e-5
        num_epochs: int
            optional, number of epochs to train model, default is 20
        verbose: bool
            optional, use plotting to plot the training progress. Default as false.

        Returns
        -------
        None
        """

        query_df = self.read_queries(read_path, overwrite)

        train_examples = self.load_sentence_transformer_example(
            query_df, use_accelerate
        )

        if use_accelerate is True and num_processes != 0:

            self.set_up_accelerate_config(
                compute_environment=compute_environment,
                num_machines=num_machines,
                num_processes=num_processes,
            )

            if self.__is_notebook():
                notebook_launcher(
                    self.train_model,
                    args=(
                        train_examples,
                        self.model_id,
                        output_model_path,
                        output_model_name,
                        use_accelerate,
                        learning_rate,
                        num_epochs,
                        verbose,
                    ),
                    num_processes=num_processes,
                )
            else:
                subprocess.run(
                    [
                        "accelerate launch self.train_model(train_examples, self.model_id, output_model_path, output_model_name,use_accelerate,learning_rate,num_epochs,verbose)"
                    ]
                )
        else:
            self.train_model(
                train_examples,
                self.model_id,
                output_model_path,
                output_model_name,
                use_accelerate,
                learning_rate,
                num_epochs,
                verbose,
            )

        self.zip_model(output_model_path, output_model_name, zip_file_name)
        return None

    #    public step by step functions:
    def read_queries(self, read_path: str, overwrite: bool = False) -> pd.DataFrame:
        """
        Read the queries generated from the Synthetic Query Generator (SQG) model, unzip files to current directory
        within synthetic_queries/ folder, output as a dataframe

        Parameters
        ----------
        read_path: str
            required, path to the zipped file that contains generated queries, if None, raise exception
        overwrite: bool
            optional, synthetic_queries/ folder in current directory is to store unzip queries files.
            Default to set overwrite as false and if the folder is not empty, raise exception to recommend users
            to either clean up folder or enable overwriting is True.

        Returns
        -------
        The dataframe of queries.
        """

        if read_path is None:
            raise Exception(
                "No file provided. Please provide the path to synthetic query zip file."
            )

        # assign a local folder 'synthetic_queries/' to store the unzip file,
        # check if the folder contains sub-folders and files, remove and clean up the folder before unzip.
        # walk through the zip file and read the file paths into file_list
        unzip_path = os.path.join(self.folder_path, "synthetic_queries")

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
                                    "Fail to delete files, please delete all files in "
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
                "Zipped file is empty. Please provide a zip file with nonzero synthetic queries."
            )

        for file_path in file_list:
            f = open(file_path, "rb")
            print("reading synthetic query file: " + file_path + "\n")
            process.append(pickle.load(f))
            f.close()

        # reading the files to get the probability, queries and passages
        prob = []
        query = []
        passages = []
        for j in range(0, num_file):
            for dict_str in process[j]:
                if "probability" in dict_str.keys():
                    prob.append(dict_str["probability"])
                if "passage" in dict_str.keys():
                    passages.append(dict_str["passage"])
                if "query" in dict_str.keys():
                    query.append(dict_str["query"])

        df = pd.DataFrame(
            list(zip(prob, query, passages)), columns=["prob", "query", "passages"]
        )

        df = df.drop_duplicates(subset=["query"])
        df["passages"] = df.apply(lambda x: self.__qryrem(x), axis=1)

        df = df.sample(frac=1)
        return df

    def load_sentence_transformer_example(
        self, df, use_accelerate: bool = False
    ) -> List[str]:
        """
        Create input data for training

        Parameters
        ----------
        df: pd.DataFrame
            required for loading sentence transformer examples.
        use_accelerate: bool = False,
            Optional, use accelerate to fine tune model. Default as false to not use accelerator.

        Returns
        -------
        the list of train examples.
        """

        train_examples = []
        print("Loading training examples... \n")

        if use_accelerate is False:
            for i in tqdm(range(len(df)), total=len(df)):
                train_examples.append(
                    InputExample(
                        texts=[
                            df[i : i + 1]["passages"].values[0],
                            df[i : i + 1]["query"].values[0],
                        ]
                    )
                )
        else:
            queries = list(df["query"])
            passages = list(df["passages"])
            for i in tqdm(range(len(df)), total=len(df)):
                train_examples.append([queries[i], passages[i]])
        return train_examples

    def train_model(
        self,
        train_examples: List[str],
        model_id: str = None,
        output_path: str = None,
        output_model_name: str = None,
        use_accelerate: bool = False,
        learning_rate: float = 2e-5,
        num_epochs: int = 20,
        verbose: bool = False,
    ):
        """
        Description:
        Takes in training data and a sentence transformer url to train a custom semantic search model

        Parameters
        ----------
        train_examples:
            required, input for the sentence transformer model training
        model_id: str = None
            optional,the url to download sentence transformer model, if None, default as 'sentence-transformers/msmarco-distilbert-base-tas-b'
        output_path: str=None
            optional,the path to store trained custom model. If None, default as default_folder_path from constructor
        output_model_name: str=None
            optional,the name of the trained custom model. If None, default as model_id + '.pt'
        use_accelerate: bool = False,
            Optional, use accelerate to fine tune model. Default as false to not use accelerator.
        learning_rate: float
            optional, learning rate to train model, default is 2e-5
        num_epochs: int
            optional, number of epochs to train model, default is 20
        verbose: float
            optional, use plotting to plot the training progress. Default as false.

        Returns
        -------
        the torch script format trained model.
        """

        if output_path is None:
            output_path = self.folder_path
        if model_id is None:
            model_id = "sentence-transformers/msmarco-distilbert-base-tas-b"
        if output_model_name is None:
            output_model_name = str(self.model_id.split("/")[-1] + ".pt")

        # prepare an output model path for later saving the pt model.
        output_model_path = os.path.join(output_path, output_model_name)

        # declare variables before assignment for training
        batch_size = 32
        corp_len = []
        batch = []
        batch_q = []

        # Load a model from HuggingFace
        model = SentenceTransformer(model_id)

        # Calculate the length of queries
        if use_accelerate is True:
            for i in range(len(train_examples)):
                corp_len.append(len(train_examples[i][0].split(" ")))
        else:
            for i in range(len(train_examples)):
                corp_len.append(len(train_examples[i].__dict__["texts"][0].split(" ")))

        # Calculate the length of tokenizers
        # We obtain the max length of the top 95% of the documents. Since this length is measured in terms of
        # words and not tokens we multiply it by 1.4 to approximate the fact that 1 word in the english vocabulary
        # roughly translates to 1.3 to 1.5 tokens. For instance the word butterfly will be split by most tokeniers into
        # butter and fly, but the word sun will be kept as it is.
        # Note that this ratio will be higher if the corpus is jargon heavy
        # and/or domain specific.

        corp_max_tok_len = int(np.percentile(corp_len, 95) * 1.4)
        model.tokenizer.model_max_length = corp_max_tok_len
        model.max_seq_length = corp_max_tok_len

        # use accelerator for training
        if use_accelerate is True:
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

            print("Start training with accelerator...\n")
            print(f"The number of training epoch are {num_epochs}\n")
            print(
                f"The total number of steps training epoch are {len(train_dataloader)}\n"
            )

            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=min(10000, 0.05 * len(train_dataloader)),
                num_training_steps=num_epochs * len(train_dataloader),
            )

            accelerator = Accelerator(fp16=training_args.fp16)
            model, optimizer, train_dataloader, scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, scheduler
            )
            init_time = time.time()
            total_loss = []

            for epoch in range(num_epochs):
                print("Training epoch " + str(epoch) + "...\n")
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

                    den = torch.sum(XY, dim=0)
                    den1 = torch.sum(XY, dim=1)

                    batch_loss = -torch.sum(torch.log(num / den)) - torch.sum(
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

        # IF ACCELERATE IS FALSE
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
            print(f"The total number of steps training epoch are {steps_size}\n")

            for epoch in range(num_epochs):
                random.shuffle(train_examples)
                print("Training epoch " + str(epoch) + "...\n")
                for j in tqdm(range(steps_size), total=steps_size):
                    batch = []
                    batch_q = []
                    for example in train_examples[
                        j * batch_size : (j + 1) * batch_size
                    ]:
                        batch_q.append(example.texts[1])
                        batch.append(example.texts[0])

                    out_q = model.tokenize(batch_q)
                    for key in out_q.keys():
                        out_q[key] = out_q[key].to(device)
                    Y = model(out_q)["sentence_embedding"]

                    out = model.tokenize(batch)
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

                    if not j % 500 and j != 0:
                        plt.plot(loss[::100])
                        plt.show()

        # saving the pytorch model and the tokenizers.json file is saving at this step
        model.save(output_path)
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

        print("Preparing model to save...\n")
        torch.jit.save(traced_cpu, output_model_path)
        print("Model saved to path: " + output_model_path + "\n")
        return traced_cpu

    def zip_model(
        self, model_path: str = None, model_name: str = None, zip_file_name: str = None
    ) -> None:
        """
        Description:
        zip the model file and its tokenizer.json file to prepare to upload to the Open Search cluster

        Parameters
        ----------
        model_path: str
            Optional, path to find the model file, if None, default as concatenate model_id and '.pt' file in current path
        model_name: str=None
            the name of the trained custom model. If None, default as concatenate model_id and '.pt'
        zip_file_name: str =None
            Optional, file name for zip file. if None, default as concatenate model_id and '.zip'

        Returns
        -------
        None
        """
        if model_name is None:
            model_name = str(self.model_id.split("/")[-1] + ".pt")

        if model_path is None:
            model_path = os.path.join(self.folder_path, str(model_name))
        else:
            model_path = os.path.join(model_path, str(model_name))

        if zip_file_name is None:
            zip_file_name = str(model_name + ".zip")

        tokenizer_json_path = os.path.join(self.folder_path, "tokenizer.json")
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
            zipObj.write(model_path)
            zipObj.write(tokenizer_json_path)
        print("zip file is saved to " + zip_file_path + "\n")

    def save_as_pt(
        self,
        sentences: [str],
        model=None,
        model_name: str = None,
        save_json_folder_path: str = None,
        zip_file_name: str = None,
    ):
        """
        download sentence transformer model directly from huggingface, convert model to torch script format,
        zip the model file and its tokenizer.json file to prepare to upload to the Open Search cluster

        Parameters
        ----------
        sentences:[str]
            Required, for example  sentences = ['today is sunny']
        model: str
            Optional, if provide model in parameters, will convert model to torch script format,
            else, not provided model then it will download sentence transformer model from huggingface.
            If None, default takes model_id = "sentence-transformers/msmarco-distilbert-base-tas-b".
        model_name: str
            Optional, model name to name the model file, e.g, "sample_model.pt". If None, default takes the
            model_id and add the extension with ".pt".
        save_json_folder_path:
             Optional, path to save model json file, e.g, "home/save_pre_trained_model_json/"). If None, default as
             default_folder_path from the constructor.
        zip_file_name: str =None
            Optional, file name for zip file. e.g, "sample_model.zip". If None, default takes the model_id
            and add the extension with ".zip".

        Returns
        -------
        the torch script format model
        """

        if model is None:
            model = SentenceTransformer(self.model_id)

        if model_name is None:
            model_name = str(self.model_id.split("/")[-1] + ".pt")

        model_path = os.path.join(self.folder_path, model_name)

        if save_json_folder_path is None:
            save_json_folder_path = self.folder_path

        if zip_file_name is None:
            zip_file_name = str(self.model_id.split("/")[-1] + ".zip")
        zip_file_path = os.path.join(self.folder_path, zip_file_name)

        # save tokenizer.json in save_json_folder_name
        model.save(save_json_folder_path)

        # convert to pt format will need to be in cpu,
        # set the device to cpu, convert its input_ids and attention_mask in cpu and save as .pt format
        device = torch.device("cpu")
        torch.tensor(1)
        cpu_model = model.to(device)
        features = cpu_model.tokenizer(sentences)
        out_q = features
        input_ids = torch.tensor(out_q["input_ids"]).to("cpu")
        attention_mask = torch.tensor(out_q["attention_mask"]).to("cpu")
        compiled_model = torch.jit.trace(
            cpu_model.to("cpu"),
            ({"input_ids": input_ids, "attention_mask": attention_mask}),
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
                str(save_json_folder_path + "tokenizer.json"), arcname="tokenizer.json"
            )
        print("zip file is saved to ", zip_file_path, "\n")
        return compiled_model

    def set_up_accelerate_config(
        self,
        compute_environment: str = None,
        num_machines: int = 1,
        num_processes: int = None,
    ) -> None:
        """
        get default config setting based on the number of GPU on the machine
        if users require other configs, users can run !acclerate config for more options.

        Parameters
        ----------
        compute_environment: str
            optional, compute environment type to run model, if None, default using 'LOCAL_MACHINE'
        num_machines: int
            optional, number of machine to run model , if None, default using 1

        Returns
        -------
        None
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
        print("generated config file: at " + file_path + "\n")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if num_processes is None:
            if torch.cuda.is_available():
                num_processes = torch.cuda.device_count()

            else:
                num_processes = 1

        default_file = [
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
                "use_cpu": False,
            }
        ]
        print(default_file)

        with open(file_path, "w") as file:
            yaml.dump(default_file, file)

    def make_model_config_json(
        self,
        model_name: str = None,
        version_number: int = 1,
        embedding_dimension: int = 768,
        all_config: str = None,
        model_type: str = None,
    ) -> None:
        """
        parse from config.json file of pre-trained hugging-face model to generate a model_config.json file. If all required
        fields are given by users, use the given parameters and will skip reading the config.json,

        Parameters
        ----------
        model_name: str = None
            Optional, The name of the model. If None, default to parse from model id, for example,
            'msmarco-distilbert-base-tas-b'
        version_number: int = 1,
            Optional, The version number of the model. If None, default to be 1.
        embedding_dimension: int = 768,
            Optional, the embedding_dimension of the model. If None, parse embedding_dimension from the config file of
             pre-trained hugging-face model, if not found, default to be 768.
        all_config: dict = None,
            Optional, the all_config of the model. If None, parse all contents from the config file of pre-trained
            hugging-face model.
        model_type: str = None,
            Optional, the model_type of the model. If None, parse model_type from the config file of pre-trained
            hugging-face model.

        Returns
        -------
        None
        """
        folder_path = self.folder_path
        config_json_file_path = os.path.join(folder_path, "config.json")
        if model_name is None:
            model_name = self.model_id.split("/")[-1]

        if all_config is None or model_type is None or embedding_dimension == 768:
            if not os.path.exists(config_json_file_path):
                raise Exception(
                    str(
                        "Cannot find config.json in"
                        + config_json_file_path
                        + ". Please check the config.son file in the path."
                    )
                )
            f = open(config_json_file_path)
            if all_config is None:
                all_config = json.load(f)
            if model_type is None:
                if "model_type" in all_config.keys():
                    model_type = all_config["model_type"]
                else:
                    raise Exception(
                        str(
                            "Cannot find model_type in config.json file"
                            + config_json_file_path
                            + ". Please check the config.son file in the path."
                        )
                    )

            embedding_dimension_mapping_list = ["dim", "hidden_size", "d_model"]
            for mapping_item in embedding_dimension_mapping_list:
                if mapping_item in all_config.keys():
                    embedding_dimension = all_config[mapping_item]
                    break

            f.close()

        print("reading config file: at" + config_json_file_path)

        default_file = {
            "name": model_name,
            "version": version_number,
            "model_format": "TORCH_SCRIPT",
            "model_task_type": "TEXT_EMBEDDING",
            "model_config": {
                "model_type": model_type,
                "embedding_dimension": embedding_dimension,
                "framework_type": "sentence_transformers",
                "all_config": str(all_config),
            },
        }

        print("generating model_config.json file...")
        print(default_file)

        model_config_file_path = os.path.join(folder_path, "model_config.json")
        os.makedirs(os.path.dirname(model_config_file_path), exist_ok=True)
        with open(model_config_file_path, "w") as file:
            json.dump(default_file, file)
        print("model_config.json file is saved at", model_config_file_path, ".")

    # private methods
    def __qryrem(self, x):
        # for removing the "QRY:" token
        y = x.passages.split(" QRY:")[0].split("<|startoftext|>")
        return y[1]

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
