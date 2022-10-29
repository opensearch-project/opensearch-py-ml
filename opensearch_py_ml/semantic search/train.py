# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import argparse
import os
import pickle
import random
import shutil
import time
from typing import List
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


def train_semantic_search_model(
    read_path: str = None,
    model_url: str = None,
    output_model_name: str = None,
    output_model_path: str = None,
    zip_file_name: str = None,
) -> None:
    """
    Description:
    read the synthetic queries and use it to fine tune/train a sentence transformer model to save a zip file

    Parameters:
    read_path: str
        Optional, path to read the generated queries zip file, if None, default as 'synthetic_query' folder in current directory
    model_url: str = None
        the url to download sentence transformer model, if None, default as 'sentence-transformers/msmarco-distilbert-base-tas-b'
    output_model_path: str=None
        the path to store trained custom model. If None, default as current folder path
    output_model_name: str=None
        the name of the trained custom model. If None, default as 'trained_model.pt'
    zip_file_name: str =None
        Optional, file name for zip file. if None, default as zip_model.zip
    Return:
        None
    """

    query_df = read_queries(read_path)
    train_examples = load_sentence_transformer_example(query_df)
    train_model(train_examples, model_url, output_model_path, output_model_name)
    zip_model(output_model_path, zip_file_name)
    return None


#     helpful functions:


def read_queries(read_path: str = None) -> pd.DataFrame:
    """
    Description:
    read the queries generated from the SQG model

    Parameters:
    read_path: str
        required, path to read the generated queries, if None, raise exception
    Return:
        The data frame of queries.
    """

    process = []
    file_list = []

    if read_path is None:
        raise Exception(
            "No file provided. Please provide the path to synthetic query zip file."
        )

    # assign a local folder 'synthetic_queries/' to store the unzip file,
    # check if the folder contains sub-folders and files, remove and clean up the folder before unzip.
    # walk through the zip file and read the file paths into file_list
    unzip_path = os.path.join(os.getcwd(), "synthetic_queries/")

    if os.path.exists(unzip_path):
        if len(os.listdir(unzip_path)) > 0:
            for files in os.listdir(unzip_path):
                sub_path = os.path.join(unzip_path, files)
                try:
                    shutil.rmtree(sub_path)
                except OSError as err:
                    print(
                        "Fail to delete files, please delete all files in "
                        + str(unzip_path)
                        + " "
                        + str(err)
                    )

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
        print("reading synthetic query file: " + file_path)
        process.append(pickle.load(f))
        f.close()

    # reading the files and get the probability, queries and passages,
    # TODO: this input part will change after the generate.py change the output format

    prob = []
    query = []
    passages = []
    for j in range(0, num_file):
        for key, dict_str in process[j].items():
            ourdict = dict_str[0]

            for probs, queries in ourdict:
                passages.append(dict_str[3])
                prob.append(probs)
                query.append(queries)

    df = pd.DataFrame(
        list(zip(prob, query, passages)), columns=["prob", "query", "passages"]
    )
    df = df.drop_duplicates(subset=["query"])
    df["passages"] = df.apply(lambda x: qryrem(x), axis=1)
    df = df.sample(frac=1)
    return df


def load_sentence_transformer_example(df) -> List[str]:
    """
    Description:
    Create input data for the sentence transformer by using InputExample

    Parameters:
    df: pd.DataFrame
        required for loading sentence transformer examples.

    Return:
        the list of train examples.
    """

    train_examples = []
    print("Loading training examples... ")
    for i in tqdm(range(len(df)), total=len(df)):
        train_examples.append(
            InputExample(
                texts=[
                    df[i : i + 1]["passages"].values[0],
                    df[i : i + 1]["query"].values[0],
                ]
            )
        )
    return train_examples


def train_model(
    train_examples: List[str],
    model_url: str = None,
    output_path: str = None,
    output_model_name: str = None,
):
    """
    Description:
    pass on train examples, sentences transformer url to train a custom semantic search model

    Parameters:
    train_examples:
        the InputExample for the sentence transformer model training
    model_url: str = None
        the url to download sentence transformer model, if None, default as 'sentence-transformers/msmarco-distilbert-base-tas-b'
    output_path: str=None
        the path to store trained custom model. If None, default as current folder path
    output_model_name: str=None
        the name of the trained custom model. If None, default as 'trained_model.pt'
    Return:
        None

    """

    if output_path is None:
        output_path = os.getcwd()
    if output_model_name is None:
        output_model_name = "trained_model.pt"
    if model_url is None:
        model_url = "sentence-transformers/msmarco-distilbert-base-tas-b"

    # Load a model from HuggingFace
    model = SentenceTransformer(model_url)
    # prepare an output model path for later saving the pt model.
    output_model_path = output_path + "/" + output_model_name
    # identify if running on gpu or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    corp_len = []
    for i in range(len(train_examples)):
        corp_len.append(len(train_examples[i].__dict__["texts"][0].split(" ")))

    corp_max_len = int(np.percentile(corp_len, 90) * 1.5)
    model.model_max_length = corp_max_len
    model.tokenizer.model_max_length = corp_max_len
    model.max_seq_length = corp_max_len

    # num of epochs and batch will change to adapting size based on the Input Example sizes
    num_epochs = 20
    batch_size = 32
    total_steps = (len(train_examples) // batch_size) * num_epochs
    steps_size = len(train_examples) // batch_size

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10000, total_steps * 0.05),
        num_training_steps=total_steps,
    )
    loss = []
    init_time = time.time()

    for epoch in range(num_epochs):
        random.shuffle(train_examples)
        print(" Training epoch " + str(epoch) + "...")
        for j in tqdm(range(steps_size), total=steps_size):
            batch = []
            batch_q = []
            for example in train_examples[j * batch_size : (j + 1) * batch_size]:
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

            if not j % 500 and j != 0:  # MS: change 500
                plt.plot(loss[::100])
                plt.show()

    # saving the pytorch model and the tokenizers.json file is saving at this step
    model.save("trained_pytorch_model/")
    device = "cpu"
    cpu_model = model.to(device)
    print(f" Total training time: {time.time() - init_time}")

    out_q = model.tokenize(batch_q)
    for key in out_q.keys():
        out_q[key] = out_q[key].to(device)
    traced_cpu = torch.jit.trace(
        cpu_model,
        ({"input_ids": out_q["input_ids"], "attention_mask": out_q["attention_mask"]}),
        strict=False,
    )

    print("Preparing model to save")
    torch.jit.save(traced_cpu, output_model_path)

    print("Model saved to path: " + output_model_path)
    return model


def zip_model(model_path: str = None, zip_file_name: str = None) -> None:
    """
    Description:
    zip the model file and its tokenizer.json file to prepare to upload to the Open Search cluster

    Parameters:
    model_path: str
        Optional, path to find the model file, if None, default as trained_model.pt file in current path
    zip_file_name: str =None
        Optional, file name for zip file. if None, default as zip_model.zip

    Return:
        None
    """

    if model_path is None:
        model_path = os.path.join(os.getcwd(), "trained_model.pt")

    if zip_file_name is None:
        zip_file_name = "zip_model.zip"

    if not os.path.exists("trained_pytorch_model/tokenizer.json"):
        raise Exception(
            "Cannot find tokenizer.json in trained_pytorch_model/tokenizer.json"
        )
    if not os.path.exists(model_path):
        raise Exception("Cannot find model in the model path")
    # Create a ZipFile Object
    with ZipFile("zip_model.zip", "w") as zipObj:
        zipObj.write(model_path)
        zipObj.write("trained_pytorch_model/tokenizer.json")
    print("zip file is saved to" + os.getcwd() + "/" + zip_file_name)


def qryrem(x):
    #         for removing the "QRY:" token
    y = x.passages.split(" QRY:")[0].split("<|startoftext|>")
    return y[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Please enter argument to start semantic search training."
    )
    parser.add_argument("--read_path", type=str, default=None)
    parser.add_argument("--model_url", type=str, default=None)
    parser.add_argument("--model_name", type=list, default=None)
    parser.add_argument("--output_model_path", default=None)
    parser.add_argument("--zip_file_name", type=str, default=None)

    args = parser.parse_args()
    read_path = args.read_path
    model_url = args.model_url
    model_name = args.model_name
    output_model_path = args.output_model_path
    zip_file_name = args.zip_file_name

    print("initiated a semantic search training.. ")

    train_semantic_search_model(
        read_path, model_url, model_name, output_model_path, zip_file_name
    )
