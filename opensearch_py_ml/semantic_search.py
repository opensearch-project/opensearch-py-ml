# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import os
from sentence_transformers import SentenceTransformer, InputExample, losses
import json, pyserini, pickle
import argparse, random, time
import torch, transformers, pickle, sklearn, numpy as np, pandas as pd, torch.nn as nn
from accelerate import Accelerator
from accelerate import notebook_launcher
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import TrainingArguments, Trainer, logging
from rich.progress import track
import warnings
import matplotlib.pyplot as plt
import yaml


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings('ignore')


class Semantic_Search():

    # constructor
    # each object should have attributes for its corpus, sentences, attn_masks and input_ids.

    def __init__(self, data_folder: str = None,
                 column_list: list = None,
                 sentences: list = None,
                 corpus_file: str = "corpus.jsonl",
                 client=None,
                 index_name=None,
                 max_size = 9999):
        self.sentences = []
        self.input_ids = []
        self.attn_masks = []
        self.corpus = {}

        #         Option 1: read from local data folder in jsonl file
        if data_folder is not None:
            self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
            print(self.corpus_file)
            if not len(self.corpus):
                print("Loading Corpus...")

                num_lines = sum(1 for i in open(self.corpus_file, 'rb'))

                with open(self.corpus_file, 'r') as f:
                    for line in tqdm(f, total=num_lines):
                        line = json.loads(line)
                        id = ''
                        column_count = 0
                        for column in line:
                            if column_count == 0:
                                id = line.get(column)
                                self.corpus[id] = {}
                            else:
                                self.corpus[id][column] = line.get(column)

                            column_count += 1

                print("Loaded %d  Documents.", len(self.corpus))
                print("Doc Example: %s", list(self.corpus.values())[0])
                self.sentences = self.generate_sentences(column_list)
            else:
                raise Exception("Corpus file is empty! Please provide accurate file.")

        #         Option 2: read from list of sentences
        elif sentences is not None:
            self.sentences = sentences

        #         Option 3: read from OpenSearch client
        elif client is not None:
            response = client.search(index=index_name,
                                     body={"size": max_size,
                                           "query": {
                                               "match_all": {}}})

            cor = response['hits']['hits']

            for records in cor:
                id = records['_id']
                self.corpus[id] = {}
                source = records['_source']
                for column in source:
                    self.corpus[id][column] = source.get(column)

            self.generate_sentences(column_list)

        else:
            raise Exception("Neither json file nor sentences is present! Please provide accurate file.")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl', bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                                  pad_token='<|pad|>', additional_special_tokens=["QRY:"])
        self.tokenizer = tokenizer

    def generate_sentences(self, column_list: list = None) -> list:
        sentences = []

        for key in self.corpus:
            row = self.corpus[key]
            sent = ''
            for column in row:
                if column_list is None:
                    if row[column]:
                        sent = sent + row[column].strip()
                else:
                    if column in column_list:
                        sent = sent + row[column].strip()
            sentences.extend([sent])

        self.sentences = sentences

        return self.sentences

    def generate_token(self, gpt2_type="gpt2-xl", max_length=580, path=None) -> None:
        # check if  the folder existed . throw warning
        if path is None:
            path = os.path.join(os.getcwd(), 'pre_trained_tokenizers')

        # Checking if the list is empty or not
        if not self.check_empty_directory(path):
            raise Exception("the directory %s is not empty, please input other paths or empty path", path)

        txt_list = self.sentences
        tokenizer = self.tokenizer
        i = 0
        count = 0
        qry_id = tokenizer("QRY:")['input_ids'][0]
        for txt in txt_list:
            if type(txt) == str:
                encodings_dict = self.tokenizer('<|startoftext|>' + txt + ' QRY: ', truncation=True,
                                                max_length=max_length)
                if qry_id in encodings_dict['input_ids']:
                    self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                    count += 1
                    self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
                if i % 1000 == 0:
                    print(i, count)
                i += 1

        # auto save token when it generated

        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

    def save_tokenizers(self, folder_path) -> None:
        f = open(folder_path, "wb")
        pickle.dump(self, f)
        f.close()

    def save_tokenizers_json(self, folder_path=None) -> None:
        if folder_path is None:
            folder_path = os.path.join(os.getcwd(), 'pre_trained_tokenizers')
            # Checking if the list is empty or not
        if not self.check_empty_directory(folder_path):
            raise Exception("the directory %s is not empty, please input other paths or empty path", folder_path)
        self.tokenizer.save_pretrained(folder_path)

    def intersection_lists(self, lst1, lst2) -> list:
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    def jaccard_similarity(self, query, document) -> int:
        query = list((map(lambda x: x.lower(), query)))
        document = list((map(lambda x: x.lower(), document)))
        intersection = set(query).intersection(set(document))
        union = set(query).union(set(document))
        return len(intersection) / len(union)

    def jaccard_len_similarity(self, query, document) -> int:
        query = list((map(lambda x: x.lower(), query)))
        document = list((map(lambda x: x.lower(), document)))
        intersection = self.intersection_lists(query, document)
        union = query + document
        return len(intersection) / len(union)

    # helpful function to get default config running through GPU based on the number of GPU on the machine
    # if users require other configs, users can run !acclerate config for more options.
    def set_up_accelerate_config(compute_environment=None, num_machines=None) -> None:
        if compute_environment is None or compute_environment == 0:
            compute_environment = 'LOCAL_MACHINE'
        else:
            compute_environment = 'AWS'  # need to double check options

        if num_machines is None:
            num_machines = 1

        hf_cache_home = os.path.expanduser(
            os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
        )
        cache_dir = os.path.join(hf_cache_home, "accelerate")
        file_path = os.path.join(cache_dir + '/default_config.yaml')
        print('generated config file: at' + file_path)
        if torch.cuda.is_available():
            num_processes = torch.cuda.device_count()

        else:
            torch.cuda.device_count()

        default_file = [{'compute_environment': compute_environment,
                         'deepspeed_config':
                             {'gradient_accumulation_steps': 1,
                              'offload_optimizer_device': 'none',
                              'offload_param_device': 'none',
                              'zero3_init_flag': False,
                              'zero_stage': 2},
                         'distributed_type': 'DEEPSPEED',
                         'downcast_bf16': 'no',
                         'fsdp_config': {},
                         'machine_rank': 0,
                         'main_process_ip': None,
                         'main_process_port': None,
                         'main_training_function': 'main',
                         'mixed_precision': 'no',
                         'num_machines': num_machines,
                         'num_processes': num_processes,
                         'use_cpu': False}
                        ]
        print(default_file)

        with open(file_path, 'w') as file:
            documents = yaml.dump(default_file, file)

    def generate_query(self, model_path=None, data_path=None, output_path=None) -> None:
        notebook_launcher(self.run_model(model_path=model_path, data_path=data_path, output_path=output_path))

    def run_model(self, model_path=None, data_path=None, output_path=None) -> None:

        default_args = {
            "output_dir": "~/",
            "evaluation_strategy": "steps",
            "num_train_epochs": 1,
            "log_level": "error",
            "report_to": "none",
        }
        if data_path is None:
            data_path = os.path.join(os.getcwd(), 'pre_trained_tokenizers')

        # Checking if the list is empty or not
        if self.check_empty_directory(data_path):
            raise Exception("the directory %s is  empty, please generate token and save to the path", data_path)

        if model_path is None:
            data_path = os.getcwd()

        model = GPT2LMHeadModel.from_pretrained(model_path)

        soft = nn.Softmax(dim=0)

        training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=1, fp16=False,
                                          **default_args)

        #         read tokenized data as training set
        f = open(data_path, "rb")
        train_dataset = pickle.load(f)
        f.close()

        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                      batch_size=training_args.per_device_train_batch_size)

        learning_rate = 2e-5
        epsilon = 1e-8
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=epsilon)
        accelerator = Accelerator(fp16=training_args.fp16)
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        tokenizer = self.tokenizer
        numseq = 5
        print("The number of steps for creating queries that each process will take are", len(train_dataloader))
        output_dir = {}
        save_path = output_path
        with torch.no_grad():
            for step, batch in track(enumerate(train_dataloader), description="Generating queries in progress"):
                numgens = 8
                b_input_ids = batch[0]
                for gen in range(numgens):
                    if gen % 2 == 0:
                        temp = 0.95
                    else:
                        temp = 1.0

                    beam = model.generate(b_input_ids,
                                          do_sample=True,
                                          top_k=50,
                                          max_length=len(b_input_ids[0]) + 125,
                                          # min_length = len(b_input_ids[0])+5,
                                          top_p=0.95,
                                          temperature=temp,
                                          num_return_sequences=numseq,
                                          # length_penalty = 20,
                                          repetition_penalty=1.2,
                                          no_repeat_ngram_size=3,
                                          return_dict_in_generate=True,
                                          output_scores=True)

                    mydict = {}
                    myjac = {}
                    myjaclen = {}
                    docu = tokenizer.convert_ids_to_tokens(
                        beam["sequences"][0][1:list(beam["sequences"][0]).index(50259)])
                    doc = tokenizer.decode(beam["sequences"][0], skip_special_tokens=False).split("QRY: ")[0][16:]
                    for i in range(numseq):
                        counter = 0
                        prob = 1e3
                        for k in range(len(b_input_ids[0]), beam["sequences"].shape[1]):
                            if beam["sequences"][i][k] == 50256:
                                break
                            softmult = soft(beam["scores"][counter][i])[beam["sequences"][i][k]]
                            prob = prob * softmult
                            counter += 1
                        qu = tokenizer.convert_ids_to_tokens(
                            beam["sequences"][i][list(beam["sequences"][i]).index(50259) + 1:-1])
                        q = tokenizer.decode(beam["sequences"][i], skip_special_tokens=False).split("QRY: ")[-1].split(
                            "<|")[0]

                        prob = prob.cpu()
                        mydict[prob] = q
                        jac_score = self.jaccard_similarity(qu, docu)
                        jac_len_score = self.jaccard_len_similarity(qu, docu)
                        myjac[prob] = jac_score
                        myjaclen[prob] = jac_len_score

                    mydict = sorted(mydict.items())
                    myjac = sorted(myjac.items())
                    myjaclen = sorted(myjaclen.items())
                    output_dir[accelerator.process_index, step, gen, temp] = [mydict, myjac, myjaclen,
                                                                              tokenizer.decode(b_input_ids[0])]
                    del beam
                    del mydict
                    torch.cuda.empty_cache()

                if step % 100 == 0:
                    print(accelerator.process_index, step, time.time())
            #                     f = open(save_path + str(accelerator.process_index), "wb")
            #                     print("output file is stored at " + save_path)
            #                     pickle.dump(output_dir, f)
            #                     f.close()

            f = open(save_path + str(accelerator.process_index), "wb")
            pickle.dump(output_dir, f)
            f.close()

    # Helper function for removing the "QRY:" token
    def qryrem(self, x):
        y = x.passages.split(" QRY:")
        return y[0]

    def strem(self, x):
        y = x.passages.split("<|startoftext|>")
        return y[1]

    def check_empty_directory(self, data_path):

        if os.path.isfile(data_path):
            if not os.path.exists(data_path):
                return True
            else:
                return False

        if not os.path.exists(data_path):
            return True
        else:
            dir = os.listdir(data_path)

        # Checking if the list is empty or not
        if len(dir) == 0:
            print("Empty directory")
            return True
        else:
            return False

        return False

    def read_queries(self, read_path=None):
        if read_path is None:
            read_path = os.getcwd()

        if self.check_empty_directory(read_path):
            raise Exception("the directory %s is  empty, please generate query and save to the path", read_path)

        process = []
        num_gpu = 1

        if torch.cuda.is_available():
            num_gpu = torch.cuda.device_count()



        for i in range(0, num_gpu):
            f = open(read_path + '/output' + str(i), "rb")
            process.append(pickle.load(f))
            f.close()


        prob = []
        query = []
        passages = []
        for j in range(num_gpu):
            for key, dict_str in process[j].items():
                ourdict = dict_str[0]
                for probs,queries in ourdict.items():
                    passages.append(dict_str[1])
                    prob.append(probs)
                    query.append(queries)
                    #       build data frame
        df = pd.DataFrame(list(zip(prob, query, passages)), columns =[ 'prob', 'query', 'passages'])
        df = df.drop_duplicates(subset=['query'])
        df["passages"] = df.apply(lambda x: self.qryrem(x), axis=1)
        df["passages"] = df.apply(lambda x: self.strem(x), axis=1)
        df = df.sample(frac=1)
        return df


    def load_sentence_transformer_example(self, df) -> list:
        # Create input data for the sentence transformer by using InputExample from sentence_transformers library
        init = time.time()
        train_examples = []
        for i in range(len(df)):
            train_examples.append(
                InputExample(texts=[df[i:i + 1]["passages"].values[0], df[i:i + 1]["query"].values[0]]))
            if not i % 1000:
                print(i, time.time() - init)
                init = time.time()
        return train_examples

    def read_train_samples(self, path) -> list:
        f = open(path, "rb")
        train_examples = pickle.load(f)
        f.close()
        return train_examples

    def save_train_samples(self, train_examples, path) -> None:
        f = open(path, "wb")
        if self.check_empty_directory(path):
            raise Exception("the directory %s is  empty, please generate query and save to the path", path)
        pickle.dump(train_examples, f)
        f.close()

    def train_model(self, train_examples, model_url, path=None, model_name=None):
        # Load a model from HuggingFace
        if path is None:
            path = os.getcwd()
        if model_name is None:
            model_name = 'trained_model.pt'

        model = SentenceTransformer(model_url)
        model.save('save_tokenizers')
        model_path = path + '/' + model_name
        print(model_path)
        device = "cuda"
        model.to(device)

        corp_len = []
        for i in range(len(train_examples)):
            corp_len.append(len(train_examples[i].__dict__["texts"][0].split(" ")))

        corp_max_len = int(np.percentile(corp_len, 90) * 1.5)
        model.model_max_length = corp_max_len
        model.tokenizer.model_max_length = corp_max_len
        model.max_seq_length = corp_max_len

        num_epochs = 20
        batch_size = 32
        total_steps = (len(train_examples) // batch_size) * num_epochs

        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(10000, total_steps * 0.05),
                                                    num_training_steps=total_steps)
        losser = []
        init_time = time.time()
        for epoch in range(num_epochs):
            random.shuffle(train_examples)
            for j in range(len(train_examples) // batch_size):
                batch = []
                batch_q = []
                ids = []
                for example in train_examples[j * batch_size:(j + 1) * batch_size]:
                    batch_q.append(example.texts[1])
                    batch.append(example.texts[0])
                    ids.append(example.guid)

                batch_neg = []
                i = 0
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

                den = torch.sum(XY, dim=0)
                den1 = torch.sum(XY, dim=1)

                l = - torch.sum(torch.log(num / den)) - torch.sum(torch.log(num / den1))
                losser.append(l.item())
                l.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if not j % 500 and j != 0:
                    print(f' Epoch: {epoch}, Step: {j}. Time taken for 500 steps {time.time() - init_time}')
                    plt.plot(losser[::100])
                    plt.show()

            cpu_model = model.to(device)

        out_q = model.tokenize(batch_q)
        for key in out_q.keys():
            out_q[key] = out_q[key].to(device)
        traced_cpu = torch.jit.trace(cpu_model,
                                     ({'input_ids': out_q['input_ids'], 'attention_mask': out_q['attention_mask']}),
                                     strict=False)

        torch.jit.save(traced_cpu, model_path)

        print("model is saved to the path" + model_path)
        return model

    def zip_model(self, model_path=None, zip_path=None) -> None:
        # Create a ZipFile Object
        if model_path is None:
            model_path = os.path.join(os.getcwd(), '/trained_model.pt')

        if zip_path is None:
            zip_path = os.path.join(os.getcwd(), '/zip_model.zip')

        with ZipFile(zip_path, 'w') as zipObj:
            # Add multiple files to the zip
            zipObj.write(model_path)
            zipObj.write('save_tokenizers/tokenizer.json')
        print('trained_model.zip is save to' + zip_path)
