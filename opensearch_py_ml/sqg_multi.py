import json, pyserini, os,  pickle
from datetime import datetime
import argparse, random, sys, time
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch, transformers, pickle, scipy, sklearn, numpy as np, pandas as pd, torch.nn as nn
import torch.distributed as dist
from accelerate import Accelerator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import TrainingArguments, Trainer, logging
import datasets
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
 

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl', bos_token = '<|startoftext|>', eos_token = '<|endoftext|>', pad_token='<|pad|>', additional_special_tokens=["QRY:"]) 
# To be changed to download from url 
model_path = os.getcwd()
model = GPT2LMHeadModel.from_pretrained(model_path)

default_args = {
    "output_dir": "~/",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}
soft = nn.Softmax(dim=0)

# Define the class for the dataset. An instantiation of this class will be the input dataset. 

class GPT2Dataset_runtime(Dataset):
    def __init__(self, txt_list, tokenizer, gpt2_type="gpt2-xl", max_length=580):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        i = 0
        count = 0
        for txt in txt_list:
            if type(txt) == str:
                encodings_dict = tokenizer('<|startoftext|>' + txt + ' QRY: ' , truncation=True, max_length=max_length)
                if 50259 in encodings_dict['input_ids']:
                    self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                    count += 1
                    self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
                if i % 1000 == 0:
                    print(i,count)
                i += 1        
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx] 
                  
training_args = TrainingArguments(per_device_train_batch_size = 1, gradient_accumulation_steps = 1, fp16 = False,**default_args) 

def intersection_lists(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def jaccard_similarity(query, document):
    query = list((map(lambda x: x.lower(), query)))
    document = list((map(lambda x: x.lower(), document)))
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def jaccard_len_similarity(query, document):        
    query = list((map(lambda x: x.lower(), query)))
    document = list((map(lambda x: x.lower(), document)))    
    intersection = intersection_lists(query, document)
    union = query + document
    return len(intersection)/len(union)

# 
data_path = os.path.join(os.getcwd(), 'pre_trained_tokenizers')
f = open(data_path, "rb")
train_dataset = pickle.load(f)
f.close()

train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = training_args.per_device_train_batch_size)

learning_rate = 2e-5
epsilon = 1e-8
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, eps = epsilon)
accelerator = Accelerator(fp16=training_args.fp16)
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

numseq = 5
print("The number of steps for creating queries that each process will take are", len(train_dataloader))
output_dir = {}
save_path = os.path.join(os.getcwd(), '/output')

with torch.no_grad():
    for step, batch in enumerate(train_dataloader):
        for step in len(train_dataloader): 
            numgens = 8
            b_input_ids = batch[0]
            for gen in range(numgens):    
                if gen % 2 == 0:
                    temp = 0.95
                else:
                    temp = 1.0        

                beam = model.generate(b_input_ids, do_sample = True, 
                                        top_k = 50, 
                                        max_length = len(b_input_ids[0])+125, 
                                        #min_length = len(b_input_ids[0])+5, 
                                        top_p = 0.95, 
                                        temperature = temp, 
                                        num_return_sequences = numseq,
                                        #length_penalty = 20,
                                        repetition_penalty = 1.2,
                                        no_repeat_ngram_size = 3, 
                                        return_dict_in_generate = True, 
                                        output_scores = True)     

                mydict = {}   
                myjac = {}
                myjaclen = {}
                docu = tokenizer.convert_ids_to_tokens(beam["sequences"][0][1:list(beam["sequences"][0]).index(50259)]) 
                doc = tokenizer.decode(beam["sequences"][0], skip_special_tokens = False).split("QRY: ")[0][16:]
                for i in range(numseq):    
                    counter = 0
                    prob = 1e3
                    for k in range(len(b_input_ids[0]),beam["sequences"].shape[1]):        
                        if beam["sequences"][i][k] == 50256:
                            break 
                        softmult = soft(beam["scores"][counter][i])[beam["sequences"][i][k]] 
                        prob = prob*softmult
                        counter += 1
                    qu = tokenizer.convert_ids_to_tokens(beam["sequences"][i][list(beam["sequences"][i]).index(50259)+1:-1]) 
                    q = tokenizer.decode(beam["sequences"][i], skip_special_tokens = False).split("QRY: ")[-1].split("<|")[0]    

                    prob = prob.cpu()
                    mydict[prob] = q
                    jac_score = jaccard_similarity(qu,docu)            
                    jac_len_score = jaccard_len_similarity(qu,docu)            
                    myjac[prob] = jac_score
                    myjaclen[prob] = jac_len_score
                mydict = dict(sorted(mydict.items()))
                myjac = dict(sorted(myjac.items()))
                myjaclen = dict(sorted(myjaclen.items()))                
                output_dir[accelerator.process_index, step, gen, temp] = [mydict, myjac, myjaclen, tokenizer.decode(b_input_ids[0])]
                del beam 
                del mydict 
                torch.cuda.empty_cache() 

            if step % 100 == 0:
                print(accelerator.process_index, step, time.time())        
                f = open(save_path + str(accelerator.process_index),"wb")
                pickle.dump(output_dir, f)
                f.close()


    f = open(save_path + str(accelerator.process_index),"wb")
    pickle.dump(output_dir, f)
    f.close()

    

