# -*- coding: utf-8 -*-
"""DLA4_Alessandro_De_Grandi.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13eFBgaunSeeeOPALk3XhJygnVRAa71Yc
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import math

from tqdm import tqdm

from torch.utils.data import Dataset


class Vocabulary:

    def __init__(self, pad_token="<pad>", unk_token='<unk>', eos_token='<eos>',
                 sos_token='<sos>'):
        self.id_to_string = {}
        self.string_to_id = {}
        
        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0
        
        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1
        
        # add the default unknown token
        self.id_to_string[2] = eos_token
        self.string_to_id[eos_token] = 2   

        # add the default unknown token
        self.id_to_string[3] = sos_token
        self.string_to_id[sos_token] = 3

        # shortcut access
        self.pad_id = 0
        self.unk_id = 1
        self.eos_id = 2
        self.sos_id = 3

    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    # if extend_vocab is True, add the new word
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


# Read the raw txt files and generate parallel text dataset:
# self.data[idx][0] is the tensor of source sequence
# self.data[idx][1] is the tensor of target sequence
# See examples in the cell below.
class ParallelTextDataset(Dataset):

    def __init__(self, src_file_path, tgt_file_path, src_vocab=None,
                 tgt_vocab=None, extend_vocab=False, device='cuda'):
        (self.data, self.src_vocab, self.tgt_vocab, self.src_max_seq_length,
         self.tgt_max_seq_length) = self.parallel_text_to_data(
            src_file_path, tgt_file_path, src_vocab, tgt_vocab, extend_vocab,
            device)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def parallel_text_to_data(self, src_file, tgt_file, src_vocab=None,
                              tgt_vocab=None, extend_vocab=False,
                              device='cuda'):
        # Convert paired src/tgt texts into torch.tensor data.
        # All sequences are padded to the length of the longest sequence
        # of the respective file.

        assert os.path.exists(src_file)
        assert os.path.exists(tgt_file)

        if src_vocab is None:
            src_vocab = Vocabulary()

        if tgt_vocab is None:
            tgt_vocab = Vocabulary()
        
        data_list = []
        # Check the max length, if needed construct vocab file.
        src_max = 0
        with open(src_file, 'r') as text:
            for line in text:
                tokens = list(line)[:-1]  # remove line break
                length = len(tokens)
                if src_max < length:
                    src_max = length

        tgt_max = 0
        with open(tgt_file, 'r') as text:
            for line in text:
                tokens = list(line)[:-1]
                length = len(tokens)
                if tgt_max < length:
                    tgt_max = length
        tgt_max += 2  # add for begin/end tokens
                    
        src_pad_idx = src_vocab.pad_id
        tgt_pad_idx = tgt_vocab.pad_id

        tgt_eos_idx = tgt_vocab.eos_id
        tgt_sos_idx = tgt_vocab.sos_id

        # Construct data
        src_list = []
        print(f"Loading source file from: {src_file}")
        with open(src_file, 'r') as text:
            for line in tqdm(text):
                seq = []
                tokens = list(line)[:-1]
                for token in tokens:
                    seq.append(src_vocab.get_idx(
                        token, extend_vocab=extend_vocab))
                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)
                # padding
                new_seq = var_seq.data.new(src_max).fill_(src_pad_idx)
                new_seq[:var_len] = var_seq
                src_list.append(new_seq)

        tgt_list = []
        print(f"Loading target file from: {tgt_file}")
        with open(tgt_file, 'r') as text:
            for line in tqdm(text):
                seq = []
                tokens = list(line)[:-1]
                # append a start token
                seq.append(tgt_sos_idx)
                for token in tokens:
                    seq.append(tgt_vocab.get_idx(
                        token, extend_vocab=extend_vocab))
                # append an end token
                seq.append(tgt_eos_idx)

                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)

                # padding
                new_seq = var_seq.data.new(tgt_max).fill_(tgt_pad_idx)
                new_seq[:var_len] = var_seq
                tgt_list.append(new_seq)

        # src_file and tgt_file are assumed to be aligned.
        assert len(src_list) == len(tgt_list)
        for i in range(len(src_list)):
            data_list.append((src_list[i], tgt_list[i]))

        print("Done.")
            
        return data_list, src_vocab, tgt_vocab, src_max, tgt_max


from google.colab import drive
drive.mount('/content/drive',force_remount=True)

# `DATASET_DIR` should be modified to the directory where you downloaded
# the dataset. On Colab, use any method you like to access the data
# e.g. upload directly or access from Drive, ...

#DATASET_DIR = "/content/"
DATASET_DIR = "/content/drive/MyDrive/Datasets for Assignment 4-20211224"

TRAIN_FILE_NAME = "train"
VALID_FILE_NAME = "interpolate"

INPUTS_FILE_ENDING = ".x"
TARGETS_FILE_ENDING = ".y"

#TASK = "numbers__place_value"
TASK = "comparison__sort"
# TASK = "algebra__linear_1d"

# Adapt the paths!

src_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{INPUTS_FILE_ENDING}"
tgt_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{TARGETS_FILE_ENDING}"

print(src_file_path)
print(tgt_file_path)

train_set = ParallelTextDataset(src_file_path, tgt_file_path, extend_vocab=True)

# get the vocab
src_vocab = train_set.src_vocab
tgt_vocab = train_set.tgt_vocab

src_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{INPUTS_FILE_ENDING}"
tgt_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{TARGETS_FILE_ENDING}"

valid_set = ParallelTextDataset(
    src_file_path, tgt_file_path, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    extend_vocab=False)



from torch.utils.data import DataLoader

batch_size = 64

train_data_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True)

valid_data_loader = DataLoader(
    dataset=valid_set, batch_size=batch_size, shuffle=False)

print(len(train_data_loader))
print(len(train_data_loader)*64)
print(len(train_data_loader)*64*53)
print(len(valid_data_loader))
print(len(valid_data_loader)*64)
print(len(valid_data_loader)*64*51)
print(len(src_vocab.id_to_string))
print(len(tgt_vocab.id_to_string))
# Example train batch
batch = next(iter(train_data_loader))
q = batch[0]  
print(q.shape)
a = batch[1]  
print(a.shape)
# Example valid batch
batch = next(iter(valid_data_loader))
q = batch[0]  
print(q.shape)
a = batch[1]  
print(a.shape)

########
# Taken from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# or also here:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # shape (max_len, 1, dim)
        self.register_buffer('pe', pe)  # Will not be trained.

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        assert x.size(0) < self.max_len, (
            f"Too long sequence length: increase `max_len` of pos encoding")
        # shape of x (len, B, dim)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#tgt_vocab.id_to_string[i.item()]

def src2seq(s):
    return [src_vocab.id_to_string[i] for i in s]
    
def seq2src(s):
    return [src_vocab.string_to_id[i] for i in s]

def tgt2seq(s):
    return [tgt_vocab.id_to_string[i] for i in s]

def seq2tgt(s):
    return [tgt_vocab.string_to_id[i] for i in s]

def greedyDecodeOne(model,srcBatch,tgtBatch,batch_size=64,max_len=100,show=False): ##One Batch
#def greedyDecode(model, dataloader, max_batches=10,show=False): ##For Test

    model.eval()
    correct=0

    with torch.no_grad():
        count=0

        src = srcBatch.transpose(0, 1)
        trg = tgtBatch.transpose(0, 1)   

        max_len= trg.size(dim=0) 
        #print(max_len)  
        memory = model.transformer.encoder(model.pos_encoder(model.encoder(src)))
        out_indexes = ((torch.ones((batch_size,1), dtype=torch.int64)*tgt_vocab.sos_id).T).cuda()
      
        for i in range(max_len-1):            
            trg_tensor = out_indexes
            out = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))             
            distr= torch.nn.functional.softmax(out, dim=2)            
            out_tokens = distr.argmax(2)[-1]
            out_indexes = torch.cat([out_indexes, out_tokens.unsqueeze(0)])
            #if torch.all(out_tokens==tgt_vocab.eos_id):
            #    break
                    
        for bi in range(batch_size):
          if(out_indexes[:,bi].shape == trg[:,bi].shape):
            correct+=int(torch.all(out_indexes[:,bi]==trg[:,bi]))
          #torch.eq(a, b)
        if(show):
          bi=0 
          question = src2seq(src[:,bi].tolist())
          answer = tgt2seq(trg[:,bi].tolist())
          prediction = tgt2seq(out_indexes[:,bi].tolist())
          
          print('\nQ: ', ''.join(question))
          print('A: ', ''.join(answer))
          print('P: ', ''.join(prediction))
    
    return correct/batch_size*100

def greedyDecode(model, dataloader, max_batches=10,max_len=100,show=False,testing=False): ##PARALLEL OK
#def greedyDecode(model, dataloader, max_batches=10,show=False): ##For Test

    batch_size=64
    

    #for test
    if(testing):
      batch_size=10                                        
      dataloader = DataLoader(
          dataset=dataloader, batch_size=batch_size, shuffle=False)
    
    #print("Parallel")
    
    model.eval()
    correct=0
    #print("evaluating...")
    with torch.no_grad():
        count=0

        #eos_tensor= ((torch.ones((batch_size,1), dtype=torch.int64)*tgt_vocab.eos_id).T).cuda()
        for i,batch in enumerate(dataloader):
            src, trg = batch
            src, trg = src.cuda(), trg.cuda()
            src = src.transpose(0, 1)
            trg = trg.transpose(0, 1)

            max_len= trg.size(dim=0)
            #print(max_len)

            #print(src.shape)
            #print(trg.shape)
            #print(src.shape)  
            #print(src)  
            #print(trg) 
            memory = model.transformer.encoder(model.pos_encoder(model.encoder(src)))
            #out_indexes = [tgt_vocab.sos_id, ]
            #dec input = torch.cat([dec input, predicted])
            out_indexes = ((torch.ones((batch_size,1), dtype=torch.int64)*tgt_vocab.sos_id).T).cuda()
            #out_indexes = torch.cat([out_indexes, out_indexes*2])         
            for i in range(max_len-1):            
                trg_tensor = out_indexes
                out = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))             
                distr= torch.nn.functional.softmax(out, dim=2)            
                out_tokens = distr.argmax(2)[-1]
                out_indexes = torch.cat([out_indexes, out_tokens.unsqueeze(0)])
                #if (torch.all(out_tokens==tgt_vocab.eos_id)):
                #    break
                        
            for bi in range(batch_size):
              
              #prediction = tgt2seq(out_indexes[:,bi].tolist())
              #answer = tgt2seq(trg[:,bi].tolist())         
              #correct += int(prediction == answer)

              if(out_indexes[:,bi].shape == trg[:,bi].shape):
                correct+=int(torch.all(out_indexes[:,bi]==trg[:,bi]))
              #torch.eq(a, b)
              if(show):
                  
                  question = src2seq(src[:,bi].tolist())
                  answer = tgt2seq(trg[:,bi].tolist())
                  prediction = tgt2seq(out_indexes[:,bi].tolist())
                 
                  print('\nQ: ', ''.join(question))
                  print('A: ', ''.join(answer))
                  print('P: ', ''.join(prediction))

            count+=1
            if(count>=max_batches):
              break
            #print(i)
            
          
    return correct/(max_batches*batch_size)*100

def evaluate(model, criterion, iterator):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():    
        for i, batch in enumerate(iterator):
            src, trg = batch
            src, trg = src.cuda(), trg.cuda()
            src = src.transpose(0, 1)
            trg = trg.transpose(0, 1)

            output = model(src, trg[:-1,:])
            loss = criterion(output.transpose(0, 1).transpose(1, 2), trg[1:,:].transpose(0, 1))
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

##@title
class TransformerModel(nn.Module):
    def __init__(self, intoken, outtoken, hidden, enc_layers, dec_layers, dim_feedforward, nhead,dropout=0.2):
        super(TransformerModel, self).__init__()
        
        self.encoder = nn.Embedding(intoken, hidden)
        self.pos_encoder = PositionalEncoding(hidden)

        self.decoder = nn.Embedding(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden)

        self.transformer = nn.Transformer(d_model=hidden, nhead=nhead, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, dim_feedforward=dim_feedforward, dropout=dropout)#, activation='relu')
        self.fc_out = nn.Linear(hidden, outtoken)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.transformer.generate_square_subsequent_mask(len(trg)).to(trg.device)

        #print(self.trg_mask)

        #src_pad_mask = self.make_len_mask(src)
        #trg_pad_mask = self.make_len_mask(trg)
        src_pad_mask = None
        trg_pad_mask = None

        #print(src_pad_mask)
        #print(trg_pad_mask)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask,
                                  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)

        return output

INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(tgt_vocab)

print(INPUT_DIM)
print(OUTPUT_DIM)

#ORIGINAL 
d_model=256
nhead=8 
num_encoder_layers=3
num_decoder_layers=2
dim_feedforward=1024
dropout=0.2

#TEST 1 
#d_model=128
#nhead=4 
#num_encoder_layers=2
#num_decoder_layers=1
#dim_feedforward=512
#dropout=0.1

#TEST 2 
#d_model=128
#nhead=8 
#num_encoder_layers=3
#num_decoder_layers=2
#dim_feedforward=512
#dropout=0.2

#TEST 3 
#d_model=256
#nhead=8 
#num_encoder_layers=2
#num_decoder_layers=1
#dim_feedforward=1024
#dropout=0.2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = TransformerModel(INPUT_DIM, OUTPUT_DIM, hidden=128, enc_layers=3, dec_layers=1).to(device)

model = TransformerModel(INPUT_DIM, OUTPUT_DIM, hidden=d_model, enc_layers=3, dec_layers=2, dim_feedforward=1024, nhead=8,dropout=dropout).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
print(model)
#lr = 1e-4  # learning rate
lr= 0.0001
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

TRG_PAD_IDX = 0


BEST_model=0
BEST_srcB=0
BEST_trgB=0

#criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
criterion = nn.CrossEntropyLoss()
train_acc_list=[]
val_acc_list=[]
train_loss_list=[]
val_loss_list=[]

N_EPOCHS = 10
stop_training=False
init_time = time.time()

for epoch in range(N_EPOCHS):
    BEST_ACC=0
      
    epoch_loss = 0
    num_batches=len(train_data_loader)
    
    start_time = time.time()
    grad_accum=10
    correct=0

    #log_interval=num_batches*grad_accum
    log_interval=1000

    print("START TRAINING:")
    for i, batch in enumerate(train_data_loader):
        model.train()
        srcB, trgB = batch
        srcB, trgB = srcB.cuda(), trgB.cuda()
        ##print("SRC TRG")
        #print(srcB.shape)
        #print(trgB.shape)
        #print(src)
        #print(trg)
        src = srcB.transpose(0, 1)
        trg = trgB.transpose(0, 1)
        #print("SRC TRG")
        #print(srcB.shape)
        #print(trgB.shape)
        #print(src)
        #print(trg)
        
        #optimizer.zero_grad()
        output = model(src, trg[:-1,:])
        #print("OUT")
        #print(output.shape)
        output=output.transpose(0, 1).transpose(1, 2)
        #print("OUT T")
        #print(output.shape)
        trg = trg[1:,:].transpose(0, 1)
        #print("trg T")
        #print(trg.shape)
        loss = criterion(output, trg) / grad_accum
        #print("loss")
        #print(loss)
        #loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        #optimizer.step()
        #epoch_loss += loss.item()
        #if(epoch == 0 and i ==0):
        #  print("START")
        #  ppl=math.exp(loss*grad_accum)
        #  print(f'loss {loss} | ppl {ppl} ')

        loss.backward()
            
        #epoch_loss += loss.item()
        if ((i + 1) % grad_accum == 0) or (i + 1 == num_batches):
          #print(f'#UPDATE {i}/{num_batches}')
          torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
          optimizer.step()
          optimizer.zero_grad()

        if i % log_interval == 0 and i > 0 :
            #lr = scheduler.get_last_lr()[0]
          ms_per_batch = (time.time() - start_time) * 1000 / log_interval
          #cur_loss = epoch_loss / log_interval
          ppl = math.exp(loss)
          #ppl = math.exp(loss)
          print(f'| epoch {epoch:3d} | {i:5d}/{num_batches:5d} batches | '
                #f'ms/batch {ms_per_batch:5.4f} | '
                #f'loss {loss:5.4f} | ppl {ppl:5.4f} || ',end="")
                f'train_loss {loss:5.4f} || ',end="")
                #f'expected time to finish epoch {(((num_batches-i)*ms_per_batch)/(60000)):5.2f} min')
          val_loss=evaluate(model,criterion,valid_data_loader)/grad_accum
          print(f'valid_loss {val_loss:5.4f} || ',end="")
          
          
          correct_training = greedyDecodeOne(model,srcB,trgB,show=True)
          print(f'Train_acc {correct_training:5.2f}% || ',end="")
          correct_valid = greedyDecode(model,dataloader=valid_data_loader,max_batches=10,show=False)
          print(f'Val_acc {correct_valid:5.2f}%')

          train_loss_list.append(loss)
          val_loss_list.append(val_loss)
          train_acc_list.append(correct_training)
          val_acc_list.append(correct_valid)





          if(correct_valid>BEST_ACC):
            BEST_ACC=correct_valid

          if(correct_valid >90 or (time.time()-init_time)/60 > 300):  
            greedyDecode(model,dataloader=valid_data_loader,max_batches=1,show=True)
            print(f'BEST_ACC {BEST_ACC}')
            stop_training=True
            break
            #BEST_ACC=0
            #BEST_model=0
            #BEST_srcB=0
            #BEST_trgB=0
            #BEST_model=0
            #correct = greedyDecode(model,srcB,trgB,show=True)

            
          start_time = time.time()
          #break
    if(stop_training):
       break

print(greedyDecode(model,dataloader=train_data_loader,max_batches=10,show=False))
print(greedyDecode(model,dataloader=valid_data_loader,max_batches=10,show=False))

print(evaluate(model,criterion,valid_data_loader))

import matplotlib.pyplot as plt
plt.plot(range(len(val_acc_list)), val_acc_list, c='blue',alpha=0.5, label="val_acc")
plt.plot(range(len(train_acc_list)),train_acc_list, c='orange',alpha=0.5, label="train_acc")
plt.legend(loc="lower right")
plt.show()

plt.plot(range(len(train_loss_list)),train_loss_list, c='orange',alpha=0.5, label="train_loss")
plt.plot(range(len(val_loss_list)),val_loss_list, c='blue',alpha=0.5, label="val_loss")
plt.legend(loc="upper right")
plt.show()

