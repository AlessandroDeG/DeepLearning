import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Vocabulary:

    def __init__(self, pad_token="<pad>", unk_token='<unk>'):
        self.id_to_string = {}
        self.string_to_id = {}
        
        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0
        
        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1        
        
        # shortcut access
        self.pad_id = 0
        self.unk_id = 1
        
    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


# Read the raw txt file and generate a 1D PyTorch tensor
# containing the whole text mapped to sequence of token IDs, and a vocab object.
class LongTextData:

    def __init__(self, file_path, vocab=None, extend_vocab=True, device='cuda'):
        self.data, self.vocab, self.full_text= self.text_to_data(file_path, vocab, extend_vocab, device)
        
    def __len__(self):
        return len(self.data)

    def text_to_data(self, text_file, vocab, extend_vocab, device):
        """Read a raw text file and create its tensor and the vocab.

        Args:
          text_file: a path to a raw text file.
          vocab: a Vocab object
          extend_vocab: bool, if True extend the vocab
          device: device

        Returns:
          Tensor representing the input text, vocab file

        """
        assert os.path.exists(text_file)
        if vocab is None:
            vocab = Vocabulary()

        data_list = []

        # Construct data
        full_text = []
        print(f"Reading text file from: {text_file}")
        n_lines=0
        with open(text_file, 'r') as text:
            for line in text:
                n_lines+=1
                tokens = list(line)
                for token in tokens:
                    # get index will extend the vocab if the input
                    # token is not yet part of the text.
                    full_text.append(vocab.get_idx(token, extend_vocab=extend_vocab))

        # convert to tensor
        data = torch.tensor(full_text, device=device, dtype=torch.int64)

        print("Done.\n")

        print("STATS:")
        print("Vocab size")
        print(len(vocab))
        print("N. Lines")
        print(n_lines)
        print("N. Chars")
        print(len(full_text))
        print("N. Chars per line")
        print(len(full_text)/n_lines)


        return data, vocab, full_text
    

# Since there is no need for schuffling the data, we just have to split
# the text data according to the batch size and bptt length.
# The input to be fed to the model will be batch[:-1]
# The target to be used for the loss will be batch[1:]
class ChunkedTextData:

    def __init__(self, data, bsz, bptt_len, pad_id):
        self.batches = self.create_batch(data, bsz, bptt_len, pad_id)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def create_batch(self, input_data, bsz, bptt_len, pad_id):
        """Create batches from a TextData object .

        Args:
          input_data: a TextData object.
          bsz: int, batch size
          bptt_len: int, bptt length
          pad_id: int, ID of the padding token

        Returns:
          List of tensors representing batches

        """
        batches = []  # each element in `batches` is (len, B) tensor
        text_len = len(input_data)
        segment_len = text_len // bsz + 1

        # Question: Explain the next two lines!
        padded = input_data.data.new_full((segment_len * bsz,), pad_id)
        #print(padded)
        print(padded.shape)
        padded[:text_len] = input_data.data
        
        #print(padded)
        print(padded.shape)
        padded = padded.view(bsz, segment_len).t()
        num_batches = segment_len // bptt_len + 1

        for i in range(num_batches):
            # Prepare batches such that the last symbol of the current batch
            # is the first symbol of the next batch.
            if i == 0:
                # Append a dummy start symbol using pad token
                batch = torch.cat(
                    [padded.new_full((1, bsz), pad_id),
                     padded[i * bptt_len:(i + 1) * bptt_len]], dim=0)
                #print(batch)
                batches.append(batch)
            else:
                batches.append(padded[i * bptt_len - 1:(i + 1) * bptt_len])

        return batches

# downlaod the text
# Make sure to go to the link and check how the text looks like.

#!wget http://www.gutenberg.org/files/49010/49010-0.txt

# This is for Colab. Adapt the path if needed.
"""
text_path = "/content/49010-0.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cuda"
batch_size = 32
bptt_len = 64

my_data = LongTextData(text_path, device=device)

torch.set_printoptions(threshold=10000000)
print("LEN DATA")
print(my_data.__len__())
batches = ChunkedTextData(my_data, batch_size, bptt_len, pad_id=0)
print("LEN CHUNK")
print(batches.__len__())

print(64*32*87)

#print(batches.batches)
"""
#BONUS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cuda"
batch_size = 32
bptt_len = 128
text_path="data/ABunchOfStuffFromTheInternet.txt"
my_data = LongTextData(text_path, device=device)

torch.set_printoptions(threshold=10000000)
print("LEN DATA")
print(my_data.__len__())
batches = ChunkedTextData(my_data, batch_size, bptt_len, pad_id=0)
print("LEN CHUNK")
print(batches.__len__())

#STATS

print(my_data.data[0])

print(len(my_data.vocab))

#prints the whole vocabulary
for key, value in my_data.vocab.string_to_id.items():
    print("(%s,%d)" %(key,value))

#prints the first n characters of the book
def printN(n, text_keys): #full text is a list of keys
  for key in text_keys: 
    print(my_data.vocab.id_to_string[key],end="")
    n-=1
    if(n==0):
      break
  print()


print()
print(64,my_data.full_text)

print()
print(64,my_data.full_text)

len(batches)

batches[0].shape

# input to the network
print(batches[0][:-1].shape)
print(batches[0][:-1, 0])
print(batches[0][:-1, 1])
print(batches[0][:-1, 2])

# target tokens to be predicted
print(batches[0][1:].shape)
print(batches[0][1:, 0])
print(batches[0][1:, 1])
print(batches[0][1:, 2])

# last token of the current batch should be the first token of the next one:
for i in batches[20][:, 11]:
    print(my_data.vocab.id_to_string[i.item()])
print('================')
for i in batches[21][:, 11]:
    print(my_data.vocab.id_to_string[i.item()])

print(my_data.vocab.id_to_string)
print(my_data.vocab.string_to_id)


"""MODEL"""

def decode(model,prompt,out_len,sampling=True):
  model.eval()
  h = torch.zeros(num_layers, 1, hidden_size).to(device)
  c = torch.zeros(num_layers, 1, hidden_size).to(device)
  prompt = list(prompt)
  output=0
  out=0

  for char in prompt:
    input_tensor = torch.ones((1,1)).new_full((1,1),my_data.vocab.string_to_id[char],dtype=torch.int64).to(device)
    #print(input_tensor)
    output, (h, c) = model(input_tensor, (h, c))
  
  #prediction for last char of prompt
  distr= torch.nn.functional.softmax(output, dim=1)
  if(sampling):
      out=torch.multinomial(distr, num_samples=1).item()
  else:
      out=torch.argmax(distr).item()
    
  for i in range(out_len):
    out=my_data.vocab.id_to_string[out]
    print(out, end='')
    input_tensor = torch.ones((1,1)).new_full((1,1),my_data.vocab.string_to_id[out],dtype=torch.int64).to(device)
    output, (h, c) = model(input_tensor, (h, c))
    #print(output.shape)
    distr= torch.nn.functional.softmax(output,dim=1)
    if(sampling):
      out=torch.multinomial(distr, num_samples=1).item()
    else:
      out=torch.argmax(distr).item()

# Define the model

embed_size = 64
hidden_size = 2048
num_layers = 1
num_epochs = 100
batch_size = 32
learning_rate = 0.001
num_batches= len(batches)


class LSTMModel(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size):
        super(LSTMModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size,padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)#, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)

        #print(x)

        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)

        # Reshape output to (batch_size * sequence_length, hidden_size)
        #print(out.size(0))
        #print(out.size(0))
        #print(out.size(0))

        out = out.reshape(out.size(0) * out.size(1), out.size(2))

        #print(out.shape)

        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)

#model
vocab_size = len(my_data.vocab)
model = LSTMModel(embed_size, hidden_size, num_layers, vocab_size)
print(model)
model = model.to(device)

# Set loss and optimizer function
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model

h = torch.zeros(num_layers, batch_size, hidden_size)
c = torch.zeros(num_layers, batch_size, hidden_size)

stop_training=False
for epoch in range(num_epochs): 
    
    for i in range(num_batches):
        model.train()
        #h = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        #c = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        h=h.to(device)
        c=c.to(device)
        x= batches[i][:-1,:]
        y= batches[i][1:,:]
       
        # Forward pass
        outputs, (h, c) = model(x, (h, c))
        #print("outputs")
        #print(outputs)
        #print(outputs.shape)
        loss = loss_fn(outputs, y.flatten())

        # Backward and optimize
        #model.zero_grad()
        h = h.detach()
        c = c.detach()
        optimizer.zero_grad() 
        model.zero_grad()
        #torch.autograd.set_detect_anomaly(True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
       
        step = i  
        perplexity = np.exp(loss.item())
        if step % 20 == 0:
          
          print('\nEpoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'.format(epoch + 1, num_epochs, step+1, num_batches, loss.item(), perplexity))
          #prompt = "Dogs like best to"
          prompt = "This assignment is"
          out_len=15
          print(prompt, end='')
          decode(model, prompt,out_len)
          

        if (perplexity<1.03):
          stop_training=True
          break
    if stop_training :
      print("\nPerplexity<1.03 stopping..")
      break

for i in range(10):
  print("\n### TEST %d ###" % (i))
  #prompt = "THE HARE AND THE TORTOISE"
  #prompt = "THE STUDENT AND THE PROFESSOR"
  #prompt = "TWO men were traveling to"
  #prompt = "Anything you think might be inter"
  #prompt = "What is the perplexity of a language model that always predicts each character with equal probability of 1/V where V is the vocabulary size?"
  #prompt = "In this assignment, we looked at the next-word prediction in text as a sequence prediction problem. Give two other examples of sequence prediction problems that are not based on text. Examp"
  prompt = "What is the vanishing/exploding gradient problem? The vanishing/exploding gradie"# And why does this affect the models ability to learn long-term dependencies?"
  #out_len=10
  #out_len=400
  out_len=500
  print("\n--- ARGMAX ---")
  print(prompt, end='\n--ANSWER--\n')
  decode(model, prompt,out_len,False)
  
  print("\n--- SAMPLING ---")
  print(prompt, end='\n--ANSWER--\n')
  decode(model, prompt,out_len)
 
  
  print()


