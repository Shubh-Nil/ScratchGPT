import torch
from torch import nn
from torch.nn import functional as F

# Hyper-Parameters
EPOCHS = 5000
BATCH_SIZE = 64       #32   # how many independent sequences(chunks) will we process in parallel? (B)
block_size = 256      #8    # what is the maximum context length for predictions? (T)
LEARNING_RATE = 3e-4        # reduced learning-rate compared to bigram.py, 
                            # because the self-attention layer cannot tolerate very high learning-rates
emb_size = 384        #32   # Embedding-dimensions
n_heads = 6           #4    # therefore, head_size = 384/6 = 64
n_blocks = 6          #3

dropout = 0.2               # "dropout" applied to FeedForward, Head, MultiHeadAttention
eval_interval = 500
eval_iterations = 200

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)


# run the below line on the Terminal, in case you don't have the "input.txt"
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# here are all the unique characters that occur in this text
vocab = sorted(set(text))  # sorted() will return a 'list' -> ['\n', '!', ..., 'A', ... , 'z']
vocab_size = len(vocab)
# create a mapping (Codebook) from characters to integers
stoi = {ch:i for i,ch in enumerate(vocab)}
encoder = lambda s: [stoi[ch] for ch in s]                  # 'encoder' function: take a string, output a list of integers
# create a mapping from integers to characters
itos = {i:ch for i,ch in enumerate(vocab)}
decoder = lambda l: ''.join([itos[i] for i in l])           # 'decoder' function: take a list of integers, output a string


# Train and Validation splits
data = torch.tensor(encoder(text), dtype=torch.long)        # whole 'text' data --> 'list' (of token IDs) --> 'Tensor' (of token IDs)
n = int(0.9 * len(data))        # first 90% will be Train, rest 10% Val
data_train = data[:n]
data_val = data[n:]


# DataLoader
def get_batch(split, batch_size, block_size):
    # generate a small batch of data of inputs x and targets y
    data = data_train if split == 'train' else data_val
    ix = torch.randint(len(data)-block_size, (batch_size,))                 # selecting random starting points for 'batch_size' number of chunks

    x = torch.stack([data[i: i+block_size] for i in ix])                    # Maximum Context for each chunk
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])                # All possible targets for the corresponding possible sequence out of the chunk
    x, y = x.to(device), y.to(device)
    return x,y


'''
* For Training: calculating loss for a batch of chunks, 
                and then doing gradient-descent in that direction in enough
* For Evaluation: More accurate loss calculation is done, 
                  if loss is calculated for a lot of chunks (many batches), and then taken average
'''
@torch.no_grad()
def estimate_loss(model, eval_iterations, batch_size, block_size):
    losses = {'train':0, 'valid':0}
    model.eval()
    for split in ['train', 'valid']:
        for k in range(eval_iterations):
            X, Y = get_batch(split = split,
                             batch_size = batch_size,
                             block_size = block_size)            # sample a batch of data
            logits, loss = model(X, Y)
            losses[split] += loss.item()
        losses[split] /= eval_iterations
    model.train()
    return losses


class Head(nn.Module):
    '''One Head of self-attention'''

    def __init__(self, head_size, emb_size, block_size):
        super().__init__()
        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))        # 'tril' is considered a buffer tensor
                                                                                            # which should not be considered as a parameter for gradient computation
                                                                                            # because 'tril' is just a mask, to mask-out future weights
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,emb_size = x.shape
        q = self.query(x)             # (B,T,head_size)
        k = self.key(x)               # (B,T,head_size)
        v = self.value(x)             # (B,T,head_size)

        # compute attention-scores ("affinities")
        weights = q @ k.transpose(-2, -1) * emb_size**-0.5                              # (B,T,head_size) @ (B,head_size,T) -----> (B,T,T)
        weights = weights.masked_fill(self.tril[:T,:T]==0, float('-inf'))               # here T = block_size,
                                                                                        # self.tril == 0 can also be written 
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        # perform the Weighted-Average (out_single) of the 'Values'
        out_single = weights @ v                                                        # (B,T,T) @ (B,T,head_size) -----> (B,T,head_size)
        return out_single


class MultiHeadAttention(nn.Module):
    '''Multiple Heads of Self-Attention in parallel'''
    
    def __init__(self, n_heads, head_size):
        super().__init__()
        # initializes 'n_heads' instances of "Head" class
        self.multi_heads = nn.ModuleList([Head(head_size, emb_size, block_size) for _ in range(n_heads)])  
        self.proj = nn.Linear(emb_size, emb_size)                                                   # head_size * n_heads = emb_size
        self.dropout = nn.Dropout(dropout)          

    def forward(self, x):
        # head(x) returns Weighted-Average (out_single) of the 'Values'
        # multiple heads return Weighted-Average of different 'Values'
        out = torch.cat([head(x) for head in self.multi_heads], dim=-1)                             # (B,T,head_size) concat 'num_heads' times along dim=2
                                                                                                    # (B,T,head_size * num_heads) = (B,T,emb_size)
        out = self.dropout(self.proj(out))                                                          # (B,T,emb_size)
        return out


class FeedForward(nn.Module):
    '''a simple linear-layer followed by non-linearity'''
    def __init__(self, emb_size):
        super().__init__()
        '''
        mentioned in the Paper:
        inner-layer of FFNN has 4 times dim. than input, and output layers
        PAPER ---> dim(input) = dim(output) = 512, dim(inner layer) = 2048
        '''
        self.net = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),                                                      
            nn.ReLU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)                                                                          # (B,T,emb_size) -----> (B,T,emb_size)


class Block(nn.Module):
    '''Transformer Block: Communication followed by Computation'''
    def __init__(self, emb_size, n_heads):
        super().__init__()

        head_size = emb_size//n_heads
        self.self_attention = MultiHeadAttention(n_heads, head_size)                               # 'n_heads' heads of 'head_size' dim. each  -----> emb_size
        self.ffwd = FeedForward(emb_size)
        # 2 LayerNorm layers because their trainable parameters - beta, gamma 
        # are different for different Transformations layers (ffwd, self-attention etc)
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)


    def forward(self, x):
        # Residual-connection applied
        # In the Paper, LayerNorm applied to the input "After" the Transformations
        # but Now it is common to apply LayerNorm "Before" the Transformations
        x = x + self.self_attention(self.ln1(x))                                                   # (B,T,emb_size)
        x = x + self.ffwd(self.ln2(x))                                                             # (B,T,emb_size)
        return x


# Baseline Model
class GPTLanguageModel(nn.Module):                                                              # 'nn.Module' is the base-class for all the Neural-Networks

    def __init__(self, vocab_size):
        '''
        - Each token (row) has 'vocab_size' dimensions.
        - This particular Embedding-Matrix, stores the 'next-token logits (prediction)' for each token.
        - We are predicting which token comes next, based on the embedding-vector of current token ONLY.
        - Therefore 'sequence_length' = 1 of each sample
        - Hence Bigram Language Model is the 1st "Baseline".
        '''
        super().__init__()
        # each token reads off the token-embeddings from the lookup-table,
        self.token_embeding_table = nn.Embedding(vocab_size, emb_size)                              # initially assign random-vectors for each token, later update during training
        self.position_embeding_table = nn.Embedding(block_size, emb_size)

        self.blocks = nn.Sequential(*[Block(emb_size, n_heads) for _ in range(n_blocks)])           # Multiple blocks (deep Neural Nets)                                              
                                                                                                    # '*' is unpacking the elements of the list, to pass them as separate arguments to nn.Sequential()
        self.lnf = nn.LayerNorm(emb_size)                                                           # final LayerNorm
        self.linear_head = nn.Linear(emb_size, vocab_size)

    def forward(self, idx, targets=None):                                                           # default value of 'targets' is None
        '''
        'idx' and 'targets' are integer tensors of shape (B,T) where
              B (Batch) = batch_size (number of chunks),
              T (Time) = block_size (max. context length)
              emb_size = Embedding-dimension

        idx --> token-IDs (from the Codebook) of all the T tokens of a chunk, for all the B chunks
        targets --> token-IDs (from the Codebook) of all the T target-tokens of a chunk, for all the B chunks
        '''
        B, T = idx.shape
        token_emb = self.token_embeding_table(idx)                                              # (B,T,emb_size)                                                                                            
        pos_emb = self.position_embeding_table(torch.arange(T, device=device))                  # (T,emb_size)

        x = token_emb + pos_emb                                                                 # (B,T,emb_size) + (T,emb_size) -----> (B,T,emb_size)
                                                                                                # pos_emb.shape gets broadcasted to (B,T,emb_size)
        x = self.blocks(x)                                                                      # (B,T,emb_size)
        x = self.lnf(x)                                                                         # (B,T,emb_size)
        logits = self.linear_head(x)                                                            # Next-token logits          
                                                                                                # (B,T,emb_size) -----> (B,T,vocab_size)
                                                                                                                                                
        # Loss calculation 
        if targets is None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size)                # Vertically Stack all the embedding-vectors (next-token logits)
                                                                # to align with pytorch's requirements for "F.cross_entropy()", which expects 'vocab_size' as the 2nd dimension
                                                                # logits --> 2D-tensor of shape (B*T, vocab_size)

            targets = targets.view(B*T)                 # targets --> 1D-tensor of shape(B*T)
            loss = F.cross_entropy(logits, targets)     # (prediction, ground-truth)
                                                        # 'loss' is a single float value
            # target will be converted to one-hot-encoding or embedding-vector ???

        return logits, loss

    def generate(self, idx, max_new_tokens):
        '''
        'idx' is integer tensor of shape (B,T)
        It contains token-IDs of all the T tokens, of all the B chunks

        It takes the token IDs of shape (B,T), generate (T+1)th token for all the B chunks, and then append it to the END
        In this way, the length of B chunks increases from T -> T+1 -> T+2 and so on
        '''
        for _ in range(max_new_tokens):
            
            # crop idx to the last block_size tokens
            # because if len(idx) > block_size, then pos_emb will run out of scope
            idx_crop = idx[:, -block_size:] 

            logits, loss = self.forward(idx_crop)                       # get the predictions (next-token logits) from the current token ONLY
            logits = logits[:, -1, :]                                   # We want the last generated prediction (for the new generated token) for each of the B chunks
                                                                        # (B,T,vocab_size) --> (B,vocab_size)

            probs = F.softmax(logits, dim=-1)                           # (B,vocab_size)
                                                                        # apply Softmax to get probabilities of next tokens
            idx_next = torch.multinomial(probs, num_samples=1)          # sample 1 (next) token-ID for each of the B chunks, from the multinomial-distribution
                                                                        # idx_next.shape ---> (B,)
            idx = torch.cat((idx, idx_next), dim=1)                     # append the B new generated tokens to their corresponding chunks

        return idx


model = GPTLanguageModel(vocab_size)
model = model.to(device)

# Create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(),
                              lr = LEARNING_RATE)

# TRAINING
for epoch in range(EPOCHS):
    # every once in a while evaluate the loss on 'train' and 'val' sets
    if epoch % eval_interval == 0 or epoch == EPOCHS-1:
          losses = estimate_loss(model = model,
                                 eval_iterations = eval_iterations,
                                 batch_size = BATCH_SIZE,
                                 block_size = block_size)
          print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['valid']:.4f}")

    xb, yb = get_batch(split='train',
                       batch_size = BATCH_SIZE,
                       block_size = block_size)            # sample a batch of data
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1,1), dtype=torch.long, device=device)
max_new_tokens = 500
print(decoder(model.generate(context, max_new_tokens)[0].tolist()))
