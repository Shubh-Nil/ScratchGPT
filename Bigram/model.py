import torch
from torch import nn
from torch.nn import functional as F

# baseline model
class BigramLanguageModel(nn.Module):                                                   # 'nn.Module' is the base-class for all the Neural-Networks
    '''A simple Bigram Language Model'''

    def __init__(self, vocab_size):
        '''
        - Each token (row) has 'n_embed' dimensions.
        - here, (n_embed = vocab_size)
        - This particular Embedding-Matrix, stores the 'next-token logits (prediction)' for each token.
        - We are predicting which token comes next, based on the embedding-vector of current token ONLY.
        - Therefore 'sequence_length' = 1 of each sample
        - Hence Bigram Language Model is the 1st "Baseline".
        '''

        super().__init__()
        # each token directly reads off the corresponding Next-token logits from the lookup-table
        self.n_embed = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embed)               # initially assign random-vectors for each token, later update during training

    def forward(self, idx, targets=None):                                               # default value of 'targets' is None
        '''
        'idx' and 'targets' are integer tensors of shape (B,T) where
            B (Batch) = batch_size (number of chunks),
            T (Time) = block_size (max. context length)
            n_embed = Embedding-dimension = vocab_size (in this Baseline!)

        idx --> token-IDs (from the Codebook) of all the T tokens of a chunk, for all the B chunks
        targets --> token-IDs (from the Codebook) of all the T target-tokens of a chunk, for all the B chunks
        '''
        logits = self.token_embedding_table(idx)                                        # outputs embedding-vector (next-token logits) of dim = n_embed for all the T tokens, for all the B chunks
                                                                                        # logits.shape = (B,T,vocab_size)

        # Loss calculation 
        if targets is None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size)                                       # Vertically Stack all the embedding-vectors (next-token logits)
                                                                                        # to align with pytorch's requirements for "F.cross_entropy()", which expects 'vocab_size' as the 2nd dimension
                                                                                        # logits --> 2D-tensor of shape (B*T, vocab_size)

            targets = targets.view(B*T)                                                 # targets --> 1D-tensor of shape(B*T)
            loss = F.cross_entropy(logits, targets)                                     # (prediction, ground-truth)
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
            logits, loss = self(idx)                                                    # get the predictions (next-token logits) from the current token ONLY
            logits = logits[:, -1, :]                                                   # We want the last generated prediction (for the new generated token) for each of the B chunks
                                                                                        # (B,T,vocab_size) --> (B,vocab_size)

            probs = F.softmax(logits, dim=-1)                                           # apply Softmax to get probabilities of next tokens
            idx_next = torch.multinomial(probs, num_samples=1)                          # sample 1 (next) token-ID for each of the B chunks, from the multinomial-distribution
                                                                                        # idx_next.shape ---> (B,)
            idx = torch.cat((idx, idx_next), dim=1)                                     # append the B new generated tokens to their corresponding chunks

        return idx
