import torch


# run this line on the Terminal to download the 'input.txt' file
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
def load_data(input_file):
    '''Load text data from a file.'''
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def preprocess_data(text):
    '''Preprocess text into encoded Tokens-IDs.'''

    # here are all the unique characters that occur in this text
    vocab = sorted(set(text))                                                       # sorted() will return a 'list' -> ['\n', '!', ..., 'A', ... , 'z']
    vocab_size = len(vocab)

    # create a mapping (Codebook) from characters to integers
    stoi = {ch: i for i, ch in enumerate(vocab)}
    encoder = lambda s: [stoi[ch] for ch in s]                                      # 'encoder' function: take a string, output a list of integers
                                                                                    
    # create a mapping from integers to characters
    itos = {i: ch for i, ch in enumerate(vocab)}
    decoder = lambda l: ''.join([itos[i] for i in l])                               # 'decoder' function: take a list of integers, output a string

    return vocab, vocab_size, encoder, decoder


def create_splits(data, split_ratio=0.9):
    '''Create training and validation splits.'''

    n = int(split_ratio * len(data))
    data_train = data[:n]
    data_val = data[n:]
    return data_train, data_val


def get_batch(split, data_train, data_val, batch_size, block_size, device):
    '''Generate a batch of data of inputs x and targets y, for training or validation.'''

    data = data_train if split == 'train' else data_val
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


'''
* For Training: calculating loss for a batch of chunks, 
                and then doing gradient-descent in that direction in enough
* For Evaluation: More accurate loss calculation is done, 
                  if loss is calculated for a lot of chunks (many batches), and then taken average
'''
@torch.no_grad()
def estimate_loss(model, eval_iterations, batch_size, block_size, data_train, data_val, device):
    '''Estimate training and validation loss.'''

    losses = {'train': 0, 'valid': 0}
    model.eval()

    for split in ['train', 'valid']:
        for _ in range(eval_iterations):
            x, y = get_batch(split, data_train, data_val, batch_size, block_size, device)                   # sample a batch of data
            logits, loss = model(x, y)
            losses[split] += loss.item()
        losses[split] /= eval_iterations

    model.train()
    return losses
