import torch

# Hyper-Parameters
EPOCHS = 5000
BATCH_SIZE = 32                             # how many independent sequences(chunks) will we process in parallel?
block_size = 8                              # what is the maximum context length for predictions?
LEARNING_RATE = 1e-2                        # the model is simple, so the learning-rate can be high

eval_interval = 500       
eval_iterations = 200     

device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
torch.manual_seed(1337)

# File paths
input_file = '/home/shubhranil/ScratchGPT/input.txt'
