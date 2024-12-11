import torch
from torch.optim import AdamW
from config import *
from model import *
from utils import *


# Load and Preprocess data
text = load_data(input_file)
vocab, vocab_size, encoder, decoder = preprocess_data(text)
data = torch.tensor(encoder(text), dtype=torch.long)                        # whole 'text' data --> 'list' (of token IDs) --> 'Tensor' (of token IDs)
data_train, data_val = create_splits(data)


# Initialize model and optimizer
model = BigramLanguageModel(vocab_size).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)


# TRAINING
for epoch in range(EPOCHS):
    # every once in a while evaluate the loss on 'train' and 'val' sets
    if epoch % eval_interval == 0:
        losses = estimate_loss(model, eval_iterations, BATCH_SIZE, block_size, data_train, data_val, device)
        print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['valid']:.4f}")

    # Get Training-Batch
    xb, yb = get_batch('train', data_train, data_val, BATCH_SIZE, block_size, device)
    logits, loss = model(xb, yb)

    # Backpropagation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# Generate Text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = model.generate(context, max_new_tokens=500)
print(decoder(generated_tokens[0].tolist()))
