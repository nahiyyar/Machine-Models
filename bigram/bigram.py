# A simple bigram character-level language model implemented in NumPy.

import numpy as np
embed_dim = 27

text = open('shakspeare.txt').read()
chars = sorted(set(text)) 
vocab_size = len(chars)

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

xs = np.array([stoi[text[i]] for i in range(len(text)-1)], dtype=np.int64)
ys = np.array([stoi[text[i+1]] for i in range(len(text)-1)], dtype=np.int64)

W_embed = np.random.randn(vocab_size, embed_dim) * 0.01 
W = np.random.randn(embed_dim,  vocab_size) * 0.01 

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy(probs, targets):
    return -np.log(probs[np.arange(len(targets)), targets] + 1e-8).mean()

lr = 0.1

for epoch in range(200):                         

    embeds = W_embed[xs]                       
    logits = embeds @ W                         
    probs = softmax(logits)                     
    loss = cross_entropy(probs, ys)

    d_logits = probs.copy()        
    d_logits[np.arange(len(ys)), ys] -= 1       
    d_logits /= len(ys)              

    dW = embeds.T @ d_logits               
    d_embed = d_logits @ W.T                     

    dW_embed = np.zeros_like(W_embed)
    np.add.at(dW_embed, xs, d_embed)

    W -= lr * dW
    W_embed -= lr * dW_embed

    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  loss: {loss:.4f}")

def generate(start_char, n=200):
    result = [start_char]
    idx = stoi[start_char]
    for _ in range(n):
        embed = W_embed[idx]
        logits = embed @ W
        probs = softmax(logits)
        idx = np.random.choice(vocab_size, p=probs)
        result.append(itos[idx])
    return ''.join(result)

print(generate('h'))

