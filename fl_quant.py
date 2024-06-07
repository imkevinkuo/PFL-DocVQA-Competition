import torch
Q_NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, 
             -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 
             0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 
             0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]

def quantQ(W, Q):
    # Quantizes W based on the bins in Q.
    with torch.no_grad():
        T = [None] + [(Q[i] + Q[i+1]) / 2 for i in range(len(Q)-1)] + [None]
    for i,q in enumerate(Q):
        if i == 0:
            qidx = W < T[1]
        elif i == len(Q)-1:
            qidx = W > T[-2]
        else:
            qidx = (W >= T[i]) & (W < T[i+1])
        W[qidx] = q
    return W

def quantize_blockwise(W, blocksize=64):
    # Splits W in blocks and then normalizes + quantizes each block.
    if W.abs().max() == 0:
        return W
    extra_elems = W.numel() % blocksize
    extra_zeros = blocksize - extra_elems if extra_elems > 0 else 0
    zero_pad = torch.zeros(extra_zeros).to(W.device)
    W_pad = torch.cat([W.flatten(), zero_pad]).reshape(-1, blocksize)
    W_scale = W_pad.abs().max(dim=1).values
    W_scale[W_scale == 0] = 1
    W_norm = W_pad / W_scale.reshape(-1, 1)
    qW = quantQ(W_norm, Q_NF4)
    qW = qW*W_scale.reshape(-1, 1)
    qW = qW.flatten()
    if extra_zeros > 0:
        qW = qW[:-extra_zeros]
    qW = qW.reshape(W.shape)
    return qW
