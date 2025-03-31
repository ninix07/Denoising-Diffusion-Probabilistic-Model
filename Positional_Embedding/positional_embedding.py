import torch

def time_embedding(time_step,time_embedding_dim ):
    '''
    The function returns positional embedding for a given timestep t.
    time_step: 1D tensor
    It computes furequency scaling factor as 1000 ^ -2*i/d
    i-> index and d-> embedding dimension.

    Finally sinusodial embedding is returned with final output shape
    [batch_size, time_embedding_dim] (by combining two time_embedding_dim//2).
    '''
    
    factor = 10000 ** (-torch.arange(
    start=0, end=time_embedding_dim//2, device=time_step.device
    ) / (time_embedding_dim//2))    
    time_emb =time_step[:,None].repeat(1,time_embedding_dim//2)/factor
    time_emb= torch.cat(tensors=[torch.sin(time_emb),torch.cos(time_emb)],dim=-1)
    return time_emb