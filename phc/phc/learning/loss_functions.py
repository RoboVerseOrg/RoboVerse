import torch 

def kl_multi(qm, qv, pm, pv):
    """
    q: posterior
    p: prior
    ​
    """
    element_wise = 0.5 * (pv - qv + qv.exp() / pv.exp() + (qm - pm).pow(2) / pv.exp() - 1)
    kl = element_wise.sum(-1)
    return kl