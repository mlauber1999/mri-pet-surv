import torch
from torch import nn, Tensor
import itertools

def sur_loss(preds, obss, hits, bins=Tensor([[0, 24, 48, 108]])):
    bin_centers = (bins[0, 1:] + bins[0, :-1])/2
    survived_bins_censored = torch.ge(torch.mul(obss.view(-1, 1),1-hits.view(
            -1,1)), bin_centers)
    survived_bins_hits = torch.ge(torch.mul(obss.view(-1,1), hits.view(-1,1)),
                                  bins[0,1:])
    survived_bins = torch.logical_or(survived_bins_censored, survived_bins_hits)
    survived_bins = torch.where(survived_bins, 1, 0)
    event_bins = torch.logical_and(torch.ge(obss.view(-1, 1), bins[0, :-1]),
                 torch.lt(obss.view(-1, 1), bins[0, 1:]))
    event_bins = torch.where(event_bins, 1, 0)
    hit_bins = torch.mul(event_bins, hits.view(-1, 1))
    l_h_x = 1+survived_bins*(preds-1)
    n_l_h_x = 1-hit_bins*preds
    cat_tensor = torch.cat((l_h_x, n_l_h_x), axis=0)
    total = -torch.log(torch.clamp(cat_tensor, min=1e-12))
    pos_sum = torch.sum(total)
    neg_sum = torch.sum(pos_sum)
    return neg_sum

def combinations(iterable, r):
    # modified from https://docs.python.org/3/library/itertools.html#itertools.combinations
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = iterable
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield pool[indices]
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield pool[indices]

def cox_loss_discrete(risk_pred, y, e):
    y, idx = torch.sort(y.squeeze(), descending=True)
    risk_pred = risk_pred.squeeze()[idx]
    e = e.squeeze()[idx]
    exp_vals = torch.exp(risk_pred)
    unique_times = torch.unique(y)
    second_term = Tensor([])
    for i in unique_times:
        count = torch.sum(e[y == i])
        if count == 0:
            continue
        torch_values = exp_vals[y == i]
        #compute n choosek combinations, take sums
        nchoosek = Tensor([0])
        for c in combinations(torch_values, count):
            nchoosek = nchoosek + torch.sum(c)
        print(nchoosek)
        torch_log_sum = torch.log(nchoosek)
        second_term = torch.cat([second_term, torch_log_sum])
    second_term = torch.sum(second_term)
    first_term = torch.sum(risk_pred*e)
    output = (second_term-first_term)/torch.sum(e)
    return output

def cox_loss_tf(risk_pred, y, e):
    y, idx = torch.sort(y.squeeze(), descending=True)
    risk_pred = risk_pred.squeeze()[idx]
    e = e.squeeze()[idx]
    summand = torch.logcumsumexp(risk_pred, axis=0)
    unique_times = torch.unique(y)
    second_term = Tensor([])
    for i in unique_times:
        count = torch.sum(e[y == i])
        max_value = torch.max(summand[y == i])
        print(count*max_value)
        second_term = torch.cat([second_term, Tensor([count*max_value])])
    second_term = torch.sum(second_term)
    first_term = torch.sum(risk_pred*e)
    output = (second_term-first_term)/torch.sum(e)
    return output