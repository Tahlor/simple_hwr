import torch
from torch.distributions import bernoulli, uniform

def stable_softmax(X, dim=2):
    max_vec = torch.max(X, dim, keepdim=True)
    exp_X = torch.exp(X - max_vec[0])
    sum_exp_X = torch.sum(exp_X, dim, keepdim=True)
    X_hat = exp_X / sum_exp_X
    return X_hat

def sample_batch_from_out_dist(y_hat, bias, gt_size=4, size=20):
    batch_size = y_hat.shape[0]
    split_sizes = [1] + [size] * 6 + [1]
    #print(split_sizes)
    y = torch.split(y_hat, split_sizes, dim=1)

    sos_prob = torch.sigmoid(y[0])
    eos_prob = torch.sigmoid(y[7])

    mixture_weights = stable_softmax(y[1] * (1 + bias), dim=1)
    mu_1 = y[2]
    mu_2 = y[3]
    std_1 = torch.exp(y[4] - bias)
    std_2 = torch.exp(y[5] - bias)
    correlations = torch.tanh(y[6])

    sos_dist = bernoulli.Bernoulli(probs=sos_prob)
    eos_dist = bernoulli.Bernoulli(probs=eos_prob)
    sos_sample = sos_dist.sample()
    eos_sample = eos_dist.sample()

    K = torch.multinomial(mixture_weights, 1).squeeze()

    mu_k = y_hat.new_zeros((y_hat.shape[0], 2))

    mu_k[:, 0] = mu_1[torch.arange(batch_size), K]
    mu_k[:, 1] = mu_2[torch.arange(batch_size), K]
    cov = y_hat.new_zeros(y_hat.shape[0], 2, 2)
    cov[:, 0, 0] = std_1[torch.arange(batch_size), K].pow(2)
    cov[:, 1, 1] = std_2[torch.arange(batch_size), K].pow(2)
    cov[:, 0, 1], cov[:, 1, 0] = (
        correlations[torch.arange(batch_size), K]
        * std_1[torch.arange(batch_size), K]
        * std_2[torch.arange(batch_size), K],
        correlations[torch.arange(batch_size), K]
        * std_1[torch.arange(batch_size), K]
        * std_2[torch.arange(batch_size), K],
    )

    X = torch.normal(
        mean=torch.zeros(batch_size, 2, 1), std=torch.ones(batch_size, 2, 1)
    ).to(y_hat.device)
    Z = mu_k + torch.matmul(cov, X).squeeze()

    sample = y_hat.new_zeros(batch_size, 1, gt_size)
    sample[:, 0, 2:3] = sos_sample
    sample[:, 0, 3:4] = eos_sample
    sample[:, 0, 0:2] = Z.squeeze()
    return sample

def sample_batch_from_out_dist_OLD(y_hat, bias):
    batch_size = y_hat.shape[0]
    split_sizes = [1] + [20] * 6
    y = torch.split(y_hat, split_sizes, dim=1)

    eos_prob = torch.sigmoid(y[0])
    mixture_weights = stable_softmax(y[1] * (1 + bias), dim=1)
    mu_1 = y[2]
    mu_2 = y[3]
    std_1 = torch.exp(y[4] - bias)
    std_2 = torch.exp(y[5] - bias)
    correlations = torch.tanh(y[6])

    bernoulli_dist = bernoulli.Bernoulli(probs=eos_prob)
    eos_sample = bernoulli_dist.sample()

    K = torch.multinomial(mixture_weights, 1).squeeze()

    mu_k = y_hat.new_zeros((y_hat.shape[0], 2))

    mu_k[:, 0] = mu_1[torch.arange(batch_size), K]
    mu_k[:, 1] = mu_2[torch.arange(batch_size), K]
    cov = y_hat.new_zeros(y_hat.shape[0], 2, 2)
    cov[:, 0, 0] = std_1[torch.arange(batch_size), K].pow(2)
    cov[:, 1, 1] = std_2[torch.arange(batch_size), K].pow(2)
    cov[:, 0, 1], cov[:, 1, 0] = (
        correlations[torch.arange(batch_size), K]
        * std_1[torch.arange(batch_size), K]
        * std_2[torch.arange(batch_size), K],
        correlations[torch.arange(batch_size), K]
        * std_1[torch.arange(batch_size), K]
        * std_2[torch.arange(batch_size), K],
    )

    X = torch.normal(
        mean=torch.zeros(batch_size, 2, 1), std=torch.ones(batch_size, 2, 1)
    ).to(y_hat.device)
    Z = mu_k + torch.matmul(cov, X).squeeze()

    sample = y_hat.new_zeros(batch_size, 1, 3)
    sample[:, 0, 0:1] = eos_sample
    sample[:, 0, 1:] = Z.squeeze()
    return sample
