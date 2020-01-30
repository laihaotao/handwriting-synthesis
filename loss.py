import numpy as np
import torch


# maximum likelihood estimation
# reference: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
#            https://arxiv.org/abs/1706.03762
#            https://blog.csdn.net/saltriver/article/details/53364037


def neg_log_likelihood(y_pred, y_true, mask):
    eos_pred, weights, mu1, mu2, sigma1, sigma2, rho = y_pred
    eos_true = y_true.narrow(-1, 0, 1)  # end of stroke
    x1 = y_true.narrow(-1, 1, 1)   # coordinate point 1
    x2 = y_true.narrow(-1, 2, 1)   # coordinate point 2
    eps = np.finfo(float).eps      # a tiny flaot number

    # formula (23) ~ (26) from the paper
    z = (
        ((x1 - mu1) / sigma1) ** 2 + ((x2 - mu2) / sigma2) ** 2
        - 2 * rho * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
    )
    n1 = 1 / (2 * np.pi * sigma1 * sigma2 * torch.sqrt(1 - rho**2))
    n2 = -1 * z / (2 * (1 - rho**2))
    n = n1 * torch.exp(n2)

    # sum on the feature dimension
    pos_ewl = ((weights * n).sum(dim=-1) + eps).log()
    # Bernoulli distribution
    eos_ewl = (
        eos_pred * (eos_true + eps) + (1 - eos_pred) * (1 - eos_true + eps)
    ).log().squeeze()

    loss = (torch.sum(pos_ewl * mask) + torch.sum(eos_ewl * mask)) / torch.sum(mask)
    return -1 * loss
