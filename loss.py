import numpy as np
import torch

# maximum likelihood estimation
# reference: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
#            https://blog.csdn.net/saltriver/article/details/53364037

def log_likelihood(eos_hat, weights, mu1, mu2, sigma1, sigma2, rho, y, masks):
    eos = y.narrow(-1, 0, 1)  # end of stroke
    x1 = y.narrow(-1, 1, 1)   # coordinate point 1
    x2 = y.narrow(-1, 2, 1)   # coordinate point 2
    eps = np.finfo(float).eps

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

    eos_ewl = (
        eos_hat * (eos + eps) + (1 - eos_hat) * (1 - eos + eps)
    ).log().squeeze()

    loss = (torch.sum(pos_ewl * masks) + torch.sum(eos_ewl * masks)) / torch.sum(masks)
    return -1 * loss
