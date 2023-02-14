"""
Code of this file is based on the implementations at: https://github.com/jrfiedler/xynn.
"""
import torch

from torch import nn


def xavier_linear(shape):
    """
    Create a tensor with given shape, initial with Xavier uniform weights, and convert to nn.Parameter

    Parameters
    ----------
    shape : tuple
        Shape of the weights to be created.

    Returns
    -------
    weights: nn.Parameter of shape (shape)
        Xavier uniform weights as `nn.Parameter`.
    """
    weights = torch.empty(shape)
    nn.init.xavier_uniform_(weights)
    weights = nn.Parameter(weights)
    return weights


class InnerProduct(nn.Module):
    """InnerProduct

    Inner product of embedded vectors, originally used in the product-based neural network (PNN) model [1].

    Parameters
    ----------
    n_fields : int
        Number of input fields.
    output_size : int, optional (default=10)
        Size of output after product and transformation.
    device : string or torch.device, optional (default="cpu")
        Device for storing and computations.

    References
    ----------
    [1] Qu, Y., Cai, H., Ren, K., Zhang, W., Yu, Y., Wen, Y. and Wang, J., 2016, December. Product-based neural
        networks for user response prediction. In 2016 IEEE 16th international conference on data mining (ICDM)
        (pp. 1149-1154). IEEE.
    """

    def __init__(
        self,
        n_fields,
        output_size=10,
        device="cpu",
    ):
        super().__init__()
        self.weights = xavier_linear((output_size, n_fields))
        self.to(device=device)

    def forward(self, x):
        """Inner product transformation of the input.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, n_fields, embedding_size)
            Input to be transformed.

        Return
        ------
        ip: torch.Tensor of shape (batch_size, output_size)
            Input after inner product transformation.
        """
        # r = # batch size
        # f = # fields
        # e = embedding size
        # p = product output size
        delta = torch.einsum("rfe,pf->rpfe", x, self.weights)
        ip = torch.einsum("rpfe,rpfe->rp", delta, delta)
        return ip


class OuterProduct(nn.Module):
    """OuterProduct

    Outer product of embedded vectors, originally used in the product-based neural network (PNN) model [1].

    Parameters
    ----------
    embedding_size : int
        Length of embedding vectors in input; all inputs are assumed to be embedded values.
    output_size : int, optional (default=10)
        Size of output after product and transformation.
    device : string or torch.device, optional (default="cpu")
        Device for storing and computations.

    References
    ----------
    [1] Qu, Y., Cai, H., Ren, K., Zhang, W., Yu, Y., Wen, Y. and Wang, J., 2016, December. Product-based neural
        networks for user response prediction. In 2016 IEEE 16th international conference on data mining (ICDM)
        (pp. 1149-1154). IEEE.
    """

    def __init__(self, embedding_size, output_size=10, device="cpu"):
        super().__init__()
        self.weights = xavier_linear((output_size, embedding_size, embedding_size))
        self.to(device=device)

    def forward(self, x):
        """Outer product transformation of the input.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, n_fields, embedding_size)
            Input to be transformed.

        Return
        ------
        op: torch.Tensor of shape (batch_size, output_size)
            Input after outer product transformation.
        """
        # r = # batch size
        # f = # fields
        # e, m = embedding size (two letters are needed)
        # p = product output size
        f_sigma = x.sum(dim=1)  # rfe -> re
        p = torch.einsum("re,rm->rem", f_sigma, f_sigma)
        op = torch.einsum("rem,pem->rp", p, self.weights)
        return op
