"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple

def RParam(*shape, backend: TensorBackend):
    r = 0.1 * (rand(shape, backend=backend) - 0.5)
    return Parameter(r)


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN ASSIGN3_2

        self.weights: Parameter = RParam(num_embeddings, embedding_dim, backend=backend)
        ### END ASSIGN3_2
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN ASSIGN3_2

        # Map word indices to one-hot vectors
        one_hot_vectors: Tensor = one_hot(x, self.num_embeddings)  # Shape: (batch_size, seq_len, num_embeddings)
        # Flatten to (batch_size * seq_len, num_embeddings) for matrix multiplication
        one_hot_flat = one_hot_vectors.view(bs * seq_len, self.num_embeddings)
        # Project to embedding vectors
        output_flat = one_hot_flat @ self.weights.value  # Shape: (batch_size * seq_len, embedding_dim)
        # Reshape back to (batch_size, seq_len, embedding_dim)
        output = output_flat.view(bs, seq_len, self.embedding_dim)
        # Verify output shape
        assert output.shape == (bs, seq_len, self.embedding_dim), f"Expected output shape {(bs, seq_len, self.embedding_dim)}, but got {output.shape}"
        return output
        
        ### END ASSIGN3_2

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        ### BEGIN ASSIGN3_2
        if not self.training or self.p_dropout == 0:
            return x

        mask = tensor_from_numpy(np.random.binomial(1, 1 - self.p_dropout, size=x.shape), backend=x.backend)
        output = x * mask / (1 - self.p_dropout)
        return output
        ### END ASSIGN3_2


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weight - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(1/in_size), 1/sqrt(1/in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(1/in_size), 1/sqrt(1/in_size)).
        """
        self.out_size = out_size
        ### BEGIN ASSIGN3_2
        self.weights = RParam(in_size, out_size, backend=backend)
        self.bias = RParam(out_size, backend=backend) if bias else Parameter(zeros((out_size,), backend=backend))
        ### END ASSIGN3_2

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        batch, in_size = x.shape
        ### BEGIN ASSIGN3_2
        # Reshape input x to be of size (batch, in_size)
        x = x.view(batch, in_size)
        # Reshape weights to be of size (in_size, out_size)
        weights: Tensor = self.weights.value.view(in_size, self.out_size)

        # Apply Matrix Multiplication on input x and self.weights, and reshape the output to be of shape (batch, self.out_size)
        out: Tensor = (x @ weights).view(batch, self.out_size)
        # Add bias
        return out + self.bias.value
        ### END ASSIGN3_2


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN ASSIGN3_2
        self.weights = Parameter(ones((dim,), backend=backend))
        self.bias = Parameter(zeros((dim,), backend=backend))
        ### END ASSIGN3_2

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        ### BEGIN ASSIGN3_2
        mean = x.sum(1) / dim
        std = ((x - mean.view(batch, 1))**2).sum(1) / dim
        
        x_normalized = (x - mean.view(batch, 1)) / ((std + self.eps) ** 0.5).view(batch, 1)
        return self.weights.value * x_normalized + self.bias.value
        ### END ASSIGN3_2
