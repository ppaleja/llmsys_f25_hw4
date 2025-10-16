import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .tensor_functions import rand
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32

def RParam(*shape, backend: TensorBackend=None):
    r = 0.1 * (rand(shape, backend=backend) - 0.5)
    return Parameter(r)


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=False, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd    : Dimensionality of embeddings and hidden states
            n_head    : Number of heads
            p_dropout : Dropout ratio for dropout layer
            causal    : If True, then apply a causal mask during self-attention
            bias      : If True, then apply a bias in Linear layers
        
        Attributes:
            q_projection   : Linear layer projecting input to Q matrix
            k_projection   : Linear layer projecting input to K matrix
            v_project      : Linear layer projecting input to V matrix
            out_projection : Linear output projection layer
            dropout        : Dropout layer
        """
        self.backend   = backend
        self.n_embd    = n_embd 
        self.n_head    = n_head
        self.causal    = causal
        self.attn_hidden_dim = n_embd // n_head
        self.use_fused_kernel = use_fused_kernel

        ### BEGIN ASSIGN3_3
        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)
        ### END ASSIGN3_3

    def create_causal_mask(self, bs, nh, seq_len):
        """
        return a 1x1xTxt triangular causal mask for Q @ K^T (which will get broadcasted to BxHxTxT)
        """
        # mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1) # This should be ok, but may be problematic -> the loss will be NaN in Assignment 3 because the mask will not broadcast correctly in the kernel.
        mask = -np.finfo(datatype).max * np.triu(np.ones((bs, nh, seq_len, seq_len), dtype=datatype), 1) # Correct version for Assignment 3.
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """Project x to Q, transpose of K, V for self attention
        
        Args:
            x: embeddings or hidden states (batch_size x seq_len x n_embd)

        Returns:
            Q   : The Query Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
            K^T : The Key Matrix Transposed (batch_size x num_heads x attn_hidden_dim x seq_len)
            V   : The Value Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        # 1. Reshape x to 2D for linear layers: (batch_size * seq_len, n_embd)
        x_2d = x.view(batch_size * seq_len, n_embd)
        
        # 2. Project input x to Q, K, V using respective linear layers
        q_full = self.q_projection(x_2d).view(batch_size, seq_len, n_embd)  # Shape: (batch_size, seq_len, n_embd)
        k_full = self.k_projection(x_2d).view(batch_size, seq_len, n_embd)  # Shape: (batch_size, seq_len, n_embd)
        v_full = self.v_projection(x_2d).view(batch_size, seq_len, n_embd)  # Shape: (batch_size, seq_len, n_embd)
        
        # 3. Reshape to (batch_size, seq_len, n_head, attn_hidden_dim)
        # and then transpose to (batch_size, n_head, seq_len, attn_hidden_dim)
        q = q_full.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        k = k_full.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        v = v_full.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        
        # 4. Transpose keys to shape (batch_size, n_head, attn_hidden_dim, seq_len)
        kT = k.permute(0, 1, 3, 2)
        ### END ASSIGN3_3
        
        return q, kT, v

    def self_attention(self, q, kT, v):
        """Given q, kT, and v of sizes defined above, return the result of MultiHeadAttention as described in the writeup
        softmax((q @ kT) / sqrt(attn_hidden_dim)) @ V.
        NOTE: We have added support for Batch Matrix Multiplication with 4 dimensions.
        This means given tensors A of shape (a, b, m, n) and B of shape (a, b, n, p), 
        A @ B will be of the shape (a, b, m, p). Take a moment to consider why we need it.

        Args:
            q  : Queries Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)
            kT : Keys Tensor of shape (batch_size x num_heads x attn_hidden_dim x seq_len)
            v  : Values Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None
        
        if not self.use_fused_kernel:
            ### BEGIN ASSIGN3_3
            # 1. Compute attention scores: q @ kT
            attention_scores = q @ kT  # Shape: (batch_size, num_head, seq_len, seq_len)
            
            # 2. Scale by sqrt(attn_hidden_dim)
            attention_scores = attention_scores / (q_dim ** 0.5)
            
            # 3. Apply causal mask if needed
            if self.causal:
                mask = self.create_causal_mask(batch_size, num_head, queries_len)
                attention_scores = attention_scores + mask
            
            # 4. Apply softmax along last dimension
            attention_weights = softmax(attention_scores, dim=3)
            
            # 5. Apply dropout
            attention_weights = self.dropout(attention_weights)
            
            # 6. Multiply by values
            attention_output = attention_weights @ v  # Shape: (batch_size, num_head, seq_len, attn_hidden_dim)
            
            # 7. Reshape: (batch_size, num_head, seq_len, attn_hidden_dim) -> (batch_size, seq_len, n_embd)
            attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
            attention_output = attention_output.view(batch_size, queries_len, self.n_embd)
            
            # 8. Apply output projection
            result = self.out_projection(attention_output.view(batch_size * queries_len, self.n_embd))
            result = result.view(batch_size, queries_len, self.n_embd)
            ### END ASSIGN3_3
        else:
            # BEGIN ASSIGN3_3
            raise NotImplementedError
            # END ASSIGN3_3

        return result

    def forward(self, x):
        """Computes MultiHeadAttention with causal masking if needed. 

        Args:
            x : Tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        # 1. Project to Q, K, V
        q, kT, v = self.project_to_query_key_value(x)
        
        # 2. Apply self-attention
        output = self.self_attention(q, kT, v)
        
        return output
        ### END ASSIGN3_3


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """The Feed Forward Module.
        
        Args:
            n_embd     : in_size of first linear layer and out_size of last linear layer
            middle_dim : out_size of first linear layer and in_size of last linear layer
            p_dropout  : Dropout probability
            bias       : If bias should be applied in linear layers
        
        Attributes:
            linear_in  : first linear layer
            linear_out : second linear layer
            dropout    : dropout layer
        """
        ### BEGIN ASSIGN3_3
        self.linear_in = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)
        ### END ASSIGN3_3

    def forward(self, x):
        """A FFN Module in a Pre-LN Transformer with GELU Activation and dropout.

        Args:
            x : Tensor of shape (batch_size x seq_len x n_embd)

        Returns:
            output : Tensor of shape (batch_size x seq_len x n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        ### BEGIN ASSIGN3_3
        # Reshape to 2D for linear layer
        x_2d = x.view(batch_size * seq_len, n_embd)
        
        # Apply first linear layer
        x_2d = self.linear_in(x_2d)
        
        # Reshape back to 3D for GELU
        x = x_2d.view(batch_size, seq_len, -1)
        
        # Apply GELU activation
        x = GELU(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Reshape to 2D for second linear layer
        x_2d = x.view(batch_size * seq_len, -1)
        
        # Apply second linear layer
        x_2d = self.linear_out(x_2d)
        
        # Reshape back to 3D
        x = x_2d.view(batch_size, seq_len, n_embd)
        ### END ASSIGN3_3

        return x

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-8, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False):
        super().__init__()
        """A Transformer Layer in a Pre-LN Transformer.

        Args: 
            n_embd : Dimensionality of embeddings and hidden states
            n_head : Number of heads for MultiHeadAttention
            p_dropout : Dropout ratio for dropout layer
            ln_eps : A value added for numerical stability in LayerNorm
            bias : If bias should be added in linear layers
        
        Attributes:
            ln_1 : First LayerNorm1d layer before MultiHeadAttention
            ln_2 : Second LayerNorm1d layer after MultiHeadAttention
            attention : MultiHeadAttention layer
            ff : FeedForward layer
        """
        
        ### BEGIN ASSIGN3_3
        self.attention = MultiHeadAttention(n_embd, n_head, causal=True, p_dropout=p_dropout, bias=bias, backend=backend, use_fused_kernel=use_fused_kernel)
        self.ff = FeedForward(n_embd, middle_dim=4*n_embd, p_dropout=p_dropout, bias=bias, backend=backend)
        ### END ASSIGN3_3

        self.use_fused_kernel = use_fused_kernel
        if not self.use_fused_kernel:
            ### BEGIN ASSIGN3_3
            self.ln_1 = LayerNorm1d(n_embd, ln_eps, backend)
            self.ln_2 = LayerNorm1d(n_embd, ln_eps, backend)
            ### END ASSIGN3_3
        else:
            # BEGIN ASSIGN3_3
            raise NotImplementedError
            # END ASSIGN3_3

    def forward(self, x):
        """
        The forward function of a Transformer Layer for a PRENORM Transformer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """
        batch_size, seq_len, x_dim = x.shape
        
        if not self.use_fused_kernel:
            ### BEGIN ASSIGN3_3
            # Apply first layer normalization
            x_norm1 = self.ln_1(x.view(batch_size * seq_len, x_dim)).view(batch_size, seq_len, x_dim) # Shape: (batch_size, seq_len, x_dim)
            
            # Apply multi-head attention
            attn_out = self.attention(x_norm1)  # Shape: (batch_size, seq_len, x_dim)
            x = x + attn_out  # Residual connection
            
            # Apply second layer normalization
            x_norm2 = self.ln_2(x.view(batch_size * seq_len, x_dim)).view(batch_size, seq_len, x_dim) # Shape: (batch_size, seq_len, x_dim)
            
            # Apply feed-forward network
            ff_out = self.ff(x_norm2)  # Shape: (batch_size, seq_len, x_dim)
            x = x + ff_out  # Residual connection
            ### END ASSIGN3_3
        else:
            # BEGIN ASSIGN3_3
            raise NotImplementedError
            # END ASSIGN3_3

        return x


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None,
        use_fused_kernel: bool=False,
    ):
        super().__init__()
        """A Full Decoder-only Pre-LN Transformer with 4 Transformer Layers.

        Args:
            n_vocab : Vocabulary size defines the number of different tokens that can be represented by the input.
            n_embd  :  Dimensionality of the embeddings and hidden states.
            n_head  : Number of attention heads for each attention layer in the Transformer.
            n_positions : The maximum sequence length that this model might ever be used with.
            p_dropout : The dropout ratio for any dropout layer.
            ln_eps : The epsilon to use in the layer normalization layers.
            bias : If linear layers should include a bias.
        
        Attributes:
            token_embeddings : Embedding layer for tokens.
            position_embeddings : Embedding layer for token positions.
            t_layer_1 : 1st Transformer Layer.
            t_layer_2 : 2nd Transformer Layer.
            t_layer_3 : 3rd Transformer Layer.
            t_layer_4 : 4th Transformer Layer.
            dropout : Dropout layer before first transformer layer.
            ln : LayerNorm layer after last transformer layer.
            lm_head : Linear layer for projection from (*, n_embd) to (*, n_vocab)
        """
        self.backend             = backend
        self.n_embd              = n_embd
        self.n_vocab             = n_vocab
        
        ### BEGIN ASSIGN3_3
        self.token_embeddings = Embedding(n_vocab, n_embd, backend=backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend=backend)
        self.t_layer_1 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel)
        self.t_layer_2 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel)
        self.t_layer_3 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel)
        self.t_layer_4 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel)
        self.dropout = Dropout(p_dropout)
        self.lm_head = Linear(n_embd, n_vocab, bias=False, backend=backend)
        ### END ASSIGN3_3

        self.use_fused_kernel = use_fused_kernel
        if not self.use_fused_kernel:
            ### BEGIN ASSIGN3_3
            self.ln = LayerNorm1d(n_embd, ln_eps, backend)
            ### END ASSIGN3_3
        else:
            # BEGIN ASSIGN3_3
            raise NotImplementedError
            # END ASSIGN3_3
        
    def forward(self, idx):
        """A Forward pass of a Decoder-only Transformer Language model.
        Args: 
            idx: input of shape (batch_size, seq_len)
        
        Returns: 
            logits: logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape
        pos = tensor([i for i in range(seq_len)], backend=self.backend).view(1, seq_len)

        if not self.use_fused_kernel:
            ### BEGIN ASSIGN3_3
            # 1. Token embeddings: (batch_size, seq_len, n_embd)
            token_emb = self.token_embeddings(idx)

            # 2. Positional embeddings: create ids (1, seq_len) and embed
            position_ids = tensor_from_numpy(np.arange(seq_len, dtype=datatype).reshape(1, seq_len), backend=self.backend)
            pos_emb = self.position_embeddings(position_ids)  # (1, seq_len, n_embd)

            # 3. Add token and positional embeddings (broadcast pos_emb over batch)
            x = token_emb + pos_emb

            # 4. Dropout
            x = self.dropout(x)

            # 5. Transformer layers
            x = self.t_layer_1(x)
            x = self.t_layer_2(x)
            x = self.t_layer_3(x)
            x = self.t_layer_4(x)

            # 6. Final layer norm expects 2D input: flatten tokens, apply norm, restore
            x = self.ln(x.view(batch_size * seq_len, self.n_embd))

            # 7. Project to vocabulary logits
            logits = self.lm_head(x)
            return logits.view(batch_size, seq_len, self.n_vocab)
            ### END ASSIGN3_3
        else:
            # BEGIN ASSIGN3_3
            raise NotImplementedError
            # END ASSIGN3_3

        return x
