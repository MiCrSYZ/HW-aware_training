"""
GRU model for AG News classification.

This module implements a 2-layer GRU architecture for text classification
on AG News dataset.
"""

import torch
import torch.nn as nn


class GRUAGNews(nn.Module):
    """
    2-layer GRU model for AG News classification.
    
    Architecture:
    - Embedding layer: vocab_size -> embed_dim
    - 2-layer GRU: hidden_dim=256
    - Classification head: hidden_dim -> num_classes
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # Classification head
        # If bidirectional, hidden_dim is doubled
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Linear(gru_output_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    # Input-to-hidden weights
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    # Hidden-to-hidden weights
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    # Initialize biases to zero
                    nn.init.constant_(param.data, 0.0)
    
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len] (token indices)
            lengths: Optional sequence lengths for padding [batch_size]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Embedding: [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        x = self.embedding(x)
        
        # Pack padded sequences if lengths provided (lengths must be on CPU)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # GRU forward: [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, hidden_dim]
        gru_out, hidden = self.gru(x)
        
        # Unpack if packed
        if isinstance(gru_out, nn.utils.rnn.PackedSequence):
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out, batch_first=True
            )
        
        # Use the last hidden state from the last layer
        # hidden shape: [num_layers * num_directions, batch_size, hidden_dim]
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]  # [batch_size, hidden_dim]
        
        # Classification head
        logits = self.head(last_hidden)  # [batch_size, num_classes]
        
        return logits
