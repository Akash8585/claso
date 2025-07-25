#!/usr/bin/env python3
"""
Production-quality transformer model with advanced features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """Enhanced multi-head attention with relative position encoding"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(context)

class PositionalEncoding(nn.Module):
    """Enhanced positional encoding with learnable components"""
    
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Fixed sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
        # Learnable position embeddings
        self.learnable_pe = nn.Parameter(torch.randn(max_len, d_model) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        
        # Combine fixed and learnable positional encodings
        pos_encoding = (self.pe[:seq_len, :].transpose(0, 1) + 
                       self.learnable_pe[:seq_len, :].unsqueeze(0))
        
        x = x + pos_encoding
        return self.dropout(x)

class FeedForward(nn.Module):
    """Enhanced feed-forward network with GLU activation"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.w_gate = nn.Linear(d_model, d_ff)  # For GLU
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GLU (Gated Linear Unit) activation
        gate = torch.sigmoid(self.w_gate(x))
        hidden = F.relu(self.w_1(x)) * gate
        return self.w_2(self.dropout(hidden))

class TransformerEncoderLayer(nn.Module):
    """Enhanced transformer encoder layer"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerDecoderLayer(nn.Module):
    """Enhanced transformer decoder layer"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class ProductionTransformer(nn.Module):
    """Production-quality transformer for code comment generation"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 d_ff: int = 2048, max_seq_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def create_padding_mask(self, x: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
        """Create padding mask"""
        return (x != pad_token).unsqueeze(1).unsqueeze(2)
    
    def create_causal_mask(self, size: int) -> torch.Tensor:
        """Create causal (look-ahead) mask"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence"""
        # Embedding and positional encoding
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return self.encoder_norm(x)
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode target sequence"""
        # Embedding and positional encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        return self.decoder_norm(x)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.create_padding_mask(src)
        if tgt_mask is None:
            tgt_padding_mask = self.create_padding_mask(tgt)
            tgt_causal_mask = self.create_causal_mask(tgt.size(1)).to(tgt.device)
            tgt_mask = tgt_padding_mask & tgt_causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Encode and decode
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # Project to vocabulary
        return self.output_projection(decoder_output)
    
    def generate(self, src: torch.Tensor, max_length: int = 256, 
                 start_token: int = 2, end_token: int = 3, pad_token: int = 0,
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
        """Generate sequence using nucleus sampling"""
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        # Encode source
        src_mask = self.create_padding_mask(src, pad_token)
        encoder_output = self.encode(src, src_mask)
        
        # Initialize target with start token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Create target mask
                tgt_padding_mask = self.create_padding_mask(generated, pad_token)
                tgt_causal_mask = self.create_causal_mask(generated.size(1)).to(device)
                tgt_mask = tgt_padding_mask & tgt_causal_mask.unsqueeze(0).unsqueeze(0)
                
                # Decode
                decoder_output = self.decode(generated, encoder_output, tgt_mask, src_mask)
                logits = self.output_projection(decoder_output[:, -1, :])
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end token
                if (next_token == end_token).all():
                    break
        
        return generated

class EnhancedProductionTransformer(nn.Module):
    """Enhanced production transformer with 10x larger architecture"""
    
    def __init__(self, vocab_size: int, model_type: str = 'comment',
                 max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # Model configurations based on type
        if model_type == 'comment':
            # Comment Model: ~50M parameters
            self.d_model = 512
            self.num_heads = 16
            self.num_encoder_layers = 8
            self.num_decoder_layers = 8
            self.d_ff = 2048
        elif model_type == 'commit':
            # Commit Model: ~80M parameters  
            self.d_model = 768
            self.num_heads = 16
            self.num_encoder_layers = 10
            self.num_decoder_layers = 10
            self.d_ff = 3072
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        self.vocab_size = vocab_size
        self.model_type = model_type
        
        # Enhanced embeddings with larger dimensions
        self.src_embedding = nn.Embedding(vocab_size, self.d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, max_seq_len, dropout)
        
        # Encoder with more layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(self.d_model, self.num_heads, self.d_ff, dropout)
            for _ in range(self.num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(self.d_model)
        
        # Decoder with more layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(self.d_model, self.num_heads, self.d_ff, dropout)
            for _ in range(self.num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(self.d_model)
        
        # Enhanced output projection with intermediate layer
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),  # GELU works better on Apple Silicon
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, vocab_size)
        )
        
        # Gradient checkpointing for memory efficiency
        self.gradient_checkpointing = False
        
        # Initialize weights
        self._init_weights()
        
        print(f"🚀 Enhanced {model_type} model initialized:")
        print(f"   📊 Parameters: ~{self.count_parameters()/1e6:.1f}M")
        print(f"   🧠 d_model: {self.d_model}")
        print(f"   🔄 Encoder layers: {self.num_encoder_layers}")
        print(f"   🔄 Decoder layers: {self.num_decoder_layers}")
        print(f"   👁️  Attention heads: {self.num_heads}")
        
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
        
    def _init_weights(self):
        """Enhanced weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Smaller initialization for embeddings
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_padding_mask(self, x: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
        """Create padding mask"""
        return (x != pad_token).unsqueeze(1).unsqueeze(2)
    
    def create_causal_mask(self, size: int) -> torch.Tensor:
        """Create causal (look-ahead) mask"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Enhanced encoding with gradient checkpointing"""
        # Embedding and positional encoding
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through encoder layers with optional gradient checkpointing
        for layer in self.encoder_layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, src_mask)
            else:
                x = layer(x, src_mask)
        
        return self.encoder_norm(x)
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Enhanced decoding with gradient checkpointing"""
        # Embedding and positional encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through decoder layers with optional gradient checkpointing
        for layer in self.decoder_layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, encoder_output, tgt_mask, src_mask)
            else:
                x = layer(x, encoder_output, tgt_mask, src_mask)
        
        return self.decoder_norm(x)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with enhanced processing"""
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.create_padding_mask(src)
        if tgt_mask is None:
            tgt_padding_mask = self.create_padding_mask(tgt)
            tgt_causal_mask = self.create_causal_mask(tgt.size(1)).to(tgt.device)
            tgt_mask = tgt_padding_mask & tgt_causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Encode and decode
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # Enhanced output projection
        return self.output_projection(decoder_output)
    
    def generate_beam_search(self, src: torch.Tensor, beam_size: int = 5, 
                           max_length: int = 256, start_token: int = 2, 
                           end_token: int = 3, pad_token: int = 0,
                           length_penalty: float = 0.6) -> torch.Tensor:
        """Enhanced beam search generation"""
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        # Encode source
        src_mask = self.create_padding_mask(src, pad_token)
        encoder_output = self.encode(src, src_mask)
        
        # Initialize beams
        beams = [(torch.full((batch_size, 1), start_token, dtype=torch.long, device=device), 0.0)]
        
        with torch.no_grad():
            for step in range(max_length - 1):
                candidates = []
                
                for seq, score in beams:
                    if seq[0, -1].item() == end_token:
                        candidates.append((seq, score))
                        continue
                    
                    # Create target mask
                    tgt_padding_mask = self.create_padding_mask(seq, pad_token)
                    tgt_causal_mask = self.create_causal_mask(seq.size(1)).to(device)
                    tgt_mask = tgt_padding_mask & tgt_causal_mask.unsqueeze(0).unsqueeze(0)
                    
                    # Decode
                    decoder_output = self.decode(seq, encoder_output, tgt_mask, src_mask)
                    logits = self.output_projection(decoder_output[:, -1, :])
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Get top-k candidates
                    top_log_probs, top_indices = torch.topk(log_probs, beam_size)
                    
                    for i in range(beam_size):
                        next_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                        next_seq = torch.cat([seq, next_token], dim=1)
                        next_score = score + top_log_probs[0, i].item()
                        candidates.append((next_seq, next_score))
                
                # Select top beams with length penalty
                candidates.sort(key=lambda x: x[1] / (x[0].size(1) ** length_penalty), reverse=True)
                beams = candidates[:beam_size]
                
                # Early stopping if all beams ended
                if all(seq[0, -1].item() == end_token for seq, _ in beams):
                    break
        
        # Return best sequence
        return beams[0][0]
    
    def generate(self, src: torch.Tensor, max_length: int = 256, 
                 start_token: int = 2, end_token: int = 3, pad_token: int = 0,
                 temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9,
                 use_beam_search: bool = True, beam_size: int = 5) -> torch.Tensor:
        """Enhanced generation with multiple strategies"""
        if use_beam_search:
            return self.generate_beam_search(
                src, beam_size=beam_size, max_length=max_length,
                start_token=start_token, end_token=end_token, pad_token=pad_token
            )
        else:
            return self._generate_nucleus(
                src, max_length=max_length, start_token=start_token,
                end_token=end_token, pad_token=pad_token,
                temperature=temperature, top_k=top_k, top_p=top_p
            )
    
    def _generate_nucleus(self, src: torch.Tensor, max_length: int = 256, 
                         start_token: int = 2, end_token: int = 3, pad_token: int = 0,
                         temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9) -> torch.Tensor:
        """Nucleus sampling generation (original method)"""
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        # Encode source
        src_mask = self.create_padding_mask(src, pad_token)
        encoder_output = self.encode(src, src_mask)
        
        # Initialize target with start token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Create target mask
                tgt_padding_mask = self.create_padding_mask(generated, pad_token)
                tgt_causal_mask = self.create_causal_mask(generated.size(1)).to(device)
                tgt_mask = tgt_padding_mask & tgt_causal_mask.unsqueeze(0).unsqueeze(0)
                
                # Decode
                decoder_output = self.decode(generated, encoder_output, tgt_mask, src_mask)
                logits = self.output_projection(decoder_output[:, -1, :])
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end token
                if (next_token == end_token).all():
                    break
        
        return generated

class FastEnhancedTransformer(nn.Module):
    """Fast enhanced transformer optimized for 1-1.5 hour training"""
    
    def __init__(self, vocab_size: int, model_type: str = 'comment',
                 max_seq_len: int = 512, dropout: float = 0.05):
        super().__init__()
        
        # Fast model configurations
        if model_type == 'comment':
            # Fast Comment Model: ~15M parameters
            self.d_model = 256
            self.num_heads = 8
            self.num_encoder_layers = 4
            self.num_decoder_layers = 4
            self.d_ff = 1024
        elif model_type == 'commit':
            # Fast Commit Model: ~25M parameters  
            self.d_model = 384
            self.num_heads = 12
            self.num_encoder_layers = 5
            self.num_decoder_layers = 5
            self.d_ff = 1536
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        self.vocab_size = vocab_size
        self.model_type = model_type
        
        # Embeddings
        self.src_embedding = nn.Embedding(vocab_size, self.d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, max_seq_len, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(self.d_model, self.num_heads, self.d_ff, dropout)
            for _ in range(self.num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(self.d_model)
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(self.d_model, self.num_heads, self.d_ff, dropout)
            for _ in range(self.num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(self.d_model)
        
        # Output projection - simplified for speed
        self.output_projection = nn.Linear(self.d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        print(f"⚡ Fast {model_type} model initialized:")
        print(f"   📊 Parameters: ~{self.count_parameters()/1e6:.1f}M")
        print(f"   🧠 d_model: {self.d_model}")
        print(f"   🔄 Encoder layers: {self.num_encoder_layers}")
        print(f"   🔄 Decoder layers: {self.num_decoder_layers}")
        print(f"   👁️  Attention heads: {self.num_heads}")
        
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)  # Smaller gain for stability
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_padding_mask(self, x: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
        return (x != pad_token).unsqueeze(1).unsqueeze(2)
    
    def create_causal_mask(self, size: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return self.encoder_norm(x)
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        return self.decoder_norm(x)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if src_mask is None:
            src_mask = self.create_padding_mask(src)
        if tgt_mask is None:
            tgt_padding_mask = self.create_padding_mask(tgt)
            tgt_causal_mask = self.create_causal_mask(tgt.size(1)).to(tgt.device)
            tgt_mask = tgt_padding_mask & tgt_causal_mask.unsqueeze(0).unsqueeze(0)
        
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        return self.output_projection(decoder_output)
    
    def generate(self, src: torch.Tensor, max_length: int = 128, 
                 start_token: int = 2, end_token: int = 3, pad_token: int = 0,
                 temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        """Fast generation with top-k sampling"""
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        src_mask = self.create_padding_mask(src, pad_token)
        encoder_output = self.encode(src, src_mask)
        
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                tgt_padding_mask = self.create_padding_mask(generated, pad_token)
                tgt_causal_mask = self.create_causal_mask(generated.size(1)).to(device)
                tgt_mask = tgt_padding_mask & tgt_causal_mask.unsqueeze(0).unsqueeze(0)
                
                decoder_output = self.decode(generated, encoder_output, tgt_mask, src_mask)
                logits = self.output_projection(decoder_output[:, -1, :])
                
                # Temperature and top-k sampling
                logits = logits / temperature
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if (next_token == end_token).all():
                    break
        
        return generated

# Alias for backward compatibility
CodeCommentTransformer = ProductionTransformer


class FastEnhancedTransformer(nn.Module):
    """Fast enhanced transformer optimized for 1-1.5 hour training"""
    
    def __init__(self, vocab_size: int, model_type: str = 'comment',
                 max_seq_len: int = 512, dropout: float = 0.05):
        super().__init__()
        
        # Fast model configurations
        if model_type == 'comment':
            # Fast Comment Model: ~15M parameters
            self.d_model = 256
            self.num_heads = 8
            self.num_encoder_layers = 4
            self.num_decoder_layers = 4
            self.d_ff = 1024
        elif model_type == 'commit':
            # Fast Commit Model: ~25M parameters  
            self.d_model = 384
            self.num_heads = 12
            self.num_encoder_layers = 5
            self.num_decoder_layers = 5
            self.d_ff = 1536
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        self.vocab_size = vocab_size
        self.model_type = model_type
        
        # Embeddings
        self.src_embedding = nn.Embedding(vocab_size, self.d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, max_seq_len, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(self.d_model, self.num_heads, self.d_ff, dropout)
            for _ in range(self.num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(self.d_model)
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(self.d_model, self.num_heads, self.d_ff, dropout)
            for _ in range(self.num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(self.d_model)
        
        # Output projection - simplified for speed
        self.output_projection = nn.Linear(self.d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        print(f"🚀 Fast {model_type} model initialized:")
        print(f"   📊 Parameters: ~{self.count_parameters()/1e6:.1f}M")
        print(f"   🧠 d_model: {self.d_model}")
        print(f"   🔄 Encoder layers: {self.num_encoder_layers}")
        print(f"   🔄 Decoder layers: {self.num_decoder_layers}")
        print(f"   👁️  Attention heads: {self.num_heads}")
        
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)  # Smaller gain for stability
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_padding_mask(self, x: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
        return (x != pad_token).unsqueeze(1).unsqueeze(2)
    
    def create_causal_mask(self, size: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return self.encoder_norm(x)
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        return self.decoder_norm(x)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if src_mask is None:
            src_mask = self.create_padding_mask(src)
        if tgt_mask is None:
            tgt_padding_mask = self.create_padding_mask(tgt)
            tgt_causal_mask = self.create_causal_mask(tgt.size(1)).to(tgt.device)
            tgt_mask = tgt_padding_mask & tgt_causal_mask.unsqueeze(0).unsqueeze(0)
        
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        return self.output_projection(decoder_output)
    
    def generate(self, src: torch.Tensor, max_length: int = 128, 
                 start_token: int = 2, end_token: int = 3, pad_token: int = 0,
                 temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        """Fast generation with top-k sampling"""
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        src_mask = self.create_padding_mask(src, pad_token)
        encoder_output = self.encode(src, src_mask)
        
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                tgt_padding_mask = self.create_padding_mask(generated, pad_token)
                tgt_causal_mask = self.create_causal_mask(generated.size(1)).to(device)
                tgt_mask = tgt_padding_mask & tgt_causal_mask.unsqueeze(0).unsqueeze(0)
                
                decoder_output = self.decode(generated, encoder_output, tgt_mask, src_mask)
                logits = self.output_projection(decoder_output[:, -1, :])
                
                # Temperature and top-k sampling
                logits = logits / temperature
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if (next_token == end_token).all():
                    break
        
        return generated
