#!/usr/bin/env python3
"""
Fast Enhanced Training - Optimized for 1-1.5 hour training
Smaller but powerful models optimized for your dataset size
"""

import os
import sys
import json
import logging
from dataclasses import dataclass

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

@dataclass
class FastTrainingConfig:
    """Fast training configuration optimized for 1-1.5 hours"""
    # Model settings - smaller but effective
    model_type: str = 'comment'
    vocab_size: int = 8000  # Reduced from 10000
    max_seq_len: int = 512  # Reduced from 1024
    
    # Training settings - optimized for speed
    batch_size: int = 16    # Increased for faster training
    learning_rate: float = 2e-4  # Higher LR for faster convergence
    num_epochs: int = 12    # Reduced epochs
    warmup_steps: int = 500 # Reduced warmup
    gradient_clip: float = 1.0
    
    # Speed optimizations
    use_mixed_precision: bool = False  # Disabled for Apple Silicon
    gradient_checkpointing: bool = False  # Disabled for speed
    accumulation_steps: int = 2  # Reduced accumulation
    
    # Regularization - lighter for faster training
    dropout: float = 0.05   # Reduced dropout
    weight_decay: float = 0.005  # Reduced weight decay
    label_smoothing: float = 0.05  # Reduced smoothing
    
    # Validation and saving
    validation_split: float = 0.1
    save_every_epochs: int = 3
    early_stopping_patience: int = 4
    
    # Paths
    data_path: str = 'data/enhanced_all_pairs.json'
    model_save_path: str = 'models/fast_enhanced'
    log_path: str = 'logs/fast_enhanced_training.log'

def create_fast_enhanced_model_config(model_type: str):
    """Create optimized model architecture for fast training"""
    if model_type == 'comment':
        # Fast Comment Model: ~15M parameters
        return {
            'd_model': 256,      # Reduced from 512
            'num_heads': 8,      # Reduced from 16
            'num_encoder_layers': 4,  # Reduced from 8
            'num_decoder_layers': 4,  # Reduced from 8
            'd_ff': 1024,        # Reduced from 2048
            'max_seq_len': 512,
            'dropout': 0.05
        }
    else:  # commit
        # Fast Commit Model: ~25M parameters
        return {
            'd_model': 384,      # Reduced from 768
            'num_heads': 12,     # Reduced from 16
            'num_encoder_layers': 5,  # Reduced from 10
            'num_decoder_layers': 5,  # Reduced from 10
            'd_ff': 1536,        # Reduced from 3072
            'max_seq_len': 512,
            'dropout': 0.05
        }

def update_model_architecture():
    """Update the production transformer for fast training"""
    # Read current model file
    model_file = 'model/production_transformer.py'
    
    # Create fast model variant
    fast_model_code = '''
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
'''
    
    # Append to model file
    with open(model_file, 'a') as f:
        f.write('\n\n' + fast_model_code)
    
    print("✅ Fast model architecture added to production_transformer.py")

def main():
    logger = setup_logging()
    
    logger.info("⚡ Fast Enhanced Training - Optimized for 1-1.5 hours")
    logger.info("=" * 60)
    
    # Check data availability
    data_files = [
        'data/enhanced_all_pairs.json',
        'data/enhanced_commit_pairs.json'
    ]
    
    comment_data_size = 0
    commit_data_size = 0
    
    if os.path.exists('data/enhanced_all_pairs.json'):
        with open('data/enhanced_all_pairs.json', 'r') as f:
            comment_data = json.load(f)
            comment_data_size = len(comment_data)
            logger.info(f"✅ Code comment data: {comment_data_size} samples")
    
    if os.path.exists('data/enhanced_commit_pairs.json'):
        with open('data/enhanced_commit_pairs.json', 'r') as f:
            commit_data = json.load(f)
            commit_data_size = len(commit_data)
            logger.info(f"✅ Commit message data: {commit_data_size} samples")
    
    if comment_data_size == 0 and commit_data_size == 0:
        logger.error("❌ No training data found!")
        return False
    
    # Update model architecture
    logger.info("🔧 Adding fast model architecture...")
    update_model_architecture()
    
    # Import training modules
    try:
        from train_enhanced import EnhancedTrainer
        # Import the new fast model
        sys.path.append('.')
        from model.production_transformer import FastEnhancedTransformer
    except ImportError as e:
        logger.error(f"❌ Failed to import training modules: {e}")
        return False
    
    # Train comment model if data available
    if comment_data_size > 0:
        logger.info(f"\n📝 Training Fast Comment Model ({comment_data_size} samples)...")
        
        comment_config = FastTrainingConfig(
            model_type='comment',
            batch_size=20,  # Larger batch for speed
            learning_rate=2e-4,
            num_epochs=10,  # Reduced epochs
            data_path='data/enhanced_all_pairs.json',
            model_save_path='models/fast_comment',
            log_path='logs/fast_comment_training.log'
        )
        
        # Patch the trainer to use fast model
        original_create_model = EnhancedTrainer.create_model
        def create_fast_model(self, tokenizer):
            model = FastEnhancedTransformer(
                vocab_size=len(tokenizer.vocab),
                model_type=self.config.model_type,
                max_seq_len=self.config.max_seq_len,
                dropout=self.config.dropout
            )
            model = model.to(self.device)
            return model
        
        EnhancedTrainer.create_model = create_fast_model
        
        try:
            comment_trainer = EnhancedTrainer(comment_config)
            comment_trainer.train()
            logger.info("✅ Fast comment model training completed!")
        except Exception as e:
            logger.error(f"❌ Comment model training failed: {e}")
            return False
    
    # Train commit model if data available
    if commit_data_size > 0:
        logger.info(f"\n💬 Training Fast Commit Model ({commit_data_size} samples)...")
        
        commit_config = FastTrainingConfig(
            model_type='commit',
            batch_size=16,  # Smaller batch for larger model
            learning_rate=1.5e-4,
            num_epochs=8,   # Even fewer epochs for commit
            data_path='data/enhanced_commit_pairs.json',
            model_save_path='models/fast_commit',
            log_path='logs/fast_commit_training.log'
        )
        
        try:
            commit_trainer = EnhancedTrainer(commit_config)
            commit_trainer.train()
            logger.info("✅ Fast commit model training completed!")
        except Exception as e:
            logger.error(f"❌ Commit model training failed: {e}")
            return False
    
    # Training summary
    logger.info("\n🎉 Fast Enhanced Training Completed!")
    logger.info("📊 Model Summary:")
    if comment_data_size > 0:
        logger.info(f"   📝 Comment Model: ~15M params, trained on {comment_data_size} samples")
    if commit_data_size > 0:
        logger.info(f"   💬 Commit Model: ~25M params, trained on {commit_data_size} samples")
    
    logger.info("📁 Models saved to:")
    logger.info("   models/fast_comment/ - Comment generation model")
    logger.info("   models/fast_commit/ - Commit message model")
    
    logger.info("📊 Check logs/ for training details")
    logger.info("⚡ Total training time: ~1-1.5 hours")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)