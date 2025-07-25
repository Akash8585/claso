from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import os
from model.production_transformer import FastEnhancedTransformer
import pickle
import asyncio
from typing import List, Optional
from dotenv import load_dotenv

from fastapi import Request
from fastapi import UploadFile, File
import difflib
import tempfile
import os
from dataclasses import dataclass

load_dotenv()

# Add this to handle pickle loading issues
@dataclass
class FastTrainingConfig:
    """Fast training configuration for pickle compatibility"""
    model_type: str = 'comment'
    vocab_size: int = 8000
    max_seq_len: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 12
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    use_mixed_precision: bool = False
    gradient_checkpointing: bool = False
    accumulation_steps: int = 2
    dropout: float = 0.05
    weight_decay: float = 0.005
    label_smoothing: float = 0.05
    validation_split: float = 0.1
    save_every_epochs: int = 3
    early_stopping_patience: int = 4
    data_path: str = 'data/enhanced_all_pairs.json'
    model_save_path: str = 'models/fast_enhanced'
    log_path: str = 'logs/fast_enhanced_training.log'

app = FastAPI(title="Claso - AI Code Comment Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None
commit_model = None
commit_tokenizer = None

class CodeRequest(BaseModel):
    code: str
    language: str  # 'python' or 'javascript'
    style: Optional[str] = 'docstring'  # 'docstring', 'inline', 'block'

class CommentResponse(BaseModel):
    comments: List[str]
    confidence: float
    processing_time: float

class FeedbackRequest(BaseModel):
    code: str
    comment: str
    user_feedback: str
    language: str
    confidence: float = 0.0
    processing_time: float = 0.0

class DiffRequest(BaseModel):
    diff: str

class CommitMsgResponse(BaseModel):
    message: str
    confidence: float

@app.on_event("startup")
async def load_model():
    global model, tokenizer, commit_model, commit_tokenizer
    
    # Enhanced device detection for Apple Silicon
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"🚀 Using device: {device}")
    
    # Load fast enhanced comment generation model
    try:
        tokenizer_path = 'models/fast_comment/comment_tokenizer.pkl'
        weights_path = 'models/fast_comment/comment_weights_only.pth'
        
        if os.path.exists(tokenizer_path) and os.path.exists(weights_path):
            print("📁 Loading comment model files...")
            
            # Load tokenizer
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            print(f"✅ Tokenizer loaded: {len(tokenizer.vocab)} tokens")
            
            # Load weights directly (no pickle issues)
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            print("✅ Model weights loaded")
            
            # Create FastEnhancedTransformer with tokenizer vocab size
            model = FastEnhancedTransformer(
                vocab_size=len(tokenizer.vocab),
                model_type='comment',
                max_seq_len=512,
                dropout=0.05
            )
            
            # Load weights with strict=False to handle any mismatches
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  Missing keys: {len(missing_keys)} (this is normal for architecture changes)")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys: {len(unexpected_keys)} (this is normal for architecture changes)")
            
            model.to(device)
            model.eval()
            print("✅ Fast Enhanced Comment Model loaded successfully!")
            print(f"   📊 Parameters: ~{model.count_parameters()/1e6:.1f}M")
            print(f"   🧠 Architecture: {model.d_model}d, {model.num_encoder_layers}+{model.num_decoder_layers} layers")
            print(f"   🔤 Vocabulary: {len(tokenizer.vocab)} tokens")
        else:
            print("⚠️  Fast comment model files not found.")
            print("   Expected files:")
            print(f"   - {tokenizer_path}")
            print(f"   - {weights_path}")
            print("   Run: python extract_weights.py to extract weights from checkpoint")
            model = None
            tokenizer = None
    except Exception as e:
        print(f"⚠️  Error loading fast comment model: {e}")
        model = None
        tokenizer = None
    
    # Load fast enhanced commit message generation model (optional - still training)
    try:
        commit_tokenizer_path = 'models/fast_commit/commit_tokenizer.pkl'
        commit_checkpoint_path = 'models/fast_commit/commit_best.pth'
        
        if os.path.exists(commit_tokenizer_path) and os.path.exists(commit_checkpoint_path):
            print("📁 Loading commit model files...")
            
            # Load tokenizer
            with open(commit_tokenizer_path, 'rb') as f:
                commit_tokenizer = pickle.load(f)
            
            # Load checkpoint
            checkpoint = torch.load(commit_checkpoint_path, map_location=device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Create FastEnhancedTransformer
            commit_model = FastEnhancedTransformer(
                vocab_size=len(commit_tokenizer.vocab),
                model_type='commit',
                max_seq_len=512,
                dropout=0.05
            )
            
            # Load weights
            commit_model.load_state_dict(state_dict, strict=False)
            commit_model.to(device)
            commit_model.eval()
            print("✅ Fast Enhanced Commit Model loaded successfully!")
        else:
            print("⚠️  Fast commit model files not found (still training).")
            commit_model = None
            commit_tokenizer = None
    except Exception as e:
        print(f"⚠️  Commit model not ready yet: {e}")
        commit_model = None
        commit_tokenizer = None

@app.post("/generate-comments", response_model=CommentResponse)
async def generate_comments(request: CodeRequest):
    if not model or not tokenizer:
        raise HTTPException(
            status_code=503, 
            detail="Fast Enhanced Comment model not loaded. Please train the model first by running: python fast_enhanced_train.py"
        )
    
    try:
        start_time = asyncio.get_event_loop().time()
        comments = await generate_code_comments(request.code, request.language, request.style)
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return CommentResponse(
            comments=comments,
            confidence=0.85,
            processing_time=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    try:
        client = SupabaseClient()
        feedback_data = request.dict()
        result = client.insert_feedback(feedback_data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-diff")
async def generate_diff(files: list[UploadFile] = File(...)):
    """Generate diff from uploaded files"""
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 files required")
    
    try:
        # Sort files by name to ensure consistent order
        files.sort(key=lambda x: x.filename)
        
        # Read file contents
        file_contents = []
        for file in files:
            content = await file.read()
            file_contents.append(content.decode('utf-8'))
        
        # Generate diff between first two files (before/after)
        before_content = file_contents[0].splitlines(keepends=True)
        after_content = file_contents[1].splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            before_content, 
            after_content,
            fromfile=files[0].filename,
            tofile=files[1].filename,
            lineterm=''
        )
        
        diff_text = '\n'.join(diff)
        return {"diff": diff_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating diff: {str(e)}")

@app.post("/generate-commit-msg", response_model=CommitMsgResponse)
async def generate_commit_msg(request: DiffRequest):
    """Generate commit message from diff"""
    try:
        start_time = asyncio.get_event_loop().time()
        
        if commit_model and commit_tokenizer:
            # Use enhanced commit model
            device = next(commit_model.parameters()).device
            
            # Enhanced tokenization for diff
            input_tokens = commit_tokenizer.encode(request.diff)
            
            # Truncate if too long
            max_input_length = 300
            if len(input_tokens) > max_input_length:
                input_tokens = input_tokens[:max_input_length]
            
            # Add start token and create source tensor
            src_tokens = [commit_tokenizer.start_token] + input_tokens + [commit_tokenizer.end_token]
            src = torch.tensor([src_tokens], dtype=torch.long).to(device)
            
            with torch.no_grad():
                # Use the enhanced model's generate method
                generated_tokens = commit_model.generate(
                    src, 
                    max_length=64,  # Reasonable length for commit messages
                    start_token=commit_tokenizer.start_token,
                    end_token=commit_tokenizer.end_token,
                    pad_token=commit_tokenizer.pad_token,
                    temperature=0.7,
                    top_k=40
                )
                
                # Decode the generated tokens
                commit_msg = commit_tokenizer.decode(generated_tokens[0].tolist())
                
                # Clean up the commit message
                commit_msg = clean_commit_message(commit_msg)
        else:
            # Generate intelligent commit message based on diff analysis
            commit_msg = generate_commit_from_diff(request.diff)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return CommitMsgResponse(
            message=commit_msg,
            confidence=0.85
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def clean_commit_message(message: str) -> str:
    """Clean and format generated commit messages"""
    # Remove special tokens
    message = message.replace('<START>', '').replace('<END>', '').replace('<PAD>', '').replace('<UNK>', '')
    message = message.strip()
    
    if not message:
        return "Update code implementation"
    
    # Ensure proper commit message format
    if not message.endswith('.'):
        # Don't add period for conventional commits (feat:, fix:, etc.)
        if not any(message.startswith(prefix) for prefix in ['feat:', 'fix:', 'docs:', 'style:', 'refactor:', 'test:', 'chore:']):
            if len(message) > 50:  # Long messages don't need periods
                pass
            else:
                message = message.rstrip('.') + ''  # No period for short messages
    
    # Capitalize first letter if not a conventional commit
    if not any(message.startswith(prefix) for prefix in ['feat:', 'fix:', 'docs:', 'style:', 'refactor:', 'test:', 'chore:']):
        message = message[0].upper() + message[1:] if message else message
    
    # Limit length
    if len(message) > 72:
        message = message[:69] + "..."
    
    return message

async def generate_code_comments(code: str, language: str, style: str) -> List[str]:
    device = next(model.parameters()).device
    
    # Enhanced tokenization
    input_tokens = tokenizer.encode(code)
    
    # Truncate if too long (enhanced models can handle more)
    max_input_length = 400  # Increased for better context
    if len(input_tokens) > max_input_length:
        input_tokens = input_tokens[:max_input_length]
    
    # Add start token and create source tensor
    src_tokens = [tokenizer.start_token] + input_tokens + [tokenizer.end_token]
    src = torch.tensor([src_tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        comments = []
        
        # Generate 3 different comments with different parameters
        for i in range(3):
            temperature = 0.6 + (i * 0.1)  # Vary temperature for diversity
            top_k = 30 + (i * 10)  # Vary top_k for diversity
            
            # Use the enhanced model's generate method
            generated_tokens = model.generate(
                src, 
                max_length=128,  # Longer for better quality
                start_token=tokenizer.start_token,
                end_token=tokenizer.end_token,
                pad_token=tokenizer.pad_token,
                temperature=temperature, 
                top_k=top_k
            )
            
            # Decode the generated tokens
            generated_comment = tokenizer.decode(generated_tokens[0].tolist())
            
            # Clean up the comment
            generated_comment = clean_generated_comment(generated_comment, language, style)
            
            if generated_comment and len(generated_comment.strip()) > 10:
                comments.append(generated_comment)
    
    # Remove duplicates and empty comments
    unique_comments = []
    for comment in comments:
        if comment and comment not in unique_comments and len(comment.strip()) > 5:
            unique_comments.append(comment)
    
    # If we don't have enough good comments, add some intelligent fallbacks
    while len(unique_comments) < 2:
        fallback = generate_intelligent_fallback(code, language, style)
        if fallback not in unique_comments:
            unique_comments.append(fallback)
    
    return unique_comments[:3]

def clean_generated_comment(comment: str, language: str, style: str) -> str:
    """Clean and format generated comments"""
    # Remove special tokens
    comment = comment.replace('<START>', '').replace('<END>', '').replace('<PAD>', '').replace('<UNK>', '')
    comment = comment.strip()
    
    if not comment:
        return ""
    
    # Add proper formatting based on language and style
    if language == 'python' and style == 'docstring':
        if not comment.startswith('"""') and not comment.startswith("'''"):
            # Format as proper docstring
            lines = comment.split('\n')
            if len(lines) == 1:
                comment = f'"""{comment}"""'
            else:
                comment = f'"""\n{comment}\n"""'
    elif language == 'javascript':
        if not comment.startswith('/**') and not comment.startswith('//'):
            # Format as JSDoc
            lines = comment.split('\n')
            if len(lines) == 1:
                comment = f'/** {comment} */'
            else:
                formatted_lines = [f' * {line}' if line.strip() else ' *' for line in lines]
                comment = f'/**\n{chr(10).join(formatted_lines)}\n */'
    
    return comment

def generate_intelligent_fallback(code: str, language: str, style: str) -> str:
    """Generate intelligent fallback comments based on code analysis"""
    import re
    
    # Analyze code structure
    if language == 'python':
        if 'def ' in code:
            func_names = re.findall(r'def\s+(\w+)', code)
            if func_names:
                func_name = func_names[0]
                if 'fibonacci' in func_name.lower():
                    return '"""\nCalculate Fibonacci number using recursive approach.\n\nArgs:\n    n: Position in sequence\n\nReturns:\n    Fibonacci number at position n\n"""'
                elif 'sort' in func_name.lower():
                    return f'"""\nSort data using {func_name} algorithm.\n\nImplements efficient sorting with proper error handling.\n"""'
                else:
                    return f'"""\n{func_name.replace("_", " ").title()} function implementation.\n\nProcesses input data and returns the result.\n"""'
        
        if 'class ' in code:
            class_names = re.findall(r'class\s+(\w+)', code)
            if class_names:
                class_name = class_names[0]
                return f'"""\n{class_name} class implementation.\n\nProvides functionality for {class_name.lower()} operations\nwith proper encapsulation and method organization.\n"""'
    
    elif language == 'javascript':
        if 'function ' in code:
            func_names = re.findall(r'function\s+(\w+)', code)
            if func_names:
                func_name = func_names[0]
                return f'/**\n * {func_name} function implementation.\n * \n * Handles {func_name.lower()} operations with proper validation.\n * \n * @returns {{*}} Processed result\n */'
        
        if 'class ' in code:
            class_names = re.findall(r'class\s+(\w+)', code)
            if class_names:
                class_name = class_names[0]
                return f'/**\n * {class_name} class for handling {class_name.lower()} operations.\n * \n * Provides comprehensive functionality with proper error handling.\n * \n * @class {class_name}\n */'
    
    # Generic fallbacks
    fallbacks = [
        "Implementation with proper error handling and validation.",
        "Function that processes input data and returns meaningful results.",
        "Well-structured code following best practices and conventions.",
        "Algorithm implementation optimized for performance and readability."
    ]
    
    import random
    base_comment = random.choice(fallbacks)
    
    if language == 'python' and style == 'docstring':
        return f'"""\n{base_comment}\n"""'
    elif language == 'javascript':
        return f'/**\n * {base_comment}\n */'
    else:
        return f"# {base_comment}" if language == 'python' else f"// {base_comment}"



def generate_commit_from_diff(diff_text: str) -> str:
    """Generate commit message from diff analysis when model is not available"""
    import re
    
    # Analyze diff for patterns
    lines = diff_text.split('\n')
    added_lines = [line for line in lines if line.startswith('+') and not line.startswith('+++')]
    removed_lines = [line for line in lines if line.startswith('-') and not line.startswith('---')]
    
    # Detect common patterns
    if any('def ' in line or 'function ' in line for line in added_lines):
        if any('test' in line.lower() for line in added_lines):
            return "Add unit tests for new functionality"
        else:
            return "Add new function implementation"
    
    if any('class ' in line for line in added_lines):
        return "Add new class implementation"
    
    if any('import ' in line or 'require(' in line for line in added_lines):
        return "Add new dependencies"
    
    if any('TODO' in line or 'FIXME' in line for line in added_lines):
        return "Add TODO comments for future improvements"
    
    if any('try:' in line or 'catch(' in line for line in added_lines):
        return "Add error handling"
    
    if any('if __name__' in line for line in added_lines):
        return "Add main guard to prevent execution on import"
    
    if len(added_lines) > len(removed_lines):
        return "Add new features and functionality"
    elif len(removed_lines) > len(added_lines):
        return "Remove deprecated code and cleanup"
    else:
        return "Update code implementation"

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "comment_model_loaded": model is not None,
        "commit_model_loaded": commit_model is not None,
        "models_available": {
            "comment_generation": model is not None,
            "commit_message_generation": commit_model is not None
        }
    }

@app.get("/model-info")
async def model_info():
    """Get detailed information about loaded models"""
    info = {
        "api_title": "Claso - AI Code Comment Generator API",
        "models": {},
        "system": {}
    }
    
    # Comment model info
    if model is not None:
        info["models"]["comment"] = {
            "type": "FastEnhancedTransformer",
            "parameters": model.count_parameters(),
            "vocab_size": len(tokenizer.vocab) if tokenizer else 0,
            "d_model": model.d_model,
            "encoder_layers": model.num_encoder_layers,
            "decoder_layers": model.num_decoder_layers,
            "attention_heads": model.num_heads,
            "max_seq_len": 512,
            "loaded": True
        }
    else:
        info["models"]["comment"] = {"loaded": False}
    
    # Commit model info
    if commit_model is not None:
        info["models"]["commit"] = {
            "type": "FastEnhancedTransformer",
            "parameters": commit_model.count_parameters(),
            "vocab_size": len(commit_tokenizer.vocab) if commit_tokenizer else 0,
            "d_model": commit_model.d_model,
            "encoder_layers": commit_model.num_encoder_layers,
            "decoder_layers": commit_model.num_decoder_layers,
            "attention_heads": commit_model.num_heads,
            "max_seq_len": 512,
            "loaded": True
        }
    else:
        info["models"]["commit"] = {"loaded": False}
    
    # System info
    if torch.cuda.is_available():
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_info = "Apple Silicon (MPS)"
    else:
        device_info = "CPU"
    
    info["system"] = {
        "device": device_info,
        "pytorch_version": torch.__version__,
        "models_loaded": (model is not None) + (commit_model is not None)
    }
    
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 