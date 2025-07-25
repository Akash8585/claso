# Claso Architecture Diagrams

This directory contains visual diagrams for the Claso project architecture.

## Diagram Files

### system-overview.png
Complete system architecture showing:
- React frontend with glassmorphism UI
- FastAPI backend with REST endpoints
- FastEnhancedTransformer models (15M/25M parameters)
- Data pipeline from GitHub to training
- Infrastructure components (CUDA/MPS/CPU support)

### model-architecture.png
FastEnhancedTransformer architecture details:
- Encoder-decoder structure
- Multi-head attention mechanisms
- Positional encoding
- Feed-forward networks
- Model variants comparison

### api-flow.png
REST API structure and flow:
- Endpoint definitions
- Request/response formats
- Authentication and CORS
- Error handling
- Model inference pipeline

### training-pipeline.png
ML training process visualization:
- Data collection from GitHub
- Quality filtering and augmentation
- Model training configuration
- Monitoring and checkpointing
- Performance evaluation

## Usage

These diagrams are referenced in:
- README.md (main project documentation)
- ARCHITECTURE.md (detailed technical documentation)
- Frontend documentation page

## Generating Diagrams

To regenerate these diagrams from the Mermaid source code in ARCHITECTURE.md:

1. Use Mermaid CLI: `mmdc -i ARCHITECTURE.md -o diagrams/`
2. Use online Mermaid editor: https://mermaid.live/
3. Use VS Code Mermaid extension for preview and export

## File Formats

- PNG: For README display and documentation
- SVG: For scalable vector graphics (optional)
- Mermaid source: Available in ARCHITECTURE.md