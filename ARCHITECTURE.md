# 🏗️ Claso Architecture Documentation

## 📋 Table of Contents
- [System Overview](#system-overview)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [API Architecture](#api-architecture)
- [Data Pipeline](#data-pipeline)
- [Training Pipeline](#training-pipeline)
- [Frontend Architecture](#frontend-architecture)
- [Deployment Architecture](#deployment-architecture)

## 🎯 System Overview

Claso is a production-ready AI system that generates intelligent code comments and commit messages using custom-trained transformer models.

```mermaid
graph TB
    subgraph "User Interface"
        UI[React Frontend]
        API_DOCS[API Documentation]
    end
    
    subgraph "API Layer"
        FASTAPI[FastAPI Server]
        CORS[CORS Middleware]
        ENDPOINTS[REST Endpoints]
    end
    
    subgraph "ML Models"
        COMMENT_MODEL[Comment Model<br/>~15M Parameters]
        COMMIT_MODEL[Commit Model<br/>~25M Parameters]
        TOKENIZER[Enhanced Tokenizer]
    end
    
    subgraph "Data Layer"
        GITHUB_DATA[GitHub Data<br/>5,000+ Samples]
        PROCESSED_DATA[Processed Training Data]
        MODEL_FILES[Model Checkpoints]
    end
    
    UI --> FASTAPI
    API_DOCS --> FASTAPI
    FASTAPI --> CORS
    FASTAPI --> ENDPOINTS
    ENDPOINTS --> COMMENT_MODEL
    ENDPOINTS --> COMMIT_MODEL
    ENDPOINTS --> TOKENIZER
    
    COMMENT_MODEL --> MODEL_FILES
    COMMIT_MODEL --> MODEL_FILES
    PROCESSED_DATA --> COMMENT_MODEL
    PROCESSED_DATA --> COMMIT_MODEL
    GITHUB_DATA --> PROCESSED_DATA
    
    style COMMENT_MODEL fill:#e1f5fe
    style COMMIT_MODEL fill:#f3e5f5
    style FASTAPI fill:#fff3e0
    style UI fill:#e8f5e8
```

## 📁 Project Structure

```mermaid
graph TD
    subgraph "Claso Project"
        ROOT[claso/]
        
        subgraph "Backend"
            BACKEND[backend/]
            MAIN[main.py - FastAPI Server]
            
            subgraph "Models"
                MODEL_DIR[model/]
                TRANSFORMER[production_transformer.py]
                TOKENIZER_FILE[tokenizer.py]
            end
            
            subgraph "Data Processing"
                DATA_DIR[data/]
                PROCESSOR[data_processor.py]
                DATASETS[*.json - Training Data]
            end
            
            subgraph "Training Scripts"
                FAST_TRAIN[fast_enhanced_train.py]
                SMART_TRAIN[smart_enhanced_train.py]
                SETUP[setup_enhanced_models.py]
            end
            
            subgraph "Data Collection"
                TOOLS_DIR[tools/]
                ADV_COLLECTOR[advanced_data_collector.py]
                COMMIT_COLLECTOR[commit_data_collector.py]
            end
            
            subgraph "Model Storage"
                MODELS_DIR[models/]
                COMMENT_MODELS[fast_comment/]
                COMMIT_MODELS[fast_commit/]
            end
        end
        
        subgraph "Frontend"
            FRONTEND[frontend/]
            
            subgraph "Pages"
                PAGES_DIR[src/pages/]
                ASSISTANT[Assistant.tsx]
                DOCS[Documentation.tsx]
            end
            
            subgraph "Components"
                COMP_DIR[src/components/]
                NAVBAR[Navbar.tsx]
            end
        end
    end
    
    ROOT --> BACKEND
    ROOT --> FRONTEND
    BACKEND --> MAIN
    BACKEND --> MODEL_DIR
    BACKEND --> DATA_DIR
    BACKEND --> TOOLS_DIR
    BACKEND --> MODELS_DIR
    
    MODEL_DIR --> TRANSFORMER
    MODEL_DIR --> TOKENIZER_FILE
    DATA_DIR --> PROCESSOR
    DATA_DIR --> DATASETS
    TOOLS_DIR --> ADV_COLLECTOR
    TOOLS_DIR --> COMMIT_COLLECTOR
    MODELS_DIR --> COMMENT_MODELS
    MODELS_DIR --> COMMIT_MODELS
    
    FRONTEND --> PAGES_DIR
    FRONTEND --> COMP_DIR
    PAGES_DIR --> ASSISTANT
    PAGES_DIR --> DOCS
    COMP_DIR --> NAVBAR
    
    style MAIN fill:#fff3e0
    style TRANSFORMER fill:#e1f5fe
    style ASSISTANT fill:#e8f5e8
    style ADV_COLLECTOR fill:#fce4ec
```

## 🧠 Model Architecture

### FastEnhancedTransformer Architecture

```mermaid
graph TB
    subgraph "Input Processing"
        INPUT[Code Input]
        TOKENIZER[Enhanced Tokenizer<br/>8,000 vocab]
        EMBEDDING[Token Embeddings<br/>d_model dimensions]
        POS_ENC[Positional Encoding<br/>Learnable + Sinusoidal]
    end
    
    subgraph "Encoder Stack"
        ENC1[Encoder Layer 1]
        ENC2[Encoder Layer 2]
        ENC3[Encoder Layer 3]
        ENC4[Encoder Layer 4]
        ENC_NORM[Layer Normalization]
    end
    
    subgraph "Decoder Stack"
        DEC1[Decoder Layer 1]
        DEC2[Decoder Layer 2]
        DEC3[Decoder Layer 3]
        DEC4[Decoder Layer 4]
        DEC_NORM[Layer Normalization]
    end
    
    subgraph "Output Processing"
        OUTPUT_PROJ[Output Projection<br/>Linear Layer]
        SOFTMAX[Softmax]
        GENERATION[Generated Comment]
    end
    
    subgraph "Attention Mechanism"
        SELF_ATTN[Self Attention<br/>8 heads]
        CROSS_ATTN[Cross Attention<br/>8 heads]
        FEED_FORWARD[Feed Forward<br/>GELU Activation]
    end
    
    INPUT --> TOKENIZER
    TOKENIZER --> EMBEDDING
    EMBEDDING --> POS_ENC
    
    POS_ENC --> ENC1
    ENC1 --> ENC2
    ENC2 --> ENC3
    ENC3 --> ENC4
    ENC4 --> ENC_NORM
    
    ENC_NORM --> DEC1
    DEC1 --> DEC2
    DEC2 --> DEC3
    DEC3 --> DEC4
    DEC4 --> DEC_NORM
    
    DEC_NORM --> OUTPUT_PROJ
    OUTPUT_PROJ --> SOFTMAX
    SOFTMAX --> GENERATION
    
    ENC1 -.-> SELF_ATTN
    ENC1 -.-> FEED_FORWARD
    DEC1 -.-> SELF_ATTN
    DEC1 -.-> CROSS_ATTN
    DEC1 -.-> FEED_FORWARD
    
    style EMBEDDING fill:#e1f5fe
    style SELF_ATTN fill:#f3e5f5
    style CROSS_ATTN fill:#f3e5f5
    style FEED_FORWARD fill:#fff3e0
    style OUTPUT_PROJ fill:#e8f5e8
```

### Model Variants Comparison

```mermaid
graph LR
    subgraph "Comment Models"
        LIGHT_COMMENT[Lightweight<br/>4.8M params<br/>d_model=128<br/>layers=2+2]
        FAST_COMMENT[Fast Enhanced<br/>15M params<br/>d_model=256<br/>layers=4+4]
        FULL_COMMENT[Full Enhanced<br/>50M params<br/>d_model=512<br/>layers=8+8]
    end
    
    subgraph "Commit Models"
        LIGHT_COMMIT[Lightweight<br/>11.7M params<br/>d_model=256<br/>layers=4+4]
        FAST_COMMIT[Fast Enhanced<br/>25M params<br/>d_model=384<br/>layers=5+5]
        FULL_COMMIT[Full Enhanced<br/>80M params<br/>d_model=768<br/>layers=10+10]
    end
    
    subgraph "Performance"
        SPEED[Training Speed<br/>30min → 1hr → 2hr]
        QUALITY[Output Quality<br/>Good → Better → Best]
        MEMORY[Memory Usage<br/>50MB → 150MB → 300MB]
    end
    
    LIGHT_COMMENT --> SPEED
    FAST_COMMENT --> SPEED
    FULL_COMMENT --> SPEED
    
    LIGHT_COMMIT --> QUALITY
    FAST_COMMIT --> QUALITY
    FULL_COMMIT --> QUALITY
    
    style FAST_COMMENT fill:#e8f5e8
    style FAST_COMMIT fill:#e8f5e8
    style SPEED fill:#fff3e0
    style QUALITY fill:#f3e5f5
```

## 🌐 API Architecture

### REST API Endpoints

```mermaid
graph TB
    subgraph "Client Requests"
        WEB_CLIENT[Web Frontend]
        API_CLIENT[API Client]
        CURL[cURL/Postman]
    end
    
    subgraph "FastAPI Server"
        FASTAPI_APP[FastAPI Application]
        CORS_MW[CORS Middleware]
        
        subgraph "Endpoints"
            HEALTH[GET /health]
            MODEL_INFO[GET /model-info]
            GEN_COMMENTS[POST /generate-comments]
            GEN_COMMIT[POST /generate-commit-msg]
            GEN_DIFF[POST /generate-diff]
        end
    end
    
    subgraph "Model Layer"
        COMMENT_MODEL[Comment Model<br/>FastEnhancedTransformer]
        COMMIT_MODEL[Commit Model<br/>FastEnhancedTransformer]
        TOKENIZERS[Enhanced Tokenizers]
    end
    
    subgraph "Response Processing"
        VALIDATION[Input Validation]
        INFERENCE[Model Inference]
        POST_PROCESS[Post Processing]
        JSON_RESPONSE[JSON Response]
    end
    
    WEB_CLIENT --> FASTAPI_APP
    API_CLIENT --> FASTAPI_APP
    CURL --> FASTAPI_APP
    
    FASTAPI_APP --> CORS_MW
    CORS_MW --> HEALTH
    CORS_MW --> MODEL_INFO
    CORS_MW --> GEN_COMMENTS
    CORS_MW --> GEN_COMMIT
    CORS_MW --> GEN_DIFF
    
    GEN_COMMENTS --> VALIDATION
    GEN_COMMIT --> VALIDATION
    VALIDATION --> INFERENCE
    INFERENCE --> COMMENT_MODEL
    INFERENCE --> COMMIT_MODEL
    INFERENCE --> TOKENIZERS
    
    COMMENT_MODEL --> POST_PROCESS
    COMMIT_MODEL --> POST_PROCESS
    POST_PROCESS --> JSON_RESPONSE
    
    style FASTAPI_APP fill:#fff3e0
    style COMMENT_MODEL fill:#e1f5fe
    style COMMIT_MODEL fill:#f3e5f5
    style JSON_RESPONSE fill:#e8f5e8
```

### API Request/Response Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Model
    participant Tokenizer
    
    Client->>FastAPI: POST /generate-comments
    Note over Client,FastAPI: {"code": "def func():", "language": "python"}
    
    FastAPI->>FastAPI: Validate Input
    FastAPI->>Tokenizer: Encode Code
    Tokenizer-->>FastAPI: Token IDs
    
    FastAPI->>Model: Generate Comment
    Note over Model: Forward Pass + Generation
    Model-->>FastAPI: Generated Tokens
    
    FastAPI->>Tokenizer: Decode Tokens
    Tokenizer-->>FastAPI: Comment Text
    
    FastAPI->>FastAPI: Post-process & Format
    FastAPI-->>Client: JSON Response
    Note over Client,FastAPI: {"comments": [...], "confidence": 0.85}
```

## 📊 Data Pipeline

### Data Collection & Processing

```mermaid
graph TD
    subgraph "Data Sources"
        GITHUB[GitHub API<br/>5,000+ Repositories]
        MANUAL[Manual Curation]
        EXISTING[Existing Datasets]
    end
    
    subgraph "Collection Tools"
        ADV_COLLECTOR[Advanced Data Collector<br/>Multi-threaded Scraping]
        COMMIT_COLLECTOR[Commit Data Collector<br/>Diff Analysis]
        BASIC_COLLECTOR[Basic Data Collector<br/>Simple Scraping]
    end
    
    subgraph "Raw Data"
        CODE_PAIRS[Code-Comment Pairs<br/>Python & JavaScript]
        COMMIT_PAIRS[Diff-Message Pairs<br/>Git Commits]
        METADATA[Quality Scores<br/>Language Tags]
    end
    
    subgraph "Data Processing"
        QUALITY_FILTER[Quality Filtering<br/>Score > 0.5]
        DEDUPLICATION[Deduplication<br/>Hash-based]
        AUGMENTATION[Data Augmentation<br/>3x Expansion]
        BALANCING[Language Balancing<br/>60% Python, 40% JS]
    end
    
    subgraph "Processed Data"
        TRAIN_DATA[Training Dataset<br/>~5,000 samples]
        VAL_DATA[Validation Dataset<br/>~500 samples]
        TOKENIZED[Tokenized Data<br/>Ready for Training]
    end
    
    GITHUB --> ADV_COLLECTOR
    GITHUB --> COMMIT_COLLECTOR
    MANUAL --> BASIC_COLLECTOR
    EXISTING --> BASIC_COLLECTOR
    
    ADV_COLLECTOR --> CODE_PAIRS
    COMMIT_COLLECTOR --> COMMIT_PAIRS
    BASIC_COLLECTOR --> CODE_PAIRS
    
    CODE_PAIRS --> QUALITY_FILTER
    COMMIT_PAIRS --> QUALITY_FILTER
    METADATA --> QUALITY_FILTER
    
    QUALITY_FILTER --> DEDUPLICATION
    DEDUPLICATION --> AUGMENTATION
    AUGMENTATION --> BALANCING
    
    BALANCING --> TRAIN_DATA
    BALANCING --> VAL_DATA
    TRAIN_DATA --> TOKENIZED
    VAL_DATA --> TOKENIZED
    
    style ADV_COLLECTOR fill:#e1f5fe
    style QUALITY_FILTER fill:#fff3e0
    style TRAIN_DATA fill:#e8f5e8
```

### Data Quality Pipeline

```mermaid
graph LR
    subgraph "Quality Metrics"
        LENGTH[Comment Length<br/>10-100 words]
        STRUCTURE[Code Structure<br/>Functions/Classes]
        KEYWORDS[Quality Keywords<br/>Args, Returns, etc.]
        LANGUAGE[Language Detection<br/>Python/JavaScript]
    end
    
    subgraph "Scoring System"
        BASE_SCORE[Base Score: 0.0]
        LENGTH_SCORE[+0.3 for good length]
        STRUCT_SCORE[+0.2 for structure]
        KEYWORD_SCORE[+0.2 for keywords]
        FINAL_SCORE[Final Score: 0.0-1.0]
    end
    
    subgraph "Filtering"
        THRESHOLD[Threshold: 0.5]
        ACCEPT[Accept Sample]
        REJECT[Reject Sample]
    end
    
    LENGTH --> LENGTH_SCORE
    STRUCTURE --> STRUCT_SCORE
    KEYWORDS --> KEYWORD_SCORE
    LANGUAGE --> BASE_SCORE
    
    BASE_SCORE --> FINAL_SCORE
    LENGTH_SCORE --> FINAL_SCORE
    STRUCT_SCORE --> FINAL_SCORE
    KEYWORD_SCORE --> FINAL_SCORE
    
    FINAL_SCORE --> THRESHOLD
    THRESHOLD --> ACCEPT
    THRESHOLD --> REJECT
    
    style FINAL_SCORE fill:#e8f5e8
    style ACCEPT fill:#c8e6c9
    style REJECT fill:#ffcdd2
```

## 🎓 Training Pipeline

### Training Architecture

```mermaid
graph TB
    subgraph "Training Configuration"
        CONFIG[Training Config<br/>Batch Size, LR, Epochs]
        DEVICE[Device Detection<br/>CUDA/MPS/CPU]
        OPTIMIZER[AdamW Optimizer<br/>Weight Decay]
        SCHEDULER[Warmup Scheduler<br/>Cosine Decay]
    end
    
    subgraph "Data Loading"
        DATASET[Enhanced Dataset<br/>Tokenized Samples]
        DATALOADER[DataLoader<br/>Batching & Shuffling]
        VALIDATION[Validation Split<br/>10% of data]
    end
    
    subgraph "Training Loop"
        FORWARD[Forward Pass<br/>Model Inference]
        LOSS[Loss Calculation<br/>Label Smoothing]
        BACKWARD[Backward Pass<br/>Gradient Computation]
        OPTIMIZER_STEP[Optimizer Step<br/>Parameter Update]
        VALIDATION_STEP[Validation Step<br/>Performance Check]
    end
    
    subgraph "Monitoring"
        LOGGING[Training Logs<br/>Loss, Perplexity]
        CHECKPOINTS[Model Checkpoints<br/>Best Model Saving]
        PLOTS[Training Plots<br/>Loss Curves]
        EARLY_STOP[Early Stopping<br/>Patience-based]
    end
    
    CONFIG --> FORWARD
    DEVICE --> FORWARD
    OPTIMIZER --> OPTIMIZER_STEP
    SCHEDULER --> OPTIMIZER_STEP
    
    DATASET --> DATALOADER
    DATALOADER --> FORWARD
    VALIDATION --> VALIDATION_STEP
    
    FORWARD --> LOSS
    LOSS --> BACKWARD
    BACKWARD --> OPTIMIZER_STEP
    OPTIMIZER_STEP --> VALIDATION_STEP
    
    VALIDATION_STEP --> LOGGING
    LOGGING --> CHECKPOINTS
    CHECKPOINTS --> PLOTS
    PLOTS --> EARLY_STOP
    
    style FORWARD fill:#e1f5fe
    style LOSS fill:#fff3e0
    style CHECKPOINTS fill:#e8f5e8
    style EARLY_STOP fill:#f3e5f5
```

### Training Options Comparison

```mermaid
graph TD
    subgraph "Training Scripts"
        FAST[fast_enhanced_train.py<br/>1-1.5 hours<br/>15M/25M params]
        SMART[smart_enhanced_train.py<br/>Adaptive<br/>Auto-sizing]
        FULL[train_enhanced.py<br/>2-3 hours<br/>50M/80M params]
    end
    
    subgraph "Performance Trade-offs"
        SPEED[Training Speed<br/>Fast → Smart → Full]
        QUALITY[Model Quality<br/>Good → Better → Best]
        MEMORY[Memory Usage<br/>Low → Medium → High]
    end
    
    subgraph "Use Cases"
        DEMO[Quick Demos<br/>Prototyping]
        PRODUCTION[Production Use<br/>Balanced Performance]
        RESEARCH[Research<br/>Maximum Quality]
    end
    
    FAST --> SPEED
    SMART --> SPEED
    FULL --> SPEED
    
    FAST --> QUALITY
    SMART --> QUALITY
    FULL --> QUALITY
    
    FAST --> DEMO
    SMART --> PRODUCTION
    FULL --> RESEARCH
    
    style FAST fill:#c8e6c9
    style SMART fill:#e8f5e8
    style PRODUCTION fill:#fff3e0
```

## 🎨 Frontend Architecture

### React Component Structure

```mermaid
graph TB
    subgraph "App Structure"
        APP[App.tsx<br/>Main Application]
        ROUTER[React Router<br/>Navigation]
        
        subgraph "Pages"
            ASSISTANT[Assistant.tsx<br/>Main Interface]
            DOCS[Documentation.tsx<br/>API Docs]
        end
        
        subgraph "Components"
            NAVBAR[Navbar.tsx<br/>Navigation Bar]
            CODE_EDITOR[Monaco Editor<br/>Code Input]
            RESULT_DISPLAY[Result Display<br/>Generated Comments]
        end
        
        subgraph "Utilities"
            API_CLIENT[API Client<br/>HTTP Requests]
            TYPES[TypeScript Types<br/>Type Definitions]
            UTILS[Utility Functions<br/>Helpers]
        end
    end
    
    subgraph "Styling"
        TAILWIND[Tailwind CSS<br/>Utility Classes]
        FRAMER[Framer Motion<br/>Animations]
        GLASSMORPHISM[Glassmorphism<br/>Modern UI]
    end
    
    APP --> ROUTER
    ROUTER --> ASSISTANT
    ROUTER --> DOCS
    
    ASSISTANT --> NAVBAR
    ASSISTANT --> CODE_EDITOR
    ASSISTANT --> RESULT_DISPLAY
    
    ASSISTANT --> API_CLIENT
    API_CLIENT --> TYPES
    TYPES --> UTILS
    
    ASSISTANT --> TAILWIND
    NAVBAR --> FRAMER
    RESULT_DISPLAY --> GLASSMORPHISM
    
    style APP fill:#e8f5e8
    style ASSISTANT fill:#e1f5fe
    style CODE_EDITOR fill:#fff3e0
    style API_CLIENT fill:#f3e5f5
```

### User Interaction Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Monaco
    participant API
    participant Model
    
    User->>Frontend: Open Application
    Frontend->>Monaco: Initialize Code Editor
    
    User->>Monaco: Type/Paste Code
    Monaco->>Frontend: Code Change Event
    
    User->>Frontend: Click Generate
    Frontend->>Frontend: Validate Input
    
    Frontend->>API: POST /generate-comments
    API->>Model: Process Request
    Model-->>API: Generated Comments
    API-->>Frontend: JSON Response
    
    Frontend->>Frontend: Format Results
    Frontend->>User: Display Comments
    
    User->>Frontend: Select Comment
    Frontend->>Monaco: Insert Comment
    Monaco->>User: Updated Code
```

## 🚀 Deployment Architecture

### Development Environment

```mermaid
graph TB
    subgraph "Local Development"
        DEV_BACKEND[Backend Server<br/>localhost:8000]
        DEV_FRONTEND[Frontend Dev Server<br/>localhost:5173]
        DEV_MODELS[Local Models<br/>models/ directory]
    end
    
    subgraph "Development Tools"
        VITE[Vite Dev Server<br/>Hot Reload]
        UVICORN[Uvicorn Server<br/>Auto Reload]
        PYTHON_ENV[Python Virtual Env<br/>Isolated Dependencies]
        NODE_MODULES[Node Modules<br/>Frontend Dependencies]
    end
    
    subgraph "Development Workflow"
        CODE_CHANGE[Code Changes]
        HOT_RELOAD[Hot Reload]
        API_TEST[API Testing]
        MODEL_TEST[Model Testing]
    end
    
    DEV_FRONTEND --> VITE
    DEV_BACKEND --> UVICORN
    DEV_BACKEND --> PYTHON_ENV
    DEV_FRONTEND --> NODE_MODULES
    
    CODE_CHANGE --> HOT_RELOAD
    HOT_RELOAD --> API_TEST
    API_TEST --> MODEL_TEST
    
    style DEV_BACKEND fill:#fff3e0
    style DEV_FRONTEND fill:#e8f5e8
    style VITE fill:#e1f5fe
```

### Production Deployment

```mermaid
graph TB
    subgraph "Frontend Deployment"
        VERCEL[Vercel<br/>Static Hosting]
        CDN[Global CDN<br/>Fast Delivery]
        DOMAIN_F[claso.vercel.app]
    end
    
    subgraph "Backend Deployment"
        RAILWAY[Railway/Render<br/>Container Hosting]
        API_SERVER[FastAPI Server<br/>Production Mode]
        DOMAIN_B[claso-api.railway.app]
    end
    
    subgraph "Model Storage"
        MODEL_FILES[Model Files<br/>Persistent Storage]
        CHECKPOINTS[Model Checkpoints<br/>Version Control]
    end
    
    subgraph "Monitoring"
        LOGS[Application Logs<br/>Error Tracking]
        METRICS[Performance Metrics<br/>Response Times]
        HEALTH[Health Checks<br/>Uptime Monitoring]
    end
    
    VERCEL --> CDN
    CDN --> DOMAIN_F
    
    RAILWAY --> API_SERVER
    API_SERVER --> DOMAIN_B
    API_SERVER --> MODEL_FILES
    MODEL_FILES --> CHECKPOINTS
    
    API_SERVER --> LOGS
    LOGS --> METRICS
    METRICS --> HEALTH
    
    style VERCEL fill:#e8f5e8
    style RAILWAY fill:#fff3e0
    style MODEL_FILES fill:#e1f5fe
    style HEALTH fill:#c8e6c9
```

### CI/CD Pipeline

```mermaid
graph LR
    subgraph "Source Control"
        GITHUB[GitHub Repository<br/>Akash8585/claso]
        MAIN[Main Branch]
        PR[Pull Requests]
    end
    
    subgraph "CI/CD"
        ACTIONS[GitHub Actions<br/>Automated Workflows]
        BUILD[Build Process<br/>Frontend & Backend]
        TEST[Automated Tests<br/>API & Model Tests]
        DEPLOY[Deployment<br/>Vercel & Railway]
    end
    
    subgraph "Environments"
        STAGING[Staging Environment<br/>Testing]
        PRODUCTION[Production Environment<br/>Live]
    end
    
    GITHUB --> MAIN
    MAIN --> ACTIONS
    PR --> ACTIONS
    
    ACTIONS --> BUILD
    BUILD --> TEST
    TEST --> DEPLOY
    
    DEPLOY --> STAGING
    STAGING --> PRODUCTION
    
    style GITHUB fill:#f3e5f5
    style ACTIONS fill:#fff3e0
    style PRODUCTION fill:#c8e6c9
```

---

## 📚 Additional Resources

- **API Documentation**: [Interactive API Docs](http://localhost:8000/docs)
- **Model Information**: [Model Info Endpoint](http://localhost:8000/model-info)
- **Training Guide**: [TRAINING.md](backend/TRAINING.md)
- **Hackathon Submission**: [HACKATHON_SUBMISSION.md](HACKATHON_SUBMISSION.md)

---

**Built with ❤️ for developers who deserve better documentation tools!** 🚀