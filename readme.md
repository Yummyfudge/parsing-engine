


# Parsing Engine

A modular document layout and OCR pipeline designed for flexible, pluggable layout detection and downstream processing.

## 🔍 Overview

This engine supports detecting layout blocks in scanned or digital documents using multiple models (Detectron2, EfficientDet, Paddle), and processes them with configurable logic modules.

## ✅ Features

- Plug-and-play model architecture (Detectron2, EfficientDet, Paddle)
- Customizable document type profiles with filter and constructor hooks
- Dynamic runtime config via `config/runtime_config.json`
- Clean separation between model, logic, and data
- NFS-mounted support for large assets and workspace parity

## 🧱 Project Structure

```
parsing_engine/
├── test_layout.py                     # Entry-point to run layout detection
├── layout_models.py                  # Supported layout backends (Detectron2, etc.)
├── layout_runner.py                  # Optional orchestrator for pipelines
├── profiles/
│   └── document_types/
│       ├── test_doc_one/
│       │   └── tables/
│       │       ├── layout_hooks.py
│       │       └── layout_profiles.py
│       └── test_doc_two/
│           └── default/
│               ├── layout_hooks.py
│               └── layout_profiles.py
├── config/
│   └── runtime_config.json           # Declares model, document_type, profile
├── data/ → symlink to NAS
│   ├── input_docs/
│   └── output_results/
├── code_local/ → symlink to NAS
│   └── models/
└── pyproject.toml                    # Package config
```

## 🧪 Running the Engine

Edit `config/runtime_config.json`:

```json
{
  "model": "detectron2",
  "document_type": "test_doc_two",
  "profile": "default"
}
```

Then run:

```bash
python test_layout.py
```

This will:
- Load the specified model
- Apply the specified document type and profile logic
- Output annotated layout JSON and images into `data/output_results/`

## 🔧 Notes

- Large files like model weights are stored in `code_local/` and mounted from NAS
- Profiles can define multiple strategies per document type
- Each profile can inject logic to control layout filtering, block construction, or postprocessing

- This flexibility ensures the engine adapts easily to new document layouts.
