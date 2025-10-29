# Source Code Directory

This directory contains reusable Python modules and utility functions for the LLM Quantization project.

## Contents

- **Quantization utilities**: Reusable quantization functions
- **Benchmarking tools**: Performance measurement utilities
- **Data processing**: Dataset loading and preprocessing
- **Visualization**: Plotting and chart generation functions
- **Hardware profiling**: GPU monitoring and profiling utilities

## Module Organization

### `quantization_utils.py`
- Model loading and quantization functions
- BitsAndBytes configuration helpers
- Quantization parameter management

### `benchmark_utils.py`
- Inference speed measurement
- Memory usage tracking
- Throughput calculation
- Result logging and formatting

### `hardware_profiling.py`
- GPU utilization monitoring
- CUDA memory tracking
- Tensor core usage analysis
- Performance profiling utilities

### `visualization.py`
- Result plotting functions
- Comparison charts
- Performance graphs
- Summary visualizations

### `data_utils.py`
- Dataset loading
- Tokenization utilities
- Data preprocessing
- Calibration data management

## Usage

Import modules in notebooks:
```python
import sys
sys.path.append('../src')

from quantization_utils import load_quantized_model
from benchmark_utils import measure_inference_speed
from visualization import plot_comparison
```

## Code Standards

- Follow PEP 8 style guidelines
- Include docstrings for all functions
- Add type hints where appropriate
- Write unit tests for critical functions
- Keep functions modular and reusable

## Team Access

Both team members (Utkarsh & Sami) have read/write access to this directory.

