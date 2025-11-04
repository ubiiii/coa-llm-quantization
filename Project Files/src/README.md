# Source Code Directory

This directory contains 9 reusable Python modules and utility functions for the LLM Quantization project. All code was written by CipherCore (Utkarsh & Sami) and tested on Google Colab with Tesla T4 GPU.

## Directory Contents

- **benchmark.py** (717 lines): Main benchmarking utilities for measuring speed, memory, accuracy
- **visualization.py** (467 lines): Chart generation for all 13 PNG visualizations
- **colab_test_benchmark.py**: Complete benchmark suite for reproducing results
- **colab_accuracy_test.py**: Perplexity measurement for accuracy analysis
- **colab_visualization.py**: Script to generate all performance charts
- **validate_onnx_models.py** (142 lines): ONNX model validation for Task 3.9
- **colab_onnx_export_simple.py**: ONNX model export for hardware inference
- **accuracy_test_script.py**: Standalone perplexity calculation
- **onnx_gpt2_sampler.py**: ONNX model inference testing
- **test_benchmark.py**: Unit tests for benchmarking utilities

## Module Details

### **1. `benchmark.py` - Core Benchmarking Class**
Main utilities for measuring quantization performance:

```python
from benchmark import LLMBenchmark

benchmark = LLMBenchmark(model, tokenizer, device="cuda")

# Key methods:
speed = benchmark.measure_inference_speed(num_runs=100, warmup_runs=10)
memory = benchmark.measure_memory_usage()
hardware = benchmark.measure_hardware_utilization()
perplexity = benchmark.calculate_perplexity(num_samples=50)
quality = benchmark.measure_output_quality("test prompt")
comprehensive = benchmark.run_comprehensive_benchmark()
```

**What it does:**
- Measures inference speed over 100 runs with 10 warmup iterations
- Tracks peak GPU memory usage during inference
- Calculates perplexity on WikiText-2 dataset
- Monitors GPU utilization and CUDA memory
- Generates quality scores for text output
- Runs complete benchmark suite with all metrics

**Usage:** Imported in notebooks for all performance measurements

### **2. `visualization.py` - Chart Generation**
Creates all performance visualization charts:

```python
from visualization import QuantizationVisualizer

viz = QuantizationVisualizer()

# Key methods:
viz.plot_speed_comparison(save_path="Graphs/speed_comparison.png")
viz.plot_memory_comparison(save_path="Graphs/memory_comparison.png")
viz.plot_perplexity_comparison(save_path="Graphs/perplexity_comparison.png")
viz.create_dashboard(save_path="Graphs/comprehensive_dashboard_4metrics.png")
viz.plot_model_size_vs_performance(save_path="Graphs/model_size_vs_performance.png")
viz.plot_gpu_utilization_comparison(save_path="Graphs/gpu_utilization_comparison.png")
```

**What it does:**
- Creates speed comparison bar charts
- Generates memory reduction visualizations
- Plots GPU utilization metrics
- Builds comprehensive dashboards with 4-8 metrics
- Creates scatter plots for model size vs performance
- Adds annotations and insights to charts

**Output:** All 13 PNG charts in `Project Files/Graphs/`

### **3. `colab_test_benchmark.py` - Complete Benchmark Suite**
Full benchmarking script that reproduces all experimental results:

```bash
cd Project\ Files\src
python colab_test_benchmark.py
```

**What it does:**
- Loads models (DialoGPT-small, TinyLlama-1.1B, distilgpt2)
- Runs FP16 baseline benchmarks
- Tests BitsAndBytes INT8 quantization
- Tests ONNX Runtime INT8 quantization
- Measures all metrics (speed, memory, GPU utilization)
- Saves results to CSV and JSON files

**Runtime:** ~2 hours on Tesla T4 GPU  
**Output:** CSV files in `Project Files/results/`

### **4. `colab_accuracy_test.py` - Accuracy Measurement**
Measures perplexity for all model/precision combinations:

```bash
cd Project\ Files\src
python colab_accuracy_test.py --models distilgpt2,dialogpt-small
```

**What it does:**
- Loads FP16, INT8, INT4 model variants
- Calculates perplexity on WikiText-2 dataset
- Measures accuracy degradation
- Compares quantization impact on quality
- Saves results to CSV

**Runtime:** ~10 minutes per model  
**Output:** `accuracy_results.csv` with perplexity scores

### **5. `colab_visualization.py` - Generate All Charts**
Creates all visualization charts from experimental results:

```bash
cd Project\ Files\src
python colab_visualization.py
```

**What it does:**
- Reads experimental data
- Creates speed comparison charts
- Generates memory analysis plots
- Builds comprehensive dashboard
- Saves all PNG files to Graphs/ folder

**Runtime:** ~30 seconds  
**Output:** All 13 PNG charts in `Project Files/Graphs/`

### **6. `validate_onnx_models.py` - ONNX Validation**
Validates all ONNX model files for Task 3.9:

```bash
cd Project\ Files\src
python validate_onnx_models.py
```

**What it does:**
- Checks ONNX file headers
- Verifies file sizes (460MB for FP32, 229MB for INT8)
- Validates model structure
- Confirms all 4 model variants exist
- Generates validation report

**Output:** Validation status for all ONNX models

### **7. `colab_onnx_export_simple.py` - ONNX Export**
Exports models to ONNX format for hardware inference:

```bash
cd Project\ Files\src
python colab_onnx_export_simple.py
```

**What it does:**
- Exports PyTorch models to ONNX format
- Creates FP32 ONNX models
- Generates INT8 quantized ONNX models
- Adds KV cache support for autoregressive generation
- Creates model documentation

**Output:** 4 ONNX files in `Project Files/Model/`

### **8. `accuracy_test_script.py` - Standalone Perplexity**
Command-line tool for perplexity calculation:

```bash
cd Project\ Files\src
python accuracy_test_script.py --model distilgpt2 --precision INT8
```

**What it does:**
- Loads specified model and precision
- Calculates perplexity on WikiText-2
- Measures accuracy degradation
- Prints results to console

**Usage:** Quick accuracy checks during development

### **9. `onnx_gpt2_sampler.py` - ONNX Inference**
Tests ONNX model inference and validates functionality:

```python
from onnx_gpt2_sampler import ONNXGPT2Sampler

sampler = ONNXGPT2Sampler("model.int8.onnx", "tokenizer")
text = sampler.generate("Hello, how are you?", max_length=50)
print(text)
```

**What it does:**
- Loads ONNX models
- Runs inference on GPU
- Generates text samples
- Validates ONNX functionality

**Usage:** Testing ONNX models after export

## Reproducing Our Results

**Complete workflow:**
```bash
# 1. Run full benchmarks (~2 hours)
python colab_test_benchmark.py

# 2. Measure accuracy (~10 minutes)
python colab_accuracy_test.py --models distilgpt2,dialogpt-small

# 3. Generate visualizations (~30 seconds)
python colab_visualization.py

# 4. Validate ONNX models (~5 seconds)
python validate_onnx_models.py
```

**Expected outputs:**
- CSV files in `../results/` with all measurements
- PNG charts in `../Graphs/` for presentation
- JSON files with formatted results
- All results match paper findings

## Integration with Notebooks

**Import in Google Colab:**
```python
import sys
sys.path.append('/content/src')

from benchmark import LLMBenchmark, quick_benchmark
from visualization import QuantizationVisualizer
```

**Used in main notebook:**
- `notebooks/coa-llm-quantization.ipynb` imports benchmark.py
- All performance measurements use LLMBenchmark class
- Visualizations use QuantizationVisualizer methods
- Results saved to results/ folder

## Code Standards

- **PEP 8 compliant:** All code follows Python style guidelines
- **Docstrings:** Every function has comprehensive documentation
- **Type hints:** Function signatures include type annotations
- **Error handling:** Try-except blocks for robustness
- **Modular design:** Single responsibility per function
- **Tested:** All scripts verified on Tesla T4 GPU

## File Statistics

| File | Lines | Purpose |
|------|-------|---------|
| benchmark.py | 717 | Core benchmarking utilities |
| visualization.py | 467 | Chart generation |
| colab_test_benchmark.py | ~150 | Complete benchmark suite |
| colab_accuracy_test.py | ~150 | Accuracy measurement |
| colab_visualization.py | ~140 | All chart generation |
| validate_onnx_models.py | 142 | ONNX validation |
| colab_onnx_export_simple.py | ~160 | ONNX export |
| accuracy_test_script.py | ~100 | Standalone perplexity |
| onnx_gpt2_sampler.py | ~120 | ONNX inference |

**Total:** ~2,146 lines of production code

## Team

**CipherCore**
- Utkarsh Lubal
- Sami Abedi

**Project:** Hardware/Software Co-Design for LLM Quantization  
**Course:** Computer Organization & Architecture  
**Platform:** Google Colab with Tesla T4 GPU

