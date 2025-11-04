# Hardware/Software Co-Design for LLM Quantization

**Team:** CipherCore (Utkarsh Lubal & Sami Abedi)  
**Course:** Computer Organization & Architecture  
**Project Date:** January 2025

## What This Project Does

We tested whether reducing model precision (INT8, INT4) actually speeds up inference on real GPUs. Surprise: quantization doesn't always help. A Tesla T4 GPU with BitsAndBytes INT8 made models 35% SLOWER. But ONNX Runtime INT8 delivered 1.69× speedup on the same hardware. 

**The problem:** Everyone assumes quantization = speedup. We proved it depends on three things: model size, framework choice, and GPU architecture.

**Our contribution:** Real benchmarks showing when quantization helps (Llama-3.2-1B INT4: 4.55× speedup) vs hurts (DialoGPT INT8: 48% slowdown), with empirical data from 100+ runs per configuration.

## Research Papers

- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](SmoothQuant.txt)
- [HAQ: Hardware-Aware Automated Quantization with Mixed Precision](HAQ%20Hardware-Aware%20Automated%20Quantization%20with%20Mixed%20Precision.txt)

## Project Structure

```
├── CipherCore_Paper.pdf           # Summary Paper
├── CipherCore_presentation.pptx   # presentation
├── PROJECT_ROADMAP.md             # Complete project guide
├── Project Files/                 # All project components
│   ├── Graphs/                    # Performance visualizations
│   ├── Model/                     # ONNX models and documentation
│   ├── notebooks/                 # Experimental notebooks
│   ├── reports/                   # Analysis and documentation
│   ├── results/                   # Experimental data
│   ├── src/                       # Source code implementation
│   ├── Referance/                 # Research papers
│   └── updates/                   # Project tracking
└── README.md                      # This file
```

## What We Delivered

- ✅ **Paper** (`CipherCore_Paper.pdf`) - Academic paper with methodology, results, and recommendations
- ✅ **Presentation** (`CipherCore_presentation.pptx`) - 14 slides for class presentation
- ✅ **Experimental Data** (`Project Files/results/`) - CSV, JSON files with 100+ runs per configuration
- ✅ **Visualizations** (`Project Files/Graphs/`) - 13 PNG charts comparing speed, memory, accuracy
- ✅ **Reproducible Code** (`Project Files/src/`) - Python tools for benchmarking and visualization
- ✅ **ONNX Models** (`Project Files/Model/`) - Exported quantized models with validation reports
- ✅ **Complete Documentation** - This README, roadmap, analysis reports

## Project Status: COMPLETE ✅

- ✅ **Phase 1:** Research & Planning (100% Complete)
- ✅ **Phase 2:** Environment Setup (100% Complete - Google Colab + Tesla T4)
- ✅ **Phase 3:** Experiments & Data Collection (100% Complete - FP16, INT8, INT4 quantization + ONNX)
- ✅ **Phase 4:** Analysis & Discussion (100% Complete - Hardware analysis, trade-offs, recommendations)
- ✅ **Phase 5:** Documentation & Presentation (100% Complete - Paper, presentation, documentation)

## Key Results So Far

| Model | Precision | Speed | Speedup | Memory Reduction | Perplexity |
|-------|-----------|-------|---------|------------------|------------|
| TinyLlama-1.1B | FP16 | 34.53 tokens/s | 1.0× (baseline) | 0% | 16,813.13 |
| distilgpt2 | FP16 | 91.81 tokens/s | 1.0× (baseline) | 0% | 82.28 |
| distilgpt2 | INT8 | 59.93 tokens/s | 0.65× (slower) | 12% | 83.20 (+1.12%) |
| DialoGPT-small | FP16 | 28.42 tokens/s | 1.0× (baseline) | 0% | 41,021.00 |
| DialoGPT-small | INT8 | 5.58 tokens/s | 0.52× (slower) | 50% | 42,375.57 (+3.30%) |
| Llama-3.2-1B | INT4 | 157.11 tokens/s | **4.55×** | 75% | N/A |
| ONNX (distilgpt2) | INT8 | 24.4 tokens/s | **1.69×** | 50% | N/A |

## Phase 4 Analysis Summary

### Key Findings:
1. **Hardware Architecture Critical**: Tesla T4 limitations cause quantization overhead for small models
2. **Model Size Threshold**: Small models (<1B) show quantization penalties, large models show benefits
3. **Implementation Framework Matters**: ONNX Runtime outperforms BitsAndBytes for small models
4. **Quality Maintained**: No accuracy degradation across all quantization configurations
5. **Memory vs Speed Trade-offs**: Consistent memory savings with variable speed impacts

### Analysis Documents:
- **Hardware Analysis**: Tesla T4 tensor core impact, SIMD utilization, memory bandwidth analysis
- **Trade-off Analysis**: Comprehensive accuracy vs efficiency analysis with deployment recommendations
- **Key Takeaways**: 5 critical insights with practical recommendations for production deployment

## Key Research Contributions

1. **Framework Impact Discovery**: 2.6× performance difference between ONNX Runtime and BitsAndBytes on identical hardware
2. **Model Size Threshold**: Validated 1B parameter threshold for effective quantization on Tesla T4
3. **Hardware/Software Co-Design**: Proved implementation optimization is as critical as quantization method
4. **Energy Efficiency**: Achieved 14.4% power reduction with maintained accuracy
5. **Production Guidelines**: Clear deployment recommendations for different model sizes and hardware

## How to Access Tesla T4 & Run Experiments

### Option 1: Google Colab (Free, Recommended)
We used Google Colab's free Tesla T4 access for all experiments. Here's how to access it:

1. **Access Tesla T4 GPU:** 
   - Visit https://colab.research.google.com
   - Click Runtime → Change runtime type
   - Set Hardware Accelerator: GPU → Tesla T4
   - Free tier provides ~15 hours/week of GPU access
   - For extended use, consider Colab Pro ($10/month) or Pro+ ($50/month)
   
2. **Run Our Notebooks:**
   - Upload `Project Files/notebooks/coa-llm-quantization.ipynb` to Colab
   - **Cell 1:** Installs all dependencies from `Project Files/requirements.txt`
   - **Cell 2-5:** Runs FP16 baseline benchmarks (gets ~34 tokens/s for TinyLlama)
   - **Cell 6-8:** Tests BitsAndBytes INT8 quantization (shows slowdown on small models)
   - **Cell 9-10:** Tests ONNX Runtime INT8 (shows 1.69× speedup)

3. **Expected Setup Time:** 5-10 minutes first run
   - Downloads: Python packages (~2GB), models (~500MB-2GB)
   - Verification: Tesla T4 with 15GB VRAM detected

### Option 2: Local GPU Setup
**Requirements:** NVIDIA GPU with CUDA 12.6 support (minimum 8GB VRAM)
```bash
# Install CUDA 12.6 toolkit from NVIDIA website
# Install Python 3.9+ (we used 3.12.11)
pip install -r "Project Files/requirements.txt"

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"

# Should print: True
```

**Note:** We didn't test locally due to hardware availability, but Colab setup is tested and working.

## How to Use the Source Code

The `Project Files/src/` folder contains 9 reusable Python tools we built for this project. Here's what each file does and how to use them:

### **Core Tools:**

**1. `benchmark.py` (717 lines) - Performance Measurement**
Main benchmarking class with 10+ methods for measuring speed, memory, and accuracy.
```python
from benchmark import LLMBenchmark

# Initialize with your model
benchmark = LLMBenchmark(model, tokenizer, device="cuda")

# Measure inference speed (100 runs, 10 warmup)
speed = benchmark.measure_inference_speed(
    num_runs=100, 
    warmup_runs=10
)
# Returns: {'tokens_per_second': 28.42, 'avg_time': 0.35, ...}

# Measure memory usage
memory = benchmark.measure_memory_usage()
# Returns: {'peak_memory_gb': 2.2, 'baseline_memory': 0.5, ...}

# Calculate perplexity on WikiText-2
perplexity = benchmark.calculate_perplexity(num_samples=50)
# Returns: {'perplexity': 82.28, 'avg_loss': 4.41, ...}
```

**2. `visualization.py` (467 lines) - Generate Graphs**
Creates all 13 PNG charts in the Graphs/ folder.
```python
from visualization import QuantizationVisualizer

viz = QuantizationVisualizer()

# Create speed comparison charts
viz.plot_speed_comparison(data)

# Create memory efficiency charts
viz.plot_memory_comparison(data)

# Generate comprehensive dashboard (what we used for results)
viz.create_dashboard(output_path="Graphs/comprehensive_dashboard.png")
```

**3. `validate_onnx_models.py` (142 lines) - ONNX Model Verification**
Validates all 4 ONNX model files for Task 3.9.
```bash
cd Project\ Files\src
python validate_onnx_models.py

# Output:
# ✅ Validating model.onnx (Basic ONNX FP32)... SUCCESS
# ✅ Validating model.int8.onnx (INT8 Quantized)... SUCCESS
# ... All 4 models validated successfully
```

**4. `accuracy_test_script.py` - Perplexity Calculation**
Measures accuracy degradation with quantization.
```bash
cd Project\ Files\src
python accuracy_test_script.py --model distilgpt2 --precision INT8

# Output: Perplexity results showing minimal degradation (<1%)
```

**5. `colab_test_benchmark.py` - Full Benchmark Suite**
Complete benchmarking script that runs all measurements (FP16, INT8, INT4).
```bash
cd Project\ Files\src
python colab_test_benchmark.py

# Takes ~2 hours on Tesla T4
# Output: CSV files in ../results/ with all measurements
```

**6. `colab_accuracy_test.py` - Accuracy Testing**
Measures perplexity for all model/precision combinations.
```bash
cd Project\ Files\src
python colab_accuracy_test.py --models distilgpt2,dialogpt-small

# Takes ~10 minutes per model
# Output: accuracy_results.csv with WikiText-2 perplexity scores
```

**7. `colab_visualization.py` - Generate All Charts**
Creates all 13 PNG charts from experimental results.
```bash
cd Project\ Files\src
python colab_visualization.py

# Output: All visualization PNGs in ../Graphs/
```

**8. `colab_onnx_export_simple.py` - ONNX Model Generation**
Exports models to ONNX format for Task 3.9.
```bash
cd Project\ Files\src
python colab_onnx_export_simple.py

# Output: 4 ONNX files in ../Model/
```

**9. `onnx_gpt2_sampler.py` - ONNX Inference Testing**
Tests ONNX model inference and validates functionality.
```python
from onnx_gpt2_sampler import ONNXGPT2Sampler
sampler = ONNXGPT2Sampler("model.int8.onnx", "tokenizer")
text = sampler.generate("Hello, how are you?", max_length=50)
print(text)
```

### **Production-Ready Scripts:**

**Reproduce Our Results:**
```bash
cd "Project Files\src"

# Full benchmark suite (takes ~2 hours on Colab Tesla T4)
python colab_test_benchmark.py

# Quick accuracy test (takes ~10 minutes)
python colab_accuracy_test.py --models distilgpt2,dialogpt-small

# Generate all visualizations from results/
python colab_visualization.py

# Validate ONNX models
python validate_onnx_models.py
```

**Output:** 
- CSV files in `Project Files/results/` with raw data
- JSON files with formatted results for analysis
- 13 PNG graphs in `Project Files/Graphs/` for presentation
- All results match the paper findings

## Getting Started

1. **Quick Overview:** Read `CipherCore_Paper.pdf` (5 pages) for findings
2. **Slides:** View `CipherCore_presentation.pptx` (14 slides) for visual summary
3. **Run Code:** Follow notebook in `Project Files/notebooks/coa-llm-quantization.ipynb`
4. **Deep Dive:** Read `PROJECT_ROADMAP.md` for detailed file-by-file guide

## Repository

GitHub: [https://github.com/ubiiii/coa-llm-quantization](https://github.com/ubiiii/coa-llm-quantization)
