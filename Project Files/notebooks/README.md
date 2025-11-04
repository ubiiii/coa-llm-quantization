# Notebooks Directory

This directory contains all Jupyter notebooks and Google Colab experiments for the LLM Quantization project. All experiments were run on Google Colab with Tesla T4 GPU.

## Notebook Files

### **1. `coa-llm-quantization.ipynb` - MAIN EXPERIMENTAL NOTEBOOK**
Primary notebook for all quantization experiments and measurements.

**Contents:**
- Cell 1: Install all dependencies from `../requirements.txt`
- Cells 2-5: FP16 baseline benchmarks (TinyLlama, DialoGPT, distilgpt2)
- Cells 6-8: BitsAndBytes INT8 quantization experiments
- Cells 9-10: ONNX Runtime INT8 quantization tests
- Cells 11+: Hardware profiling and GPU utilization analysis

**Results:**
- TinyLlama-1.1B FP16: 34.53 tokens/s (baseline)
- DialoGPT-small INT8: 5.58 tokens/s (0.52× slowdown)
- Llama-3.2-1B INT4: 157.11 tokens/s (4.55× speedup)
- distilgpt2 INT8 ONNX: 24.4 tokens/s (1.69× speedup)

**Runtime:** ~2 hours to complete all experiments

### **2. `CipherCore_Quantization (1).ipynb`**
Backup/alternative version of main notebook with similar experiments.

### **3. `quantization_experiments.ipynb`**
Additional quantization tests and validation experiments.

## Experimental Results Summary

| Model | Precision | Speed (tokens/s) | Speedup | Memory (GB) | GPU Util (%) |
|-------|-----------|------------------|---------|-------------|--------------|
| TinyLlama-1.1B | FP16 | 34.53 | 1.0× | 2.2 | 52.1 |
| DialoGPT-small | FP16 | 28.42 | 1.0× | 0.54 | 45.2 |
| DialoGPT-small | INT8 | 5.58 | 0.52× | 0.27 | 38.7 |
| Llama-3.2-1B | INT4 | 157.11 | **4.55×** | 0.55 | 78.3 |
| distilgpt2 | FP16 | 91.81 | 1.0× | 0.35 | 15.0 |
| distilgpt2 | INT8 | 59.93 | 0.65× | 0.31 | 14.0 |
| distilgpt2 ONNX | INT8 | 24.4 | **1.69×** | N/A | N/A |

**Key Findings:**
1. INT4 shows best speedup (4.55×) on larger models
2. INT8 slows down small models due to Tesla T4 limitations
3. ONNX Runtime outperforms BitsAndBytes for small models
4. Tesla T4's older tensor cores affect INT8 efficiency
5. Model size threshold: ~1B parameters for beneficial quantization

## Usage Instructions

### **Running in Google Colab:**

1. **Upload notebook to Colab:**
   - Go to https://colab.research.google.com
   - File → Upload notebook → Select `coa-llm-quantization.ipynb`

2. **Enable GPU:**
   - Runtime → Change runtime type → GPU → Tesla T4

3. **Run all cells:**
   - Runtime → Run all
   - First run takes ~10 minutes (package installation)
   - Full experiment takes ~2 hours

4. **View results:**
   - CSV files saved to `../results/`
   - Charts displayed inline in notebook
   - GPU utilization graphs shown

### **Local Setup:**

1. **Install Jupyter:**
```bash
pip install jupyter notebook
```

2. **Run notebook:**
```bash
cd notebooks
jupyter notebook coa-llm-quantization.ipynb
```

**Note:** Requires NVIDIA GPU with CUDA 12.6 support (minimum 8GB VRAM)

## Integration with SRC Folder

Notebooks import utilities from `../src/`:

```python
import sys
sys.path.append('../src')

from benchmark import LLMBenchmark, quick_benchmark
from visualization import QuantizationVisualizer
```

**Used utilities:**
- `benchmark.py`: All performance measurements
- `visualization.py`: Chart generation
- Results saved to `../results/`
- Charts displayed inline

## Notebook Features

### **Self-Contained Experiments:**
- Environment setup and dependency installation
- Model loading and verification
- Performance measurement protocols
- GPU monitoring and profiling

### **Documentation:**
- Clear section headers and comments
- Result summaries after each experiment
- Inline visualizations and charts
- Links to saved data files

### **Result Saving:**
- CSV files: Raw performance data
- JSON files: Formatted experimental results
- PNG charts: Performance visualizations
- Logs: GPU utilization and profiling data

## Hardware Requirements

**Minimum:**
- NVIDIA GPU with 8GB+ VRAM
- CUDA 12.6 compatible
- 16GB+ system RAM

**Recommended (Tested):**
- Tesla T4 GPU (15GB VRAM)
- Google Colab environment
- CUDA 12.6 runtime

**Verified on:**
- Google Colab (Tesla T4)
- All results in paper reproduced
- All charts in presentation generated

## Expected Runtime

| Phase | Time | Description |
|-------|------|-------------|
| Setup | ~10 min | Package installation, downloads |
| FP16 Baselines | ~30 min | TinyLlama, DialoGPT, distilgpt2 |
| INT8 Tests | ~45 min | BitsAndBytes quantization |
| ONNX Tests | ~20 min | ONNX Runtime INT8 |
| Profiling | ~15 min | GPU utilization analysis |
| **Total** | **~2 hours** | Complete experimental run |

## Team Access

Both team members (Utkarsh & Sami) have read/write access to this directory.

