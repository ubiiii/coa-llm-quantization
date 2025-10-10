# Experiment Log: LLM Quantization Performance Analysis

**Project:** Hardware/Software Co-Design for LLM Quantization  
**Team:** CipherCore (Utkarsh & Sami)  
**Hardware:** Google Colab Tesla T4 (15.83 GB VRAM, CUDA 12.6)  
**Software:** PyTorch 2.8.0+cu126, Transformers 4.57.0, BitsAndBytes 0.48.1

## Experiment Timeline

### **2025-10-10 - Phase 2 Completion & Baseline Establishment**

#### **Experiment 1: Benchmark Utilities Verification**
- **Date:** 2025-10-10
- **Time:** 01:08:31
- **Purpose:** Verify benchmarking utilities functionality
- **Model:** microsoft/DialoGPT-small (124.4M parameters)
- **Configuration:** FP16 baseline, 10 runs + 3 warmup
- **Results:**
  - Speed: 25.73 tokens/sec (±0.029s)
  - Memory: 0.54 GB peak
  - Quality: 3/5
- **Summary:** Initial test of LLMBenchmark class to ensure all functions work correctly
- **Files:** `benchmark_verification_results.json`

#### **Experiment 2: Comprehensive Baseline Benchmark**
- **Date:** 2025-10-10
- **Time:** 01:15:00
- **Purpose:** Establish reliable baseline for quantization comparison
- **Model:** microsoft/DialoGPT-small (124.4M parameters)
- **Configuration:** FP16 baseline, 100 runs + 10 warmup (professional standard)
- **Results:**
  - Speed: 28.42 tokens/sec (±0.055s)
  - Memory: 0.54 GB peak
  - Quality: 3/5
  - Hardware: Tesla T4, CUDA 12.6
- **Summary:** Comprehensive baseline measurement using standardized methodology. This serves as the reference point for all subsequent quantization experiments.
- **Statistical Reliability:** ±0.055s standard deviation over 100 runs
- **Files:** `baseline_benchmark_results.csv`

#### **Experiment 3: INT8 Quantization Test (Previous)**
- **Date:** 2025-10-09
- **Time:** 17:35:00
- **Purpose:** Test INT8 quantization with BitsAndBytes
- **Model:** microsoft/DialoGPT-small (124.4M parameters)
- **Configuration:** INT8 quantization, BitsAndBytesConfig
- **Results:**
  - Speed: 5.58 tokens/sec (vs 10.75 baseline)
  - Speedup: 0.52× (actually slower due to quantization overhead)
  - Memory: ~50% reduction
- **Summary:** Confirmed that small models (<1B parameters) show quantization overhead > benefits on Tesla T4
- **Key Insight:** Validates literature findings about model size thresholds for quantization benefits

#### **Experiment 4: Sami's 4-bit Quantization (External)**
- **Date:** 2025-10-09
- **Time:** 18:15:00
- **Purpose:** Test INT4 quantization with larger model
- **Model:** Llama-3.2-1B (1B parameters)
- **Configuration:** INT4 quantization, GGUF format
- **Results:**
  - Speed: 157.11 tokens/sec (vs 34.53 baseline)
  - Speedup: 4.55× (significant improvement)
  - Memory: ~75% reduction
- **Summary:** Demonstrates that larger models (>1B parameters) benefit significantly from quantization
- **Key Insight:** Model size is critical factor in quantization effectiveness

## Key Findings Summary

### **Hardware/Software Co-Design Insights:**
1. **Model Size Threshold:** Small models (<1B) show quantization overhead, large models (>1B) show benefits
2. **Tesla T4 Limitations:** Limited INT8 acceleration compared to newer GPUs
3. **Memory vs Speed Trade-off:** Quantization reduces memory but may not improve speed on older hardware
4. **Literature Validation:** Results align with SmoothQuant and HAQ paper findings

### **Performance Comparison:**
| Model | Parameters | Precision | Speed (tokens/sec) | Memory (GB) | Speedup | Notes |
|-------|------------|-----------|-------------------|-------------|---------|-------|
| DialoGPT-small | 124M | FP16 | 28.42 | 0.54 | 1.0× | Baseline reference |
| DialoGPT-small | 124M | INT8 | 5.58 | ~0.27 | 0.52× | Quantization overhead |
| Llama-3.2-1B | 1B | FP16 | 34.53 | ~2.2 | 1.0× | Sami's baseline |
| Llama-3.2-1B | 1B | INT4 | 157.11 | ~0.55 | 4.55× | Significant speedup |

### **2025-10-10 - Phase 3 Visualization & Analysis**

#### **Experiment 5: Comprehensive Performance Visualization**
- **Date:** 2025-10-10
- **Time:** 02:00:00
- **Purpose:** Create comprehensive visualization dashboard for quantization performance analysis
- **Configuration:** Multi-chart dashboard with 6 different performance metrics
- **Results:**
  - **Comprehensive Dashboard:** 4-chart analysis showing speed, memory, speedup, and GPU utilization
  - **Additional Analysis:** Memory reduction percentage and model size vs performance scatter plot
  - **Key Insights:** INT4 shows 4.55× speedup, INT8 shows 0.52× slowdown, 50-75% memory reduction
- **Summary:** Professional visualization suite created showing clear performance trade-offs and hardware/software co-design insights
- **Files:** `visualization.py`, `colab_visualization.py`, comprehensive dashboard charts

## Next Experiments Planned

### **Phase 4: Analysis & Documentation**
1. **Code Review:** Review all experimental notebooks and documentation
2. **Final Analysis:** Comprehensive hardware/software co-design analysis
3. **Report Writing:** Final project report and presentation
4. **Documentation:** Complete project documentation and GitHub updates

## File Organization

### **Results Files:**
- `baseline_benchmark_results.csv` - Formatted results for analysis
- `benchmark_verification_results.json` - Detailed verification results
- `experiment_log.md` - This comprehensive log (current file)

### **Documentation Files:**
- `reports/metrics_definition.md` - Evaluation methodology
- `reports/setup_summary.md` - Environment setup documentation
- `reports/experimental_results.md` - Previous experiment analysis

### **Code Files:**
- `src/benchmark.py` - Benchmarking utilities
- `src/test_benchmark.py` - Local test suite
- `src/colab_test_benchmark.py` - Colab test suite

## Notes for Future Experiments

1. **Always include:** Date, timestamp, purpose, summary, and context
2. **Statistical rigor:** Use 100 runs for final measurements
3. **Hardware consistency:** Document GPU state and configuration
4. **Comparison basis:** Always compare against established baseline
5. **Documentation:** Update this log after each significant experiment

---

**This log will be updated after each experiment to maintain comprehensive project documentation and enable reproducible research.**
