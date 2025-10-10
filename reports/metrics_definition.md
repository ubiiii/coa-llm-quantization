# Evaluation Metrics Definition

**Team:** CipherCore (Utkarsh & Sami)  
**Date:** October 2025  
**Project:** Hardware/Software Co-Design for LLM Quantization

## Overview

This document defines the standardized evaluation metrics and measurement methodologies for our LLM quantization experiments. These metrics ensure consistent, reproducible, and comparable results across all experiments.

## Table of Contents
1. [Performance Metrics](#performance-metrics)
2. [Memory Metrics](#memory-metrics)
3. [Accuracy Metrics](#accuracy-metrics)
4. [Hardware Metrics](#hardware-metrics)
5. [Measurement Methodology](#measurement-methodology)
6. [Data Collection Template](#data-collection-template)

## Performance Metrics

### **1. Inference Speed (Tokens/Second)**
- **Definition**: Number of tokens generated per second during inference
- **Measurement**: Average over multiple runs with identical inputs
- **Formula**: `tokens_generated / inference_time_seconds`
- **Units**: tokens/second
- **Precision**: 2 decimal places
- **Critical for**: Comparing quantization efficiency

### **2. Latency (Seconds)**
- **Definition**: Total time to generate a fixed number of tokens
- **Measurement**: Average over 100 runs with identical configuration
- **Formula**: `total_inference_time / number_of_runs`
- **Units**: seconds
- **Precision**: 3 decimal places
- **Critical for**: Real-time application requirements

### **3. Throughput (Tokens/Second)**
- **Definition**: Overall processing rate including model loading and inference
- **Measurement**: End-to-end time from input to final output
- **Formula**: `total_tokens / (loading_time + inference_time)`
- **Units**: tokens/second
- **Precision**: 2 decimal places
- **Critical for**: Production deployment analysis

### **4. Speedup Factor**
- **Definition**: Relative performance improvement over baseline
- **Formula**: `quantized_speed / baseline_speed`
- **Units**: multiplier (dimensionless)
- **Precision**: 2 decimal places
- **Critical for**: Quantization effectiveness analysis

## Memory Metrics

### **1. Peak Memory Usage (GB)**
- **Definition**: Maximum GPU memory consumed during inference
- **Measurement**: Peak VRAM usage recorded via `nvidia-smi` or PyTorch memory tracking
- **Formula**: `torch.cuda.max_memory_allocated() / 1e9`
- **Units**: Gigabytes (GB)
- **Precision**: 2 decimal places
- **Critical for**: Memory-constrained deployments

### **2. Memory Reduction Percentage**
- **Definition**: Relative memory savings compared to baseline
- **Formula**: `((baseline_memory - quantized_memory) / baseline_memory) * 100`
- **Units**: Percentage (%)
- **Precision**: 1 decimal place
- **Critical for**: Memory optimization analysis

### **3. Model Size (MB)**
- **Definition**: On-disk size of the model file
- **Measurement**: File size in megabytes
- **Units**: Megabytes (MB)
- **Precision**: 1 decimal place
- **Critical for**: Storage and deployment considerations

## Accuracy Metrics

### **1. Output Quality Assessment**
- **Definition**: Qualitative evaluation of generated text quality
- **Methods**:
  - **Identical Output**: Exact match with baseline
  - **Semantic Similarity**: Meaning preservation
  - **Coherence**: Logical flow and readability
  - **Relevance**: Appropriateness to input prompt
- **Scoring**: Binary (Pass/Fail) or scale (1-5)
- **Critical for**: Quantization impact on model quality

### **2. Perplexity (Optional)**
- **Definition**: Measure of model's uncertainty in predictions
- **Formula**: `exp(-1/N * Σ log P(wi))`
- **Units**: Dimensionless
- **Precision**: 3 decimal places
- **Critical for**: Quantitative accuracy assessment

### **3. Error Rate**
- **Definition**: Percentage of runs producing invalid outputs
- **Formula**: `(failed_runs / total_runs) * 100`
- **Units**: Percentage (%)
- **Precision**: 1 decimal place
- **Critical for**: Reliability assessment

## Hardware Metrics

### **1. GPU Utilization (%)**
- **Definition**: Percentage of GPU compute resources used
- **Measurement**: Via `nvidia-smi` monitoring
- **Units**: Percentage (%)
- **Precision**: 1 decimal place
- **Critical for**: Hardware efficiency analysis

### **2. Tensor Core Usage**
- **Definition**: Whether specialized tensor cores are utilized
- **Measurement**: Binary indicator (Yes/No)
- **Critical for**: Hardware acceleration effectiveness

### **3. Power Consumption (Optional)**
- **Definition**: GPU power draw during inference
- **Measurement**: Via `nvidia-smi` or hardware monitoring
- **Units**: Watts (W)
- **Precision**: 1 decimal place
- **Critical for**: Energy efficiency analysis

### **4. Thermal Performance (Optional)**
- **Definition**: GPU temperature during sustained inference
- **Measurement**: Via `nvidia-smi`
- **Units**: Celsius (°C)
- **Precision**: 1 decimal place
- **Critical for**: Thermal throttling analysis

## Measurement Methodology

### **Test Configuration**
- **Hardware**: Tesla T4 (15GB VRAM, CUDA 12.6)
- **Software**: PyTorch 2.8.0+cu126, Transformers 4.57.0
- **Test Duration**: Minimum 100 runs per configuration
- **Warmup Runs**: 10 runs before measurement (excluded from results)
- **Input Consistency**: Identical prompts across all tests

### **Standard Test Prompts**
```python
test_prompts = [
    "Hello, how are you?",
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a short story about a robot.",
    "Calculate 15 * 23 =",
    "What are the benefits of renewable energy?",
    "Describe the process of photosynthesis.",
    "What is machine learning?",
    "Tell me a joke.",
    "How does a neural network work?"
]
```

### **Measurement Process**
1. **Environment Setup**: Clear GPU memory, set random seeds
2. **Model Loading**: Load model and measure loading time
3. **Warmup**: Run 10 inference cycles (excluded from results)
4. **Measurement**: Run 100 inference cycles with timing
5. **Data Collection**: Record all metrics simultaneously
6. **Validation**: Verify output quality and consistency

### **Statistical Analysis**
- **Mean**: Average performance across all runs
- **Standard Deviation**: Variability in measurements
- **Confidence Intervals**: 95% confidence level
- **Outlier Detection**: Remove runs >3σ from mean

## Data Collection Template

### **CSV Structure**
```csv
model_name,quantization_method,precision,model_size_mb,parameters_millions,inference_speed_tokens_per_sec,memory_usage_gb,memory_reduction_percent,speedup_factor,accuracy_metric,hardware_config,gpu_utilization_percent,tensor_core_usage,test_prompt,generated_output,timestamp,notes
```

### **Field Definitions**

#### **Model Information**
- **model_name**: Hugging Face model identifier
- **quantization_method**: BitsAndBytes, SmoothQuant, etc.
- **precision**: FP16, INT8, INT4, etc.
- **model_size_mb**: On-disk model size in MB
- **parameters_millions**: Number of parameters in millions

#### **Performance Data**
- **inference_speed_tokens_per_sec**: Average tokens per second
- **memory_usage_gb**: Peak GPU memory usage
- **memory_reduction_percent**: Memory savings vs baseline
- **speedup_factor**: Performance improvement multiplier

#### **Quality Metrics**
- **accuracy_metric**: Output quality assessment
- **test_prompt**: Input prompt used for testing
- **generated_output**: Actual model output

#### **Hardware Data**
- **hardware_config**: GPU and CUDA version
- **gpu_utilization_percent**: GPU usage percentage
- **tensor_core_usage**: Whether tensor cores were used

#### **Metadata**
- **timestamp**: When measurement was taken
- **notes**: Additional observations or issues

## Quality Assurance

### **Validation Checks**
1. **Reproducibility**: Results consistent across multiple runs
2. **Baseline Comparison**: Quantized model vs original model
3. **Hardware Consistency**: Same GPU configuration for all tests
4. **Software Consistency**: Same library versions for all tests
5. **Input Consistency**: Identical prompts and parameters

### **Error Handling**
- **Failed Runs**: Log and exclude from averages
- **Memory Errors**: Record and analyze causes
- **Timeout Issues**: Set reasonable time limits
- **Quality Degradation**: Flag significant output changes

## Reporting Standards

### **Summary Statistics**
- **Mean ± Standard Deviation**: For continuous metrics
- **Count and Percentage**: For categorical metrics
- **Confidence Intervals**: For statistical significance
- **Effect Size**: For practical significance

### **Visualization Requirements**
- **Bar Charts**: Performance comparisons
- **Line Graphs**: Trend analysis
- **Scatter Plots**: Correlation analysis
- **Heat Maps**: Multi-dimensional comparisons

### **Documentation Standards**
- **Methodology**: Detailed measurement process
- **Assumptions**: Any limitations or constraints
- **Limitations**: Hardware or software constraints
- **Recommendations**: Based on results

## Implementation Guidelines

### **Code Standards**
```python
# Example measurement function
def measure_inference_speed(model, tokenizer, prompt, num_runs=100):
    times = []
    for i in range(num_runs):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'tokens_per_sec': 10 / np.mean(times)
    }
```

### **Data Validation**
```python
def validate_results(results):
    # Check for reasonable values
    assert results['tokens_per_sec'] > 0, "Invalid speed measurement"
    assert results['memory_usage_gb'] > 0, "Invalid memory measurement"
    assert 0 <= results['gpu_utilization_percent'] <= 100, "Invalid GPU utilization"
    
    return results
```

## Conclusion

These standardized metrics provide a comprehensive framework for evaluating LLM quantization performance. By following these definitions and methodologies, we ensure:

- **Consistency**: Comparable results across experiments
- **Reproducibility**: Reliable and verifiable measurements
- **Completeness**: Coverage of all critical performance aspects
- **Quality**: High standards for data collection and analysis

This framework supports our hardware/software co-design analysis and enables meaningful comparison of quantization strategies across different models and hardware configurations.

---

**This document serves as the definitive guide for all performance measurements in the LLM quantization project.**
