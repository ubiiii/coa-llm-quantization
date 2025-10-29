"""
Test script for benchmarking utilities.

This script tests the benchmark.py utilities on a baseline model
to ensure all functions work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from benchmark import LLMBenchmark, quick_benchmark
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_benchmark_utilities():
    """Test the benchmarking utilities with a simple model."""
    print("🧪 Testing Benchmarking Utilities")
    print("=" * 50)
    
    try:
        # Load a small model for testing
        print("Loading test model...")
        model_name = "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Test basic functionality
        print("\n1. Testing LLMBenchmark initialization...")
        benchmark = LLMBenchmark(model, tokenizer)
        print("✅ LLMBenchmark initialized successfully")
        
        print("\n2. Testing hardware utilization measurement...")
        hardware_results = benchmark.measure_hardware_utilization()
        print(f"✅ Hardware metrics: {hardware_results}")
        
        print("\n3. Testing output quality measurement...")
        quality_results = benchmark.measure_output_quality("Hello, how are you?")
        print(f"✅ Quality metrics: {quality_results['quality_score']}/5")
        
        print("\n4. Testing memory usage measurement...")
        memory_results = benchmark.measure_memory_usage("Hello, how are you?")
        print(f"✅ Memory metrics: {memory_results['peak_memory_gb']:.2f} GB")
        
        print("\n5. Testing inference speed measurement...")
        speed_results = benchmark.measure_inference_speed("Hello, how are you?", num_runs=5, warmup_runs=2)
        print(f"✅ Speed metrics: {speed_results['tokens_per_second']:.2f} tokens/sec")
        
        print("\n6. Testing comprehensive benchmark...")
        full_results = benchmark.run_comprehensive_benchmark("Hello, how are you?", num_runs=5, warmup_runs=2)
        print("✅ Comprehensive benchmark completed successfully")
        
        print("\n7. Testing quick benchmark function...")
        quick_results = quick_benchmark(model, tokenizer, "Hello, how are you?")
        print("✅ Quick benchmark completed successfully")
        
        print("\n🎉 All benchmark utilities tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_benchmark_utilities()
    if success:
        print("\n✅ Benchmark utilities are ready for use!")
    else:
        print("\n❌ Benchmark utilities need debugging!")
