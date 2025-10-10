"""
Colab-compatible test script for benchmarking utilities.

This script can be run directly in Google Colab to test all benchmark functions.
"""

# Run this entire cell in Google Colab to test the benchmark utilities

print("🧪 Testing Benchmarking Utilities in Colab")
print("=" * 50)

try:
    # Import required libraries
    from benchmark import LLMBenchmark, quick_benchmark
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    print("✅ All imports successful")
    
    # Load a small model for testing
    print("\n📥 Loading test model...")
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("✅ Model loaded successfully")
    
    # Test 1: LLMBenchmark initialization
    print("\n🔧 Test 1: LLMBenchmark initialization...")
    benchmark = LLMBenchmark(model, tokenizer)
    print("✅ LLMBenchmark initialized successfully")
    
    # Test 2: Hardware utilization
    print("\n🖥️ Test 2: Hardware utilization measurement...")
    hardware_results = benchmark.measure_hardware_utilization()
    print(f"✅ Hardware: {hardware_results['gpu_name'] if hardware_results.get('gpu_available') else 'CPU'}")
    
    # Test 3: Output quality
    print("\n📝 Test 3: Output quality measurement...")
    quality_results = benchmark.measure_output_quality("Hello, how are you?")
    print(f"✅ Quality score: {quality_results['quality_score']}/5")
    print(f"✅ Generated: '{quality_results['generated_text']}'")
    
    # Test 4: Memory usage
    print("\n💾 Test 4: Memory usage measurement...")
    memory_results = benchmark.measure_memory_usage("Hello, how are you?")
    print(f"✅ Peak memory: {memory_results['peak_memory_gb']:.2f} GB")
    
    # Test 5: Inference speed (quick test)
    print("\n⚡ Test 5: Inference speed measurement...")
    speed_results = benchmark.measure_inference_speed("Hello, how are you?", num_runs=5, warmup_runs=2)
    print(f"✅ Speed: {speed_results['tokens_per_second']:.2f} tokens/sec")
    
    # Test 6: Quick benchmark function
    print("\n🚀 Test 6: Quick benchmark function...")
    quick_results = quick_benchmark(model, tokenizer, "Hello, how are you?")
    print(f"✅ Quick benchmark: {quick_results['tokens_per_second']:.2f} tokens/sec")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("=" * 50)
    print("✅ Benchmark utilities are fully functional in Colab!")
    print("✅ Ready for Phase 3 experiments!")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    print("Please check that benchmark.py is uploaded and accessible")
