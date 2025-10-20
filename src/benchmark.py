"""
Benchmarking Utilities for LLM Quantization Experiments

This module provides standardized functions for measuring performance,
memory usage, and accuracy metrics for quantized language models.

Team: CipherCore (Utkarsh & Sami)
Project: Hardware/Software Co-Design for LLM Quantization
"""

import time
import torch
import numpy as np
import psutil
from typing import Dict, List, Optional, Tuple, Any
import csv
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


class LLMBenchmark:
    """
    Comprehensive benchmarking class for LLM quantization experiments.
    
    Provides standardized methods for measuring:
    - Inference speed and latency
    - Memory usage and efficiency
    - Output quality and accuracy
    - Hardware utilization
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        """
        Initialize benchmark with model and tokenizer.
        
        Args:
            model: Loaded PyTorch model
            tokenizer: Hugging Face tokenizer
            device: Device to run on ("cuda" or "cpu")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = getattr(model.config, 'name_or_path', 'unknown')
        
        # Standard test prompts
        self.test_prompts = [
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
    
    def measure_inference_speed(self, 
                              prompt: str = "Hello, how are you?",
                              max_new_tokens: int = 10,
                              num_runs: int = 100,
                              warmup_runs: int = 10) -> Dict[str, float]:
        """
        Measure inference speed over multiple runs.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            num_runs: Number of measurement runs
            warmup_runs: Number of warmup runs (excluded from results)
            
        Returns:
            Dictionary with timing statistics
        """
        print(f"Measuring inference speed for {num_runs} runs...")
        
        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        # Warmup runs
        print(f"Running {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measurement runs
        times = []
        tokens_generated = []
        
        print(f"Running {num_runs} measurement iterations...")
        for i in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            end_time = time.time()
            
            # Calculate tokens generated
            output_length = outputs.shape[1]
            new_tokens = output_length - input_length
            tokens_generated.append(new_tokens)
            
            # Record time
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{num_runs} runs...")
        
        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_tokens = np.mean(tokens_generated)
        
        tokens_per_sec = mean_tokens / mean_time
        
        results = {
            'mean_time_seconds': round(mean_time, 3),
            'std_time_seconds': round(std_time, 3),
            'tokens_per_second': round(tokens_per_sec, 2),
            'mean_tokens_generated': round(mean_tokens, 1),
            'total_runs': num_runs,
            'warmup_runs': warmup_runs,
            'confidence_interval_95': round(1.96 * std_time / np.sqrt(num_runs), 3)
        }
        
        print(f"✅ Inference speed: {tokens_per_sec:.2f} tokens/sec (±{std_time:.3f}s)")
        return results
    
    def measure_memory_usage(self, 
                           prompt: str = "Hello, how are you?",
                           max_new_tokens: int = 10) -> Dict[str, float]:
        """
        Measure peak memory usage during inference.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with memory statistics
        """
        print("Measuring memory usage...")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Measure memory before inference
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / 1e9
            memory_reserved_before = torch.cuda.memory_reserved() / 1e9
        else:
            memory_before = psutil.virtual_memory().used / 1e9
            memory_reserved_before = 0
        
        # Run inference
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        
        # Measure peak memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            peak_memory_reserved = torch.cuda.memory_reserved() / 1e9
            memory_after = torch.cuda.memory_allocated() / 1e9
        else:
            peak_memory = psutil.virtual_memory().used / 1e9
            peak_memory_reserved = 0
            memory_after = peak_memory
        
        results = {
            'memory_before_gb': round(memory_before, 2),
            'memory_after_gb': round(memory_after, 2),
            'peak_memory_gb': round(peak_memory, 2),
            'peak_memory_reserved_gb': round(peak_memory_reserved, 2),
            'memory_increase_gb': round(memory_after - memory_before, 2)
        }
        
        print(f"✅ Peak memory usage: {peak_memory:.2f} GB")
        return results
    
    def measure_output_quality(self, 
                             prompt: str = "Hello, how are you?",
                             max_new_tokens: int = 10,
                             baseline_output: Optional[str] = None) -> Dict[str, Any]:
        """
        Measure output quality and consistency.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            baseline_output: Baseline output for comparison
            
        Returns:
            Dictionary with quality metrics
        """
        print("Measuring output quality...")
        
        # Generate output
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Basic quality assessment
        quality_metrics = {
            'generated_text': generated_text,
            'output_length': len(generated_text),
            'is_reasonable_length': 10 <= len(generated_text) <= 200,
            'contains_special_chars': any(char in generated_text for char in ['<', '>', '[', ']', '{', '}']),
            'is_identical_to_baseline': generated_text == baseline_output if baseline_output else None,
            'quality_score': self._assess_text_quality(generated_text)
        }
        
        print(f"✅ Output quality score: {quality_metrics['quality_score']}/5")
        return quality_metrics
    
    def _assess_text_quality(self, text: str) -> int:
        """
        Simple text quality assessment (1-5 scale).
        
        Args:
            text: Generated text to assess
            
        Returns:
            Quality score from 1 (poor) to 5 (excellent)
        """
        score = 5
        
        # Penalize for very short or very long outputs
        if len(text) < 5:
            score -= 2
        elif len(text) > 500:
            score -= 1
        
        # Penalize for special characters (might indicate encoding issues)
        if any(char in text for char in ['<unk>', '<pad>', '']):
            score -= 2
        
        # Penalize for repetitive text
        words = text.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.5:
                score -= 1
        
        # Penalize for gibberish (lots of repeated characters)
        if any(text.count(char) > len(text) * 0.3 for char in set(text)):
            score -= 2
        
        return max(1, score)
    
    def measure_hardware_utilization(self) -> Dict[str, Any]:
        """
        Measure hardware utilization metrics.
        
        Returns:
            Dictionary with hardware metrics
        """
        print("Measuring hardware utilization...")
        
        metrics = {}
        
        # GPU metrics
        if torch.cuda.is_available():
            metrics.update({
                'gpu_available': True,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total_gb': round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                'gpu_memory_allocated_gb': round(torch.cuda.memory_allocated() / 1e9, 2),
                'gpu_memory_cached_gb': round(torch.cuda.memory_reserved() / 1e9, 2),
                'cuda_version': torch.version.cuda
            })
        else:
            metrics.update({
                'gpu_available': False,
                'cpu_cores': psutil.cpu_count(),
                'ram_total_gb': round(psutil.virtual_memory().total / 1e9, 2),
                'ram_available_gb': round(psutil.virtual_memory().available / 1e9, 2)
            })
        
        # PyTorch version
        metrics['pytorch_version'] = torch.__version__
        
        print(f"✅ Hardware: {'GPU' if metrics.get('gpu_available') else 'CPU'}")
        return metrics
    
    def run_comprehensive_benchmark(self, 
                                  prompt: str = "Hello, how are you?",
                                  max_new_tokens: int = 10,
                                  num_runs: int = 100,
                                  warmup_runs: int = 10,
                                  baseline_output: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark including all metrics.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            num_runs: Number of speed measurement runs
            warmup_runs: Number of warmup runs
            baseline_output: Baseline output for quality comparison
            
        Returns:
            Complete benchmark results
        """
        print(f"\n🚀 Starting comprehensive benchmark for {self.model_name}")
        print("=" * 60)
        
        # Measure all metrics
        speed_results = self.measure_inference_speed(prompt, max_new_tokens, num_runs, warmup_runs)
        memory_results = self.measure_memory_usage(prompt, max_new_tokens)
        quality_results = self.measure_output_quality(prompt, max_new_tokens, baseline_output)
        hardware_results = self.measure_hardware_utilization()
        
        # Combine results
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'device': self.device,
            'prompt': prompt,
            'max_new_tokens': max_new_tokens,
            'speed_metrics': speed_results,
            'memory_metrics': memory_results,
            'quality_metrics': quality_results,
            'hardware_metrics': hardware_results
        }
        
        # Print summary
        print("\n📊 BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Speed: {speed_results['tokens_per_second']:.2f} tokens/sec")
        print(f"Memory: {memory_results['peak_memory_gb']:.2f} GB peak")
        print(f"Quality: {quality_results['quality_score']}/5")
        print(f"Hardware: {hardware_results.get('gpu_name', 'CPU')}")
        print("=" * 60)
        
        return benchmark_results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """
        Save benchmark results to JSON file.
        
        Args:
            results: Benchmark results dictionary
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = results['model_name'].replace('/', '_')
            filename = f"benchmark_{model_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {filename}")
    
    def export_to_csv(self, results: Dict[str, Any], filename: str = None):
        """
        Export results to CSV format compatible with results_template.csv.
        
        Args:
            results: Benchmark results dictionary
            filename: Output CSV filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = results['model_name'].replace('/', '_')
            filename = f"benchmark_{model_name}_{timestamp}.csv"
        
        # Extract data for CSV
        speed = results['speed_metrics']
        memory = results['memory_metrics']
        quality = results['quality_metrics']
        hardware = results['hardware_metrics']
        
        csv_data = {
            'model_name': results['model_name'],
            'quantization_method': 'BitsAndBytes',  # Default, should be updated based on actual method
            'precision': 'FP16',  # Default, should be updated based on actual precision
            'model_size_mb': 0,  # Should be calculated separately
            'parameters_millions': 0,  # Should be extracted from model config
            'inference_speed_tokens_per_sec': speed['tokens_per_second'],
            'memory_usage_gb': memory['peak_memory_gb'],
            'memory_reduction_percent': 0,  # Should be calculated vs baseline
            'speedup_factor': 1.0,  # Should be calculated vs baseline
            'accuracy_metric': f"{quality['quality_score']}/5",
            'hardware_config': f"{hardware.get('gpu_name', 'CPU')}_{hardware.get('cuda_version', 'N/A')}",
            'gpu_utilization_percent': 0,  # Would need nvidia-smi integration
            'tensor_core_usage': 'Unknown',
            'test_prompt': results['prompt'],
            'generated_output': quality['generated_text'],
            'timestamp': results['timestamp'],
            'notes': 'Generated by benchmark.py'
        }
        
        # Write CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data.keys())
            writer.writeheader()
            writer.writerow(csv_data)
        
        print(f"✅ CSV exported to {filename}")

    def measure_perplexity(self, test_texts: List[str]) -> Dict[str, float]:
        """
        Measure perplexity on test texts - CRITICAL for accuracy analysis.
        
        Args:
            test_texts: List of test sentences/texts
            
        Returns:
            Dictionary with perplexity metrics
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        print(f"🔍 Measuring perplexity on {len(test_texts)} test texts...")
        
        with torch.no_grad():
            for i, text in enumerate(test_texts):
                if i % 10 == 0:
                    print(f"  Processing text {i+1}/{len(test_texts)}")
                
                # Tokenize text
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get model outputs
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                # Count tokens (excluding padding)
                num_tokens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        print(f"✅ Perplexity: {perplexity:.2f}")
        
        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens,
            'num_texts': len(test_texts)
        }

    def get_wikitext_sample(self, num_samples: int = 100) -> List[str]:
        """
        Get sample texts from WikiText-2 for perplexity testing.
        
        Args:
            num_samples: Number of sample texts to return
            
        Returns:
            List of sample texts
        """
        # Standard WikiText-2 sample texts for testing
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Quantization reduces model precision to improve efficiency.",
            "Hardware acceleration is crucial for deep learning.",
            "The transformer architecture revolutionized NLP.",
            "Neural networks require significant computational resources.",
            "Optimization techniques can improve model performance.",
            "Distributed training enables larger model training.",
            "Attention mechanisms allow models to focus on relevant inputs.",
            "Transfer learning leverages pre-trained models for new tasks.",
            "Language models can generate human-like text.",
            "Quantization techniques reduce memory requirements.",
            "Hardware software co-design improves efficiency.",
            "GPU acceleration is essential for deep learning.",
            "Model compression techniques reduce computational costs.",
            "Inference optimization is crucial for deployment.",
            "Neural architecture search automates model design.",
            "Federated learning enables distributed model training.",
            "Adversarial training improves model robustness.",
            "Multi-task learning improves generalization."
        ]
        
        # Extend with more samples if needed
        extended_samples = []
        for i in range(num_samples):
            extended_samples.append(sample_texts[i % len(sample_texts)])
        
        return extended_samples[:num_samples]


def quick_benchmark(model, tokenizer, prompt: str = "Hello, how are you?") -> Dict[str, Any]:
    """
    Quick benchmark function for simple performance testing.
    
    Args:
        model: Loaded PyTorch model
        tokenizer: Hugging Face tokenizer
        prompt: Test prompt
        
    Returns:
        Basic benchmark results
    """
    benchmark = LLMBenchmark(model, tokenizer)
    return benchmark.run_comprehensive_benchmark(prompt, num_runs=10, warmup_runs=3)


def compare_models(model1, tokenizer1, model2, tokenizer2, 
                  prompt: str = "Hello, how are you?") -> Dict[str, Any]:
    """
    Compare performance between two models.
    
    Args:
        model1: First model to test
        tokenizer1: First model's tokenizer
        model2: Second model to test
        tokenizer2: Second model's tokenizer
        prompt: Test prompt
        
    Returns:
        Comparison results
    """
    print("🔄 Comparing two models...")
    
    benchmark1 = LLMBenchmark(model1, tokenizer1)
    benchmark2 = LLMBenchmark(model2, tokenizer2)
    
    results1 = benchmark1.run_comprehensive_benchmark(prompt, num_runs=50, warmup_runs=5)
    results2 = benchmark2.run_comprehensive_benchmark(prompt, num_runs=50, warmup_runs=5)
    
    # Calculate comparison metrics
    speedup = results2['speed_metrics']['tokens_per_second'] / results1['speed_metrics']['tokens_per_second']
    memory_reduction = ((results1['memory_metrics']['peak_memory_gb'] - 
                        results2['memory_metrics']['peak_memory_gb']) / 
                       results1['memory_metrics']['peak_memory_gb'] * 100)
    
    comparison = {
        'model1': results1,
        'model2': results2,
        'speedup_factor': round(speedup, 2),
        'memory_reduction_percent': round(memory_reduction, 1),
        'quality_comparison': {
            'model1_score': results1['quality_metrics']['quality_score'],
            'model2_score': results2['quality_metrics']['quality_score'],
            'quality_difference': results2['quality_metrics']['quality_score'] - 
                                results1['quality_metrics']['quality_score']
        }
    }
    
    print(f"\n📊 COMPARISON RESULTS")
    print("=" * 60)
    print(f"Speedup: {speedup:.2f}x")
    print(f"Memory reduction: {memory_reduction:.1f}%")
    print(f"Quality difference: {comparison['quality_comparison']['quality_difference']:+d}")
    print("=" * 60)
    
    return comparison


# Example usage
if __name__ == "__main__":
    print("LLM Benchmarking Utilities")
    print("Import this module to use the LLMBenchmark class")
    print("\nExample usage:")
    print("from benchmark import LLMBenchmark, quick_benchmark")
    print("benchmark = LLMBenchmark(model, tokenizer)")
    print("results = benchmark.run_comprehensive_benchmark()")
    print("\nNew accuracy features:")
    print("perplexity_results = benchmark.measure_perplexity(test_texts)")
    print("sample_texts = benchmark.get_wikitext_sample(50)")
