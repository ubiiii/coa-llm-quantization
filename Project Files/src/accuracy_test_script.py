#!/usr/bin/env python3
"""
Accuracy Testing Script for LLM Quantization Project

This script runs comprehensive accuracy evaluation including perplexity measurements
on WikiText dataset for all quantized models.

Usage in Colab:
1. Upload this script to your Colab environment
2. Run the script to get accuracy measurements
3. Copy the results back to update the project

Team: CipherCore (Utkarsh & Sami)
Project: Hardware/Software Co-Design for LLM Quantization
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import json
import time
from datetime import datetime

def calculate_perplexity(model, tokenizer, texts, max_length=512, device="cuda"):
    """Calculate perplexity on a list of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts[:50]:  # Limit to first 50 texts for efficiency
            try:
                # Tokenize text
                inputs = tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=max_length,
                    padding=True
                ).to(device)
                
                # Forward pass
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                # Count tokens
                num_tokens = inputs["input_ids"].numel()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                
            except Exception as e:
                print(f"Error processing text: {e}")
                continue
    
    if total_tokens == 0:
        return {"perplexity": float('inf'), "total_tokens": 0}
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_texts": len(texts[:50])
    }

def load_wikitext_sample(num_samples=100):
    """Load a sample of WikiText data for evaluation."""
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50]
        return texts[:num_samples]
    except Exception as e:
        print(f"Error loading WikiText: {e}")
        # Fallback test texts
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "The weather today is sunny and warm.",
            "Python is a popular programming language.",
            "Neural networks are inspired by biological neurons."
        ] * 20

def test_model_accuracy(model_name, precision, device="cuda"):
    """Test accuracy for a specific model configuration."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name} with {precision} precision")
    print(f"{'='*60}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate configuration
        if precision == "FP16":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        elif precision == "INT8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        
        # Load test data
        test_texts = load_wikitext_sample(100)
        print(f"Loaded {len(test_texts)} test samples")
        
        # Calculate perplexity
        print("Calculating perplexity...")
        perplexity_results = calculate_perplexity(model, tokenizer, test_texts, device=device)
        
        # Test generation quality
        print("Testing generation quality...")
        test_prompts = [
            "Hello, how are you?",
            "What is machine learning?",
            "Tell me about artificial intelligence."
        ]
        
        generated_texts = []
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(generated)
        
        results = {
            "model_name": model_name,
            "precision": precision,
            "perplexity": perplexity_results["perplexity"],
            "avg_loss": perplexity_results["avg_loss"],
            "total_tokens": perplexity_results["total_tokens"],
            "num_samples": perplexity_results["num_texts"],
            "generated_texts": generated_texts,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"Results for {model_name} ({precision}):")
        print(f"  Perplexity: {results['perplexity']:.2f}")
        print(f"  Average Loss: {results['avg_loss']:.4f}")
        print(f"  Tokens Evaluated: {results['total_tokens']}")
        
        return results
        
    except Exception as e:
        print(f"Error testing {model_name} with {precision}: {e}")
        return {
            "model_name": model_name,
            "precision": precision,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def run_comprehensive_accuracy_tests():
    """Run comprehensive accuracy tests for all model configurations."""
    print("Starting Comprehensive Accuracy Testing")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        ("distilgpt2", "FP16"),
        ("distilgpt2", "INT8"),
        ("microsoft/DialoGPT-small", "FP16"),
        ("microsoft/DialoGPT-small", "INT8"),
        # Add more models as needed
    ]
    
    all_results = []
    
    for model_name, precision in test_configs:
        result = test_model_accuracy(model_name, precision)
        all_results.append(result)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"accuracy_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ACCURACY TEST SUMMARY")
    print(f"{'='*60}")
    
    for result in all_results:
        if "error" not in result:
            print(f"{result['model_name']} ({result['precision']}): "
                  f"Perplexity = {result['perplexity']:.2f}")
        else:
            print(f"{result['model_name']} ({result['precision']}): ERROR - {result['error']}")
    
    print(f"\nResults saved to: {filename}")
    print("\nCopy the results above to update the project analysis!")
    
    return all_results

if __name__ == "__main__":
    # Run the comprehensive accuracy tests
    results = run_comprehensive_accuracy_tests()
    
    # Print JSON results for easy copying
    print("\n" + "="*60)
    print("JSON RESULTS (copy this to update the project):")
    print("="*60)
    print(json.dumps(results, indent=2))
