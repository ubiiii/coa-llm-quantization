#!/usr/bin/env python3
"""
Fixed Accuracy Test Script for Colab

This script fixes the bitsandbytes version issue and runs accuracy tests.
Run this in Google Colab after updating bitsandbytes.

Usage in Colab:
1. First run: !pip install -U bitsandbytes transformers accelerate
2. Restart runtime: Runtime > Restart and run all
3. Upload and run this script

Team: CipherCore (Utkarsh & Sami)
Project: Hardware/Software Co-Design for LLM Quantization
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def test_model_accuracy(model_name, precision="FP16", max_samples=50):
    """Test model accuracy with perplexity measurement."""
    
    print(f"\n🔄 Testing {model_name} ({precision})...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization if needed
        quantization_config = None
        if precision == "INT8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if precision == "FP16" else None,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        model.eval()
        
        # Load WikiText-2 dataset
        print("📥 Loading WikiText-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # Calculate perplexity
        total_loss = 0
        total_tokens = 0
        num_samples = 0
        generated_texts = []
        
        print(f"🧮 Calculating perplexity on {max_samples} samples...")
        
        for i, example in enumerate(dataset):
            if i >= max_samples:
                break
                
            text = example["text"].strip()
            if len(text) < 50:  # Skip very short texts
                continue
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(model.device)
            
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
                total_loss += loss * input_ids.size(1)
                total_tokens += input_ids.size(1)
                num_samples += 1
            
            # Generate sample text
            if i < 3:  # Generate text for first 3 samples
                prompt = text[:100] + "..." if len(text) > 100 else text
                with torch.no_grad():
                    generated = model.generate(
                        input_ids[:1],  # Use first token as prompt
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                    generated_texts.append(generated_text)
        
        # Calculate final metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss) if avg_loss != float('inf') else float('inf')
        
        result = {
            "model_name": model_name,
            "precision": precision,
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
            "num_samples": num_samples,
            "generated_texts": generated_texts,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ {model_name} ({precision}): Perplexity = {perplexity:.2f}")
        return result
        
    except Exception as e:
        error_result = {
            "model_name": model_name,
            "precision": precision,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        print(f"❌ Error testing {model_name} ({precision}): {e}")
        return error_result

def run_accuracy_tests():
    """Run accuracy tests for all models and precisions."""
    
    print("🚀 Starting Accuracy Tests...")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        ("distilgpt2", "FP16"),
        ("distilgpt2", "INT8"),
        ("microsoft/DialoGPT-small", "FP16"),
        ("microsoft/DialoGPT-small", "INT8")
    ]
    
    results = []
    
    for model_name, precision in test_configs:
        result = test_model_accuracy(model_name, precision)
        results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"accuracy_test_results_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 RESULTS SUMMARY:")
    print("=" * 60)
    
    for result in results:
        if "error" in result:
            print(f"{result['model_name']} ({result['precision']}): ERROR - {result['error']}")
        else:
            print(f"{result['model_name']} ({result['precision']}): Perplexity = {result['perplexity']:.2f}")
    
    print(f"\n💾 Results saved to: {filename}")
    
    return results, filename

if __name__ == "__main__":
    results, filename = run_accuracy_tests()
    
    print("\n" + "=" * 60)
    print("🎉 Accuracy Tests Complete!")
    print(f"📁 Results file: {filename}")
    print("📋 Copy the results above to update the project analysis!")
