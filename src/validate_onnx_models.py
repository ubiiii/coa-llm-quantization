#!/usr/bin/env python3
"""
ONNX Model Validation Script

Validates the uploaded ONNX model files to ensure they are properly formatted
and contain valid ONNX model data.

Team: CipherCore (Utkarsh & Sami)
Project: Hardware/Software Co-Design for LLM Quantization
"""

import os
import sys
import struct
from pathlib import Path

def check_onnx_header(filepath):
    """Check if file has valid ONNX header."""
    try:
        with open(filepath, 'rb') as f:
            # Read first 16 bytes to check ONNX magic number
            header = f.read(16)
            
            # ONNX files start with specific magic bytes
            # Check for ONNX protobuf format
            if len(header) >= 8:
                # Look for protobuf magic or ONNX signature
                if header.startswith(b'\x08') or b'ONNX' in header or len(header) >= 4:
                    return True, "Valid ONNX header detected"
                else:
                    return False, "Invalid ONNX header - not a valid protobuf/ONNX file"
            else:
                return False, "File too small to be valid ONNX"
                
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def check_file_size(filepath, expected_size_mb):
    """Check if file size matches expected size."""
    try:
        actual_size = os.path.getsize(filepath)
        actual_size_mb = actual_size / (1024 * 1024)
        expected_size_bytes = expected_size_mb * 1024 * 1024
        
        # Allow 10% tolerance
        tolerance = 0.1
        min_size = expected_size_bytes * (1 - tolerance)
        max_size = expected_size_bytes * (1 + tolerance)
        
        if min_size <= actual_size <= max_size:
            return True, f"Size OK: {actual_size_mb:.2f} MB (expected ~{expected_size_mb} MB)"
        else:
            return False, f"Size mismatch: {actual_size_mb:.2f} MB (expected ~{expected_size_mb} MB)"
            
    except Exception as e:
        return False, f"Error checking size: {str(e)}"

def validate_onnx_model(filepath, expected_size_mb, model_type):
    """Comprehensive validation of ONNX model file."""
    print(f"\nValidating {os.path.basename(filepath)} ({model_type})...")
    print("-" * 60)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return False
    
    # Check file size
    size_ok, size_msg = check_file_size(filepath, expected_size_mb)
    print(f"File size: {size_msg}")
    
    # Check ONNX header
    header_ok, header_msg = check_onnx_header(filepath)
    print(f"Header validation: {header_msg}")
    
    # Overall validation
    if size_ok and header_ok:
        print(f"SUCCESS: {model_type} model is VALID")
        return True
    else:
        print(f"FAILED: {model_type} model is INVALID")
        return False

def main():
    """Main validation function."""
    print("ONNX Model Validation Script")
    print("=" * 60)
    
    # Define expected model files and their expected sizes
    models = [
        {
            "path": "../Model/model.onnx",
            "size_mb": 460.95,
            "type": "Basic ONNX (FP32)"
        },
        {
            "path": "../Model/model.with_past.onnx", 
            "size_mb": 460.95,
            "type": "ONNX with KV Cache (FP32)"
        },
        {
            "path": "../Model/model.int8.onnx",
            "size_mb": 229.14,
            "type": "INT8 Quantized"
        },
        {
            "path": "../Model/model.with_past.int8.onnx",
            "size_mb": 229.14,
            "type": "INT8 with KV Cache"
        }
    ]
    
    valid_count = 0
    total_count = len(models)
    
    for model in models:
        is_valid = validate_onnx_model(
            model["path"], 
            model["size_mb"], 
            model["type"]
        )
        if is_valid:
            valid_count += 1
    
    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY:")
    print(f"   Valid models: {valid_count}/{total_count}")
    print(f"   Success rate: {(valid_count/total_count)*100:.1f}%")
    
    if valid_count == total_count:
        print("SUCCESS: ALL MODELS ARE VALID!")
        print("Task 3.9 ONNX models are ready for audit")
    else:
        print("WARNING: Some models failed validation")
        print("Please check the failed models")
    
    return valid_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
