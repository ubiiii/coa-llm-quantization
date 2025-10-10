# Results Directory

This directory stores all experimental results, benchmarks, and performance metrics.

## Contents

- **CSV files**: Structured performance data (inference speed, memory usage, accuracy)
- **JSON files**: Detailed experimental configurations and results
- **Log files**: Detailed execution logs and error messages
- **Screenshots**: Visual results and GPU profiling snapshots

## File Organization

- `baseline_results.csv`: FP16/FP32 baseline performance metrics
- `quantization_results.csv`: INT8/INT4 quantization results
- `hardware_profiling.csv`: GPU utilization and performance data
- `comparison_summary.json`: Comprehensive result comparison

## Data Format

CSV files should include columns for:
- Model name
- Quantization method
- Inference speed (tokens/sec)
- Memory usage (GB)
- Accuracy metrics (perplexity, etc.)
- Hardware configuration
- Timestamp

## Git Ignore

Note: `.gitignore` is configured to exclude large result files (*.csv, *.log) from Git tracking.
Only summary files and small datasets should be committed.

## Team Access

Both team members (Utkarsh & Sami) have read/write access to this directory.

