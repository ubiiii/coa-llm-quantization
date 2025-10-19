# RUNNING.md (quick start)

## Install (Colab, GPU runtime)
pip install --no-cache-dir onnx==1.16.2 onnxruntime-gpu==1.19.2
pip install --no-cache-dir "torch==2.8.0+cu126" "torchvision==0.23.0+cu126" "torchaudio==2.8.0+cu126" -f https://download.pytorch.org/whl/cu126
pip install --no-cache-dir transformers==4.57.0 datasets==2.20.0 accelerate==1.1.0 numpy==1.26.4

## Verify ORT provider
import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())

## Steps
1) Export `distilgpt2` → `onnx/model.onnx`
2) Quantize to INT8 → `onnx/model_int8.onnx`
3) Benchmark with ORT (CUDA if available, else CPU)

## Artifacts produced
- onnx/model.onnx
- onnx/model_int8.onnx
- results/quantization_throughput.csv
- results/quantization_memory.csv
