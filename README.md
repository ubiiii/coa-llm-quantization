# COA Project: HW/SW Co-Design for LLM Quantization

**Team:** CipherCore (Utkarsh & Sami)

## Project Overview

This project investigates hardware-aware quantization strategies for Large Language Model (LLM) inference and compares performance/accuracy trade-offs between hardware-assisted and software-only quantization approaches.

## Research Papers

- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](SmoothQuant.txt)
- [HAQ: Hardware-Aware Automated Quantization with Mixed Precision](HAQ%20Hardware-Aware%20Automated%20Quantization%20with%20Mixed%20Precision.txt)

## Project Structure

```
├── Referance/          # Research papers and literature
├── updates/            # Project TODO list and tracking
├── proposal.txt        # Project proposal and objectives
└── README.md          # This file
```

## Expected Deliverables

- ✅ Survey of HW/SW co-design methods for LLM quantization
- 🔄 Experimental results using quantization libraries (BitsandBytes, SmoothQuant)
- ⏳ Analysis of performance, memory footprint, and accuracy trade-offs

## Current Progress

- ✅ **Phase 1:** Research & Planning (100% Complete)
- ✅ **Phase 2:** Environment Setup (Partial - Google Colab + Tesla T4)
- ✅ **Phase 3:** Baseline Experiments (FP16: 34.53 tokens/s, 4-bit: 157.11 tokens/s, 4.55× speedup)
- ⏳ **Phase 4:** Analysis & Discussion (Pending)
- ⏳ **Phase 5:** Documentation & Presentation (Pending)

## Key Results So Far

| Model | Precision | Speed | Speedup |
|-------|-----------|-------|---------|
| TinyLlama-1.1B | FP16 | 34.53 tokens/s | 1.0× (baseline) |
| Llama-3.2-1B | 4-bit | 157.11 tokens/s | **4.55×** |

## Next Steps

1. Complete INT8 quantization experiments
2. Hardware profiling and analysis
3. Create comparison visualizations
4. Write final report and presentation

## Repository

GitHub: [https://github.com/ubiiii/coa-llm-quantization](https://github.com/ubiiii/coa-llm-quantization)
