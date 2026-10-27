# Lab 01: GPU Profiling with NVIDIA Nsight Systems

This lab introduces GPU profiling techniques using NVIDIA Nsight Systems integrated with JupyterLab. You'll learn to identify and resolve performance bottlenecks in PyTorch applications, from simple operations to complex neural networks and LLM inference.

## Overview

Lab 01 contains hands-on exercises that demonstrate:
- Cell-level profiling in Jupyter notebooks using Nsight Systems extension
- Identifying GPU compute, memory, and synchronization bottlenecks
- Using NVTX annotations for custom profiling ranges
- Optimizing CNN training performance
- Optimizing LLM inference latency and resource utilization

## Prerequisites

- JupyterLab with NVIDIA Nsight Systems extension enabled
- CUDA-capable GPU
- PyTorch installed
- Access to Nsight Systems UI for report analysis

## Lab Structure

### 00-basic-pytorch-example
**Purpose:** Introduction to cell-level profiling with Nsight Systems

**What you'll learn:**
- Enable profiling with Nsight Systems extension in JupyterLab
- Use the green run button to profile individual cells
- Profile basic PyTorch operations (matrix multiplication, element-wise operations)
- Add custom NVTX annotations to mark specific code regions

**Key concepts:**
- Grey run button: Execute without profiling
- Green run button: Execute with profiling enabled
- NVTX push/pop ranges for custom instrumentation

### 01-simple-cnn
**Purpose:** Profile and optimize a simple CNN trained on synthetic data

**Notebooks:**
- `simple-cnn-baseline.ipynb`: Baseline implementation with bottlenecks
- `simple-cnn-optimized.ipynb`: Optimized version with performance improvements

**Bottlenecks identified:**
- **GPU Compute:** Convolution operations (conv1, conv2, conv3), matrix multiplications (fc1, fc2), ReLU activations
- **Memory:** CPU-to-GPU data transfers, intermediate tensor allocations, gradient storage
- **Synchronization:** `loss.item()` calls forcing GPU-to-CPU sync, single-threaded data loading
- **Gradient Computation:** Backward pass operations, optimizer parameter updates

**Optimizations applied:**
- Mixed precision training (FP16) for 2x memory savings
- Pre-move data to GPU to eliminate transfer overhead
- Multi-threaded DataLoader with `num_workers=4` and `pin_memory=True`
- Larger batch size (128 vs 64) for better GPU utilization
- Remove synchronous `loss.item()` calls during training
- Memory cleanup with `torch.cuda.empty_cache()`

**Expected improvements:**
- Reduced training time per epoch
- Higher GPU utilization
- Fewer CPU-GPU synchronization points
- Lower memory bandwidth usage

### 01-simple-cnn-with-nvtx
**Purpose:** Same CNN example with detailed NVTX annotations for fine-grained profiling

**Notebooks:**
- `simple-cnn-with-nvtx-baseline.ipynb`: Baseline with NVTX instrumentation
- `simple-cnn-with-nvtx-optimized.ipynb`: Optimized with NVTX instrumentation

**NVTX annotations added:**
- Model initialization
- Model forward pass
- Data preparation
- Per-batch execution
- Backward pass (loss computation, gradient computation)
- Optimizer step

**Benefits of NVTX:**
- Custom named ranges in Nsight Systems timeline
- Easier identification of specific bottlenecks
- Better correlation between code and GPU activity
- Domain-specific grouping of operations

### 02-llm-inference
**Purpose:** Profile and optimize LLM inference using TinyLlama model

**Notebooks:**
- `llm-inference-baseline.ipynb`: Baseline inference with FP32 precision
- `llm-inference-optimized.ipynb`: Optimized inference with BF16 precision

**Key metrics to observe:**
- Inference latency (time per token generation)
- GPU utilization during inference
- Memory utilization and allocation patterns

**Baseline configuration:**
- Model: TinyLlama-1.1B-Chat-v1.0
- Precision: FP32 (full precision)
- System prompt: Verbose (requests 200+ word responses)
- Generation parameters: `max_new_tokens=1024`, `temperature=1.0`, `top_k=300`, `top_p=0.95`

**Optimized configuration:**
- Precision: BF16 (bfloat16) for faster computation and lower memory
- System prompt: Concise (requests <50 word responses)
- Generation parameters: `max_new_tokens=150`, `temperature=0.7`, `top_k=50`

**Optimizations impact:**
- Reduced memory footprint (BF16 uses half the memory of FP32)
- Faster inference due to reduced token generation
- Lower latency per request
- Better GPU throughput

## How to Use This Lab

1. **Start with basic-pytorch-example** to familiarize yourself with the profiling workflow
2. **Enable Nsight Systems extension** in JupyterLab menu (only active when notebook is open)
3. **Restart kernel** when prompted to activate the green profiling button
4. **Run baseline notebooks** with profiling enabled (green button)
5. **Open generated reports** in Nsight Systems UI to analyze bottlenecks
6. **Review optimized notebooks** to understand applied fixes
7. **Run optimized notebooks** with profiling and compare reports
8. **Experiment** with different optimizations and observe their impact

## Profiling Workflow

1. Open a notebook in JupyterLab
2. Navigate to NVIDIA Nsight menu → Enable "Profiling with Nsight Systems"
3. Restart kernel when prompted
4. Select the cell you want to profile
5. Click the green run button (profiling enabled) or grey button (normal execution)
6. Wait for execution to complete
7. Access generated `.nsys-rep` files from the notebook directory
8. Open reports in Nsight Systems UI for analysis

## Key Takeaways

- **Cell-level profiling** enables targeted performance analysis without profiling entire notebooks
- **NVTX annotations** provide custom instrumentation for better trace correlation
- **Common bottlenecks** include CPU-GPU synchronization, memory transfers, and suboptimal batch sizes
- **Mixed precision** (FP16/BF16) significantly improves performance with minimal accuracy impact
- **DataLoader optimization** (num_workers, pin_memory) reduces data loading overhead
- **LLM inference** can be optimized through precision reduction and generation parameter tuning
- **Nsight Systems** provides comprehensive visibility into GPU utilization, memory operations, and kernel execution

## Next Steps

After completing Lab 01, proceed to:
- **Lab 02:** Deep dive into specific bottleneck patterns (device-to-host memcpy, OOM, GPU utilization)
- **Lab 03:** AI-assisted analysis of Nsight Systems reports

## Resources

- [NVTX Documentation](https://nvidia.github.io/NVTX/)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
