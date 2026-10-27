# Lab 02: Training Bottlenecks — GPU Utilisation and Memory

Lab 01 profiled inference and small training loops. This lab moves to fine-tuning, where the
bottleneck is usually not the GPU being slow but the GPU being **idle, or holding the wrong things**.

Both exercises fine-tune TinyLlama and ship as a **baseline / fixed pair**. Profile the baseline
first and form a hypothesis before opening the second notebook.

## Prerequisites

Complete Lab 01 first — these exercises assume you can already launch a profile from JupyterLab and
read an Nsight Systems timeline.

## Lab structure

### 01-gpu-utilization

- `ft-tinyllama-gpu-utilization-baseline.ipynb`
- `ft-tinyllama-gpu-utilization-solution.ipynb`

A fine-tuning run where the GPUs look busy and throughput is poor anyway. The saved outputs in the
baseline show four GPUs with ~23 GB each and the layer-to-device map, which is the context you need
to interpret the timeline.

**The point:** device utilisation is a misleading metric. A GPU reported as busy may be waiting on
the data pipeline, synchronising, or running kernels too small to saturate it. Find where the gaps
are before changing anything.

### 02-memory-optimization

- `ft-tinyllama-lora-bf16-baseline.ipynb`
- `ft-tinyllama-lora-bf16-optimized.ipynb`
- `training_parameters_reference_guide.md` — what each training parameter does and how it
  interacts with the others

LoRA and bf16 fine-tuning, and what each one actually buys you. Both are usually adopted on the
assumption that they are free wins; the profile shows where they help, where they shift cost rather
than remove it, and how they interact with batch size and gradient accumulation.

## Profiling workflow

1. Run the baseline **without** profiling to confirm it works
2. Profile it, and read the timeline before forming an opinion
3. Write down where you think the time is going, and why
4. Open the fixed notebook and compare against your hypothesis

Step 3 is the one worth not skipping. Comparing the two notebooks without predicting first teaches
you the fix but not the diagnosis, and the diagnosis is the transferable part.

## Reference captures

`static-assets/lab-02/` holds the Nsight timeline captures from both exercises, so the comparison
can be followed without access to a multi-GPU instance.
