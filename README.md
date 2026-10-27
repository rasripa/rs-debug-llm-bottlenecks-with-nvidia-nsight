# Debug LLM Bottlenecks with NVIDIA Nsight 

A self-paced **learning workshop** — hands-on labs for profiling LLM training and inference on
GPUs with NVIDIA Nsight Systems, finding the real bottleneck, and fixing it.

Written for teaching rather than as a reference implementation or a deployable demo: the labs
favour clarity over production hardening. Work through them in order.

## What this is, and what it isn't

This is **learning content**. The labs are meant to be read, run and broken — not cloned into a
project. Several things are deliberately simplified so the teaching point stays visible:

- models and datasets are small and synthetic, chosen so a lab finishes in minutes rather than
  to represent a real training run
- the "optimised" notebooks fix one bottleneck each; they are not fully tuned implementations
- measurements depend heavily on instance type, driver and library versions — treat the numbers
  in the saved outputs as illustrative, and trust your own profile over mine
- nothing here has been cost-modelled or reviewed for production use

If you want to take an idea from here into something real, treat the lab as the explanation and
write the production version yourself.

## How the labs work

Most exercises ship as a **baseline / optimised pair**. Profile the baseline, read the timeline,
form a hypothesis about where the time is actually going, then compare against the optimised
notebook. The point is the diagnosis, not the fix.

## Contents

**Lab 01 — profiling fundamentals**
- `00-basic-pytorch-example` — cell-level profiling in JupyterLab with the Nsight Systems
  extension; profiling basic PyTorch operations; adding custom NVTX push/pop ranges
- `01-simple-cnn` — profile and optimise a small CNN on synthetic data (baseline / optimised)
- `01-simple-cnn-with-nvtx` — the same exercise with NVTX instrumentation, so the timeline maps
  onto your own code regions rather than raw kernel names
- `02-llm-inference` — profiling an LLM inference path (baseline / optimised)

**Lab 02 — training bottlenecks**
- `01-gpu-utilization` — fine-tuning TinyLlama where the GPUs look busy and throughput is poor
  (baseline / solution)
- `02-memory-optimization` — LoRA and bf16 fine-tuning, and what each actually buys you
  (baseline / optimised), with a training-parameters reference guide

`static-assets/` holds 32 annotated Nsight timeline captures from the baseline and optimised runs,
so the diagnosis in each lab can be followed without access to a GPU.

## Prerequisites

- A GPU instance with CUDA drivers
- NVIDIA Nsight Systems, and the Nsight Systems JupyterLab extension
- PyTorch. The notebooks install their remaining dependencies inline in the first cells, so
  there is no repository-level `requirements.txt`
- Familiarity with Python and basic PyTorch training loops

In JupyterLab the grey run button executes a cell normally; the green run button executes it with
profiling enabled.

## Notes

All data used in the labs is synthetic or from public model repositories.

---

A personal project — not an official AWS project, and not supported by AWS.
