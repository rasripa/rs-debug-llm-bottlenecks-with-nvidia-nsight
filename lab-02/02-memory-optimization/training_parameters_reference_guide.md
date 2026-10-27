# Training Parameters Referrence Guide

**This guide covers the most important training parameters for PyTorch and Transformers training. Adjust values based on your specific hardware, dataset, and requirements.**
Pls note, the guide is not exhaustive. Feel free to apply your existing skills or refer to external documentation (as needed). Feel free to reach out to instructors if you like to brainstorm on any changes.


## Batch Size & Gradient Accumulation

### `per_device_train_batch_size`
- **Description**: Training batch size per GPU
- **Possible Values**: `1` to `64+` (depends on GPU memory)
- **Recommended**: `8-24` for large models, `32-64` for smaller models
- **Reason**: Balance between memory usage and training stability

### `per_device_eval_batch_size`
- **Description**: Evaluation batch size per GPU
- **Possible Values**: `1` to `128+`
- **Recommended**: `16-32` (usually larger than training batch)
- **Reason**: Evaluation uses less memory, can use larger batches

### `gradient_accumulation_steps`
- **Description**: Number of steps to accumulate gradients before updating
- **Possible Values**: `1` to `32+`
- **Recommended**: `4-8` for most setups
- **Reason**: Simulates larger batch sizes without memory increase

## Mixed Precision

### `fp16` / `bf16`
- **Description**: Enable mixed precision training for speed and memory savings
- **Possible Values**: `True` / `False`
- **Recommended**: `bf16=True` for A100+, `fp16=True` for older GPUs
- **Reason**: ~2x speedup and 50% memory reduction with minimal accuracy loss

### `fp16_full_eval` / `bf16_full_eval`
- **Description**: Use FP32 for evaluation even when training in mixed precision
- **Possible Values**: `True` / `False`
- **Recommended**: `True` for accurate metrics
- **Reason**: More precise evaluation results

## Gradient Control

### `max_grad_norm`
- **Description**: Gradient clipping to prevent exploding gradients
- **Possible Values**: `None`, `0.5`, `1.0`, `5.0`
- **Recommended**: `1.0` for stability, `None` for speed optimization
- **Reason**: Prevents training instability but causes DtoH transfers

### `gradient_checkpointing`
- **Description**: Trade compute for memory by recomputing activations
- **Possible Values**: `True` / `False`
- **Recommended**: `True` for large models with memory constraints
- **Reason**: Reduces memory usage at cost of ~20% slower training

## Data Loading

### `dataloader_num_workers`
- **Description**: Number of processes for data loading
- **Possible Values**: `0` to `8+`
- **Recommended**: `2-4` for most setups, `0` for debugging
- **Reason**: Parallel data loading improves training speed

### `dataloader_pin_memory`
- **Description**: Pin memory for faster CPU-GPU transfers
- **Possible Values**: `True` / `False`
- **Recommended**: `True` with sufficient RAM
- **Reason**: 20-30% faster data transfers to GPU

### `dataloader_persistent_workers`
- **Description**: Keep worker processes alive between epochs
- **Possible Values**: `True` / `False`
- **Recommended**: `True` when using multiple workers
- **Reason**: Avoids worker startup overhead

### `dataloader_drop_last`
- **Description**: Drop incomplete final batch
- **Possible Values**: `True` / `False`
- **Recommended**: `True` for consistent batch sizes
- **Reason**: Ensures all batches have same size

### `dataloader_prefetch_factor`
- **Description**: Number of batches to prefetch per worker
- **Possible Values**: `None`, `2`, `4`, `8`
- **Recommended**: `2-4` when using workers
- **Reason**: Better pipeline overlap between data loading and training

## Learning Rate & Scheduling

### `learning_rate`
- **Description**: Controls the size of parameter updates during training
- **Possible Values**: `1e-6` to `1e-2`
- **Recommended**: `2e-5` for fine-tuning, `1e-4` for training from scratch
- **Reason**: Good balance of stability and convergence speed

### `warmup_ratio` / `warmup_steps`
- **Description**: Gradually increases learning rate at the start of training
- **Possible Values**: `0.0` to `0.3` (ratio) or `0` to `1000+` (steps)
- **Recommended**: `warmup_ratio=0.1` or `warmup_steps=500`
- **Reason**: Prevents early training instability with large models

### `lr_scheduler_type`
- **Description**: Learning rate decay strategy
- **Possible Values**: `constant`, `linear`, `cosine`, `polynomial`
- **Recommended**: `cosine` for most cases
- **Reason**: Smooth decay with potential for escaping local minima

### `lr_scheduler_kwargs`
- **Description**: Additional parameters for LR scheduler
- **Example**: `{"T_max": 1000, "eta_min": 1e-6}` for cosine annealing
- **Recommended**: `{"T_max": total_steps, "eta_min": 1e-6}`
- **Reason**: Controls cosine cycle length and minimum LR

## Saving & Checkpointing

### `save_strategy`
- **Description**: When to save checkpoints
- **Possible Values**: `no`, `steps`, `epoch`
- **Recommended**: `steps` for long training, `epoch` for short training
- **Reason**: Regular checkpoints prevent loss of progress

### `save_steps`
- **Description**: Save checkpoint every N steps
- **Possible Values**: `100` to `5000+`
- **Recommended**: `500-1000` for most cases
- **Reason**: Balance between safety and storage space

### `save_total_limit`
- **Description**: Maximum number of checkpoints to keep
- **Possible Values**: `None`, `1`, `3`, `5`
- **Recommended**: `3` for most cases
- **Reason**: Keeps recent checkpoints while managing disk space

### `save_safetensors`
- **Description**: Use SafeTensors format instead of pickle
- **Possible Values**: `True` / `False`
- **Recommended**: `True`
- **Reason**: Safer, faster loading, and more secure format

## Logging & Monitoring

### `logging_strategy`
- **Description**: When to log training metrics
- **Possible Values**: `no`, `steps`, `epoch`
- **Recommended**: `steps` for monitoring
- **Reason**: Regular feedback on training progress

### `logging_steps`
- **Description**: Log metrics every N steps
- **Possible Values**: `1` to `1000+`
- **Recommended**: `50-100` for training, `500+` for DtoH optimization
- **Reason**: Balance between monitoring and performance overhead

### `report_to`
- **Description**: External logging services
- **Possible Values**: `[]`, `["wandb"]`, `["tensorboard"]`
- **Recommended**: `["wandb"]` for experiment tracking
- **Reason**: Excellent visualization and experiment management

### `disable_tqdm`
- **Description**: Disable progress bars
- **Possible Values**: `True` / `False`
- **Recommended**: `False` for interactive, `True` for production
- **Reason**: Cleaner output and slight performance improvement

## Evaluation

### `evaluation_strategy`
- **Description**: When to run evaluation
- **Possible Values**: `no`, `steps`, `epoch`
- **Recommended**: `epoch` for most cases
- **Reason**: Monitor overfitting without excessive overhead

### `eval_steps`
- **Description**: Evaluate every N steps
- **Possible Values**: `100` to `2000+`
- **Recommended**: `500-1000`
- **Reason**: Regular validation feedback

### `do_eval`
- **Description**: Enable/disable evaluation
- **Possible Values**: `True` / `False`
- **Recommended**: `True` for model development
- **Reason**: Essential for monitoring model performance

## Optimization

### `optim`
- **Description**: Optimizer algorithm
- **Possible Values**: `adamw_torch`, `adamw_hf`, `sgd`, `adafactor`
- **Recommended**: `adamw_torch`
- **Reason**: Best general-purpose optimizer for transformers

### `weight_decay`
- **Description**: L2 regularization strength
- **Possible Values**: `0.0` to `0.1`
- **Recommended**: `0.01`
- **Reason**: Prevents overfitting without being too aggressive

## Memory Optimization

### `skip_memory_metrics`
- **Description**: Skip GPU memory usage tracking
- **Possible Values**: `True` / `False`
- **Recommended**: `True` for DtoH optimization
- **Reason**: Reduces overhead and potential DtoH transfers

### `remove_unused_columns`
- **Description**: Remove dataset columns not used by model
- **Possible Values**: `True` / `False`
- **Recommended**: `True` for memory efficiency
- **Reason**: Reduces memory usage during training


## DtoH Optimization Settings

For minimizing Device-to-Host memory transfers:

```python
# Recommended DtoH-optimized settings:
max_grad_norm=None                    # Disable gradient clipping
logging_strategy="no"                 # Disable logging
save_strategy="no"                    # Disable checkpointing
skip_memory_metrics=True              # Skip memory tracking
disable_tqdm=True                     # Disable progress bars
evaluation_strategy="no"              # Disable evaluation
report_to=[]                          # No external logging
```

## Hardware-Specific Recommendations

### g6.2xlarge (L4 GPU)
```python
bf16=True                             # BF16 supported
per_device_train_batch_size=16        # Moderate batch size
dataloader_num_workers=2              # Enable multiprocessing
dataloader_pin_memory=True            # Faster transfers
```

## Quick Start Templates

### Standard Training
```python
TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    bf16=True,  # or fp16=True for older GPUs
    logging_steps=100,
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="epoch",
    report_to=["wandb"],
)
```

### DtoH-Optimized Training
```python
TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    bf16=True,
    max_grad_norm=None,
    logging_strategy="no",
    save_strategy="no",
    evaluation_strategy="no",
    disable_tqdm=True,
    skip_memory_metrics=True,
)
```

### Memory-Constrained Training
```python
TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    bf16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)
```

---

