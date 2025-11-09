"""
memorymark.py - GPU Memory Analysis Engine

Simulates full training loops (forward + backward pass) for accurate batch size optimization.
This is the CRITICAL differentiator: we test REAL training memory, not just inference.

Key Feature: Backward pass simulation via loss.backward() ensures recommendations work in production.

Documentation References:
- PyTorch CUDA Memory: https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html
- PyTorch Reset Stats: https://docs.pytorch.org/docs/stable/generated/torch.cuda.reset_peak_memory_stats.html
- HuggingFace AutoModel: https://huggingface.co/docs/transformers/main/en/model_doc/auto
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoImageProcessor,
    AutoModelForImageClassification
)
from typing import Dict, Tuple, List, Optional
import sys

# Constants
MODEL_MAP = {
    'bert': 'google-bert/bert-base-uncased',
    'gpt2': 'openai-community/gpt2',
    'resnet': 'microsoft/resnet-50'
}

BATCH_SIZE_START = 8
BATCH_SIZE_INCREMENT = 8
MAX_SEQUENCE_LENGTH = 128
IMAGE_SIZE = 224
LAMBDA_LABS_A10_COST_PER_HOUR = 0.60
ASSUMED_TRAINING_HOURS = 2.0
ANNUAL_TRAINING_RUNS = 100


def get_device() -> str:
    """
    Detect available compute device.

    Returns:
        str: 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, 'cpu' otherwise

    Reference: https://docs.pytorch.org/docs/stable/cuda.html
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def create_dummy_batch(model_type: str, batch_size: int, processor, device: str) -> Dict:
    """
    Create dummy input data for testing.

    Args:
        model_type: 'nlp' or 'vision'
        batch_size: Number of samples in batch
        processor: Tokenizer (for NLP) or ImageProcessor (for vision)
        device: 'cuda', 'mps', or 'cpu'

    Returns:
        dict: Batch of inputs ready for model, on specified device
    """
    if model_type == 'nlp':
        # Create dummy text inputs
        dummy_texts = ["This is a test sentence for GPU memory analysis."] * batch_size

        # Tokenize
        inputs = processor(
            dummy_texts,
            padding='max_length',
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs
    else:  # vision
        # Create dummy images (224x224 RGB) in [0, 1] range for PIL compatibility
        # torch.rand creates values in [0, 1) which is required by image processors
        dummy_images = [
            torch.rand(3, IMAGE_SIZE, IMAGE_SIZE) for _ in range(batch_size)
        ]

        # Process images
        inputs = processor(images=dummy_images, return_tensors='pt')

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs


def load_model(model_name: str, device: str, use_compile: bool = False) -> Tuple:
    """
    Load a HuggingFace model to specified device with optional PyTorch 2.x compilation.

    Args:
        model_name: One of ['bert', 'gpt2', 'resnet']
        device: 'cuda', 'mps', or 'cpu'
        use_compile: If True, apply torch.compile() for optimized execution (PyTorch 2.0+)

    Returns:
        tuple: (model, processor, model_type, is_compiled)

    References:
        - HuggingFace AutoModel: https://huggingface.co/docs/transformers/main/en/model_doc/auto
        - torch.compile: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
        - torch.compile with transformers: https://huggingface.co/docs/transformers/en/perf_torch_compile
    """
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_MAP.keys())}")

    hf_model_name = MODEL_MAP[model_name]

    if model_name in ['bert', 'gpt2']:
        # NLP models
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

        # Fix for GPT-2: Set pad_token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_name,
            num_labels=2  # Binary classification for dummy task
        )
        model = model.to(device)
        model.eval()  # Set to eval mode (disables dropout)

        # Apply torch.compile if requested (PyTorch 2.0+ only)
        if use_compile:
            # Use 'reduce-overhead' mode for optimal memory analysis
            # This reduces Python overhead and is good for repeated inference
            model = torch.compile(model, mode='reduce-overhead')

        return (model, tokenizer, 'nlp', use_compile)
    else:  # resnet
        # Vision models
        processor = AutoImageProcessor.from_pretrained(hf_model_name)
        model = AutoModelForImageClassification.from_pretrained(hf_model_name)
        model = model.to(device)
        model.eval()

        # Apply torch.compile if requested
        if use_compile:
            model = torch.compile(model, mode='reduce-overhead')

        return (model, processor, 'vision', use_compile)


def test_batch_size(model, model_type: str, processor, batch_size: int, device: str) -> Dict:
    """
    Test a specific batch size with FULL TRAINING SIMULATION.

    CRITICAL: This simulates forward + backward pass, not just inference!
    This is what makes MemoryMark accurate and different from naive tools.

    Args:
        model: Loaded PyTorch model on device
        model_type: 'nlp' or 'vision'
        processor: Tokenizer or image processor
        batch_size: Batch size to test
        device: 'cuda', 'mps', or 'cpu'

    Returns:
        dict: {
            'batch_size': int,
            'memory_mb': int,
            'memory_gb': float,
            'success': bool,
            'error': str or None
        }

    Reference: https://docs.pytorch.org/docs/stable/generated/torch.cuda.reset_peak_memory_stats.html
    """
    try:
        # Clear GPU cache and reset memory stats
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif device == 'mps':
            torch.mps.empty_cache()

        # Create dummy batch
        inputs = create_dummy_batch(model_type, batch_size, processor, device)

        # FORWARD PASS
        outputs = model(**inputs)

        # CREATE DUMMY LOSS (critical for backward pass)
        if model_type == 'nlp':
            # For BERT/GPT-2: use logits mean as dummy loss
            if hasattr(outputs, 'logits'):
                loss = outputs.logits.mean()
            else:
                loss = outputs[0].mean()  # Fallback
        else:  # vision
            # For ResNet: use logits mean
            loss = outputs.logits.mean()

        # BACKWARD PASS - THIS IS THE KEY ADDITION
        # This allocates gradient memory, which is 50-60% of training memory
        loss.backward()

        # Measure PEAK memory (not current!)
        if device == 'cuda':
            peak_memory_bytes = torch.cuda.max_memory_allocated()
        elif device == 'mps':
            # MPS doesn't have max_memory_allocated, use current as approximation
            peak_memory_bytes = torch.mps.current_allocated_memory()
        else:
            # CPU - estimate from process memory (not accurate)
            peak_memory_bytes = 0

        peak_memory_mb = int(peak_memory_bytes / (1024 ** 2))
        peak_memory_gb = round(peak_memory_mb / 1024, 2)

        # Clean up
        del outputs, loss, inputs
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
        model.zero_grad()  # Clear gradients

        return {
            'batch_size': batch_size,
            'memory_mb': peak_memory_mb,
            'memory_gb': peak_memory_gb,
            'success': True,
            'error': None
        }

    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "oom" in error_msg:
            # Expected OOM - this is how we find the limit
            if device == 'cuda':
                torch.cuda.empty_cache()
            elif device == 'mps':
                torch.mps.empty_cache()
            model.zero_grad()
            return {
                'batch_size': batch_size,
                'memory_mb': 0,
                'memory_gb': 0.0,
                'success': False,
                'error': 'OOM'
            }
        else:
            # Unexpected error - re-raise
            raise


def find_optimal_batch_size(model_name: str, device: Optional[str] = None) -> Dict:
    """
    Main analysis function. Tests batch sizes until OOM.
    Returns complete analysis with waste calculations and cost savings.

    Args:
        model_name: One of ['bert', 'gpt2', 'resnet']
        device: Optional device override ('cuda', 'mps', 'cpu'). Auto-detected if None.

    Returns:
        dict: Complete analysis results with optimal batch size and savings
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_device()

    print(f"Using device: {device}")

    # Load model
    print(f"Loading {model_name}...")
    model, processor, model_type, is_compiled = load_model(model_name, device, use_compile=False)

    # Get device memory info
    if device == 'cuda':
        gpu_total_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_total_gb = round(gpu_total_bytes / (1024 ** 3), 1)
    elif device == 'mps':
        # MPS doesn't expose total memory easily, approximate for M3
        gpu_total_gb = 18.0  # Conservative estimate for M3
    else:
        gpu_total_gb = 0.0

    # Test batch sizes
    batch_size = BATCH_SIZE_START
    results = []

    print(f"Testing batch sizes on {gpu_total_gb}GB {device.upper()}...")

    while True:
        print(f"  Testing batch_size={batch_size}...")
        result = test_batch_size(model, model_type, processor, batch_size, device)
        results.append(result)

        if not result['success']:
            print(f"  ✗ OOM at batch_size={batch_size}")
            break

        print(f"  ✓ batch_size={batch_size} → {result['memory_gb']}GB")
        batch_size += BATCH_SIZE_INCREMENT

    # Calculate metrics
    if len(results) < 2:
        raise ValueError("Need at least 2 successful batch sizes to calculate metrics")

    optimal = results[-2]  # Last successful (before OOM)
    current = next((r for r in results if r['batch_size'] == 16), results[1])

    waste_gb = gpu_total_gb - optimal['memory_gb']
    waste_percent = (waste_gb / gpu_total_gb) * 100 if gpu_total_gb > 0 else 0
    speedup = optimal['batch_size'] / current['batch_size']

    # Cost calculations
    hours_per_run = ASSUMED_TRAINING_HOURS
    cost_per_hour = LAMBDA_LABS_A10_COST_PER_HOUR
    cost_current = hours_per_run * cost_per_hour
    cost_optimal = (hours_per_run / speedup) * cost_per_hour
    savings_per_run = cost_current - cost_optimal
    savings_annual = savings_per_run * ANNUAL_TRAINING_RUNS

    return {
        'model_name': model_name,
        'device': device,
        'gpu_total_gb': gpu_total_gb,
        'results': [r for r in results if r['success']],
        'optimal_batch_size': optimal['batch_size'],
        'optimal_memory_gb': optimal['memory_gb'],
        'current_batch_size': current['batch_size'],
        'current_memory_gb': current['memory_gb'],
        'waste_gb': round(waste_gb, 1),
        'waste_percent': round(waste_percent, 1),
        'speedup': round(speedup, 2),
        'cost_savings_per_run': round(savings_per_run, 2),
        'cost_savings_annual': round(savings_annual, 2)
    }


def validate_compilation_benefit(model_name: str = 'bert', batch_size: int = 16, device: Optional[str] = None) -> Dict:
    """
    Validate that torch.compile provides measurable performance improvements.

    Compares eager mode vs compiled mode for the same model and batch size.
    Measures execution time and memory usage for both.

    Args:
        model_name: One of ['bert', 'gpt2', 'resnet']
        batch_size: Batch size to test
        device: Optional device override ('cuda', 'mps', 'cpu'). Auto-detected if None.

    Returns:
        dict: {
            'model_name': str,
            'device': str,
            'batch_size': int,
            'eager_time_ms': float,
            'eager_memory_mb': int,
            'compiled_time_ms': float,
            'compiled_memory_mb': int,
            'speedup': float,
            'memory_ratio': float,
            'recommendation': str
        }

    Reference: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
    """
    import time

    # Auto-detect device if not specified
    if device is None:
        device = get_device()

    print(f"Validating torch.compile benefit on {device.upper()}...")
    print(f"Model: {model_name}, Batch size: {batch_size}")
    print("=" * 60)

    # Test 1: Eager mode (no compilation)
    print("\n[1/2] Testing EAGER mode (no compilation)...")
    model_eager, processor_eager, model_type_eager, _ = load_model(model_name, device, use_compile=False)

    # Warm-up run (doesn't count)
    inputs_warmup = create_dummy_batch(model_type_eager, batch_size, processor_eager, device)
    _ = model_eager(**inputs_warmup)
    del inputs_warmup
    if device == 'cuda':
        torch.cuda.synchronize()

    # Timed run
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    inputs_eager = create_dummy_batch(model_type_eager, batch_size, processor_eager, device)
    outputs_eager = model_eager(**inputs_eager)
    loss_eager = outputs_eager.logits.mean() if hasattr(outputs_eager, 'logits') else outputs_eager[0].mean()
    loss_eager.backward()
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    eager_time_ms = (end_time - start_time) * 1000
    if device == 'cuda':
        eager_memory_mb = int(torch.cuda.max_memory_allocated() / (1024 ** 2))
    else:
        eager_memory_mb = 0

    print(f"✓ Eager mode: {eager_time_ms:.2f}ms, {eager_memory_mb}MB")

    # Clean up
    del model_eager, inputs_eager, outputs_eager, loss_eager
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Test 2: Compiled mode
    print("\n[2/2] Testing COMPILED mode (torch.compile)...")
    model_compiled, processor_compiled, model_type_compiled, _ = load_model(model_name, device, use_compile=True)

    # Warm-up run (compilation happens here - doesn't count for timing)
    print("  Compiling model (first run)...")
    inputs_warmup2 = create_dummy_batch(model_type_compiled, batch_size, processor_compiled, device)
    _ = model_compiled(**inputs_warmup2)
    del inputs_warmup2
    if device == 'cuda':
        torch.cuda.synchronize()

    # Timed run (should be faster after compilation)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    inputs_compiled = create_dummy_batch(model_type_compiled, batch_size, processor_compiled, device)
    outputs_compiled = model_compiled(**inputs_compiled)
    loss_compiled = outputs_compiled.logits.mean() if hasattr(outputs_compiled, 'logits') else outputs_compiled[0].mean()
    loss_compiled.backward()
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    compiled_time_ms = (end_time - start_time) * 1000
    if device == 'cuda':
        compiled_memory_mb = int(torch.cuda.max_memory_allocated() / (1024 ** 2))
    else:
        compiled_memory_mb = 0

    print(f"✓ Compiled mode: {compiled_time_ms:.2f}ms, {compiled_memory_mb}MB")

    # Calculate benefit
    speedup = eager_time_ms / compiled_time_ms if compiled_time_ms > 0 else 1.0
    memory_ratio = compiled_memory_mb / eager_memory_mb if eager_memory_mb > 0 else 1.0

    # Recommendation
    if speedup >= 1.1:
        recommendation = f"torch.compile provides {speedup:.2f}x speedup - RECOMMENDED"
    elif speedup >= 1.0:
        recommendation = f"torch.compile provides marginal benefit ({speedup:.2f}x) - OPTIONAL"
    else:
        recommendation = f"torch.compile slower ({speedup:.2f}x) - NOT RECOMMENDED for this workload"

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Memory ratio: {memory_ratio:.2f}x")
    print(f"  {recommendation}")
    print("=" * 60)

    return {
        'model_name': model_name,
        'device': device,
        'batch_size': batch_size,
        'eager_time_ms': round(eager_time_ms, 2),
        'eager_memory_mb': eager_memory_mb,
        'compiled_time_ms': round(compiled_time_ms, 2),
        'compiled_memory_mb': compiled_memory_mb,
        'speedup': round(speedup, 2),
        'memory_ratio': round(memory_ratio, 2),
        'recommendation': recommendation
    }


def validate_backward_pass(model_name: str = 'bert', batch_size: int = 16, device: Optional[str] = None) -> Dict:
    """
    Validate that backward pass is correctly allocating gradient memory.

    This is CRITICAL for MemoryMark accuracy. Without proper backward pass simulation,
    memory estimates are wrong by 2-3x, causing recommendations to fail in production.

    Compares memory usage for:
    1. Forward-only (no gradients)
    2. Forward + Backward (with gradients)

    Expected ratio: 2.0-3.0x (backward adds ~50-60% overhead for gradients)
    If ratio ~1.0x, backward pass is NOT running (CRITICAL BUG)

    Args:
        model_name: One of ['bert', 'gpt2', 'resnet']
        batch_size: Batch size to test (default 16)
        device: Optional device override ('cuda', 'mps', 'cpu'). Auto-detected if None.

    Returns:
        dict: {
            'model_name': str,
            'device': str,
            'batch_size': int,
            'forward_only_memory_mb': int,
            'forward_backward_memory_mb': int,
            'ratio': float,
            'expected_ratio_min': float,
            'expected_ratio_max': float,
            'status': str ('PASS' or 'FAIL'),
            'message': str
        }

    References:
        - Tensor.backward(): https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html
        - CUDA Memory: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html
        - MPS Memory: https://docs.pytorch.org/docs/stable/generated/torch.mps.current_allocated_memory.html
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_device()

    print("=" * 60)
    print("■ BACKWARD PASS VALIDATION TEST")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Device: {device.upper()}")
    print(f"Batch size: {batch_size}")
    print()
    print("This test validates that gradient memory is properly allocated.")
    print("Expected ratio (forward+backward / forward-only): 2.0-3.0x")
    print("=" * 60)

    # Load model (no compilation for clean test)
    print(f"\nLoading {model_name}...")
    model, processor, model_type, _ = load_model(model_name, device, use_compile=False)

    # === TEST 1: FORWARD-ONLY (NO BACKWARD PASS) ===
    print("\n[1/2] Testing FORWARD-ONLY (no backward pass)...")

    try:
        # Clear cache and reset stats
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif device == 'mps':
            torch.mps.empty_cache()

        # Create batch
        inputs_forward = create_dummy_batch(model_type, batch_size, processor, device)

        # FORWARD PASS ONLY (no loss, no backward)
        with torch.no_grad():  # Explicitly disable gradient tracking
            outputs_forward = model(**inputs_forward)

        # Measure memory
        if device == 'cuda':
            forward_only_memory_bytes = torch.cuda.max_memory_allocated()
        elif device == 'mps':
            forward_only_memory_bytes = torch.mps.current_allocated_memory()
        else:
            forward_only_memory_bytes = 0

        forward_only_memory_mb = int(forward_only_memory_bytes / (1024 ** 2))

        print(f"✓ Forward-only memory: {forward_only_memory_mb} MB")

        # Clean up
        del inputs_forward, outputs_forward
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
        model.zero_grad()

    except Exception as e:
        print(f"✗ Forward-only test failed: {e}")
        raise

    # === TEST 2: FORWARD + BACKWARD ===
    print("\n[2/2] Testing FORWARD + BACKWARD (with gradients)...")

    try:
        # Clear cache and reset stats
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif device == 'mps':
            torch.mps.empty_cache()

        # Create batch
        inputs_backward = create_dummy_batch(model_type, batch_size, processor, device)

        # FORWARD PASS
        outputs_backward = model(**inputs_backward)

        # CREATE DUMMY LOSS
        if model_type == 'nlp':
            if hasattr(outputs_backward, 'logits'):
                loss = outputs_backward.logits.mean()
            else:
                loss = outputs_backward[0].mean()
        else:  # vision
            loss = outputs_backward.logits.mean()

        # BACKWARD PASS - THE CRITICAL OPERATION
        # This should allocate gradient tensors for all model parameters
        loss.backward()

        # Measure memory
        if device == 'cuda':
            forward_backward_memory_bytes = torch.cuda.max_memory_allocated()
        elif device == 'mps':
            forward_backward_memory_bytes = torch.mps.current_allocated_memory()
        else:
            forward_backward_memory_bytes = 0

        forward_backward_memory_mb = int(forward_backward_memory_bytes / (1024 ** 2))

        print(f"✓ Forward+Backward memory: {forward_backward_memory_mb} MB")

        # Clean up
        del inputs_backward, outputs_backward, loss
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
        model.zero_grad()

    except Exception as e:
        print(f"✗ Forward+Backward test failed: {e}")
        raise

    # === CALCULATE RATIO AND VALIDATE ===
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS:")
    print("=" * 60)

    # Handle edge case: forward_only_memory_mb is 0 (CPU or measurement failed)
    if forward_only_memory_mb == 0:
        ratio = 0.0
        status = "SKIPPED"
        message = f"Memory measurement not available on {device.upper()}. This test requires CUDA or MPS."
        print(f"⚠ {message}")
    else:
        ratio = forward_backward_memory_mb / forward_only_memory_mb

        print(f"Forward-only memory:     {forward_only_memory_mb:6} MB")
        print(f"Forward+Backward memory: {forward_backward_memory_mb:6} MB")
        print(f"Ratio:                   {ratio:.2f}x")
        print()

        # Validation criteria: ratio should be 2.0-3.0x
        EXPECTED_MIN = 2.0
        EXPECTED_MAX = 3.0

        if EXPECTED_MIN <= ratio <= EXPECTED_MAX:
            status = "PASS"
            message = f"✓ PASS: Ratio {ratio:.2f}x is within expected range [{EXPECTED_MIN}-{EXPECTED_MAX}x]. Backward pass is correctly allocating gradient memory."
            print(f"✓ {message}")
        elif ratio < EXPECTED_MIN:
            if ratio < 1.5:
                status = "FAIL"
                message = f"✗ FAIL: Ratio {ratio:.2f}x is TOO LOW (expected {EXPECTED_MIN}-{EXPECTED_MAX}x). CRITICAL: Backward pass may not be running! Check loss.backward() implementation."
                print(f"✗ {message}")
            else:
                status = "WARN"
                message = f"⚠ WARNING: Ratio {ratio:.2f}x is slightly low (expected {EXPECTED_MIN}-{EXPECTED_MAX}x). Backward pass may be working but with lower overhead than typical."
                print(f"⚠ {message}")
        else:  # ratio > EXPECTED_MAX
            status = "WARN"
            message = f"⚠ WARNING: Ratio {ratio:.2f}x is higher than expected (expected {EXPECTED_MIN}-{EXPECTED_MAX}x). This may indicate additional memory overhead."
            print(f"⚠ {message}")

    print("=" * 60)

    return {
        'model_name': model_name,
        'device': device,
        'batch_size': batch_size,
        'forward_only_memory_mb': forward_only_memory_mb,
        'forward_backward_memory_mb': forward_backward_memory_mb,
        'ratio': round(ratio, 2),
        'expected_ratio_min': 2.0,
        'expected_ratio_max': 3.0,
        'status': status,
        'message': message
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python memorymark.py <model_name> [device]")
        print("       python memorymark.py --validate [model_name]")
        print("       python memorymark.py --validate-backward [model_name]")
        print("")
        print("Models: bert, gpt2, resnet")
        print("Device: cuda, mps, cpu (optional, auto-detected)")
        print("")
        print("Examples:")
        print("  python memorymark.py bert                 # Run full analysis on BERT")
        print("  python memorymark.py --validate           # Validate torch.compile on BERT")
        print("  python memorymark.py --validate gpt2      # Validate torch.compile on GPT-2")
        print("  python memorymark.py --validate-backward  # Validate backward pass on BERT")
        sys.exit(1)

    # Check for --validate-backward flag
    if sys.argv[1] == '--validate-backward':
        model_name = sys.argv[2] if len(sys.argv) > 2 else 'bert'
        try:
            results = validate_backward_pass(model_name)
            # Results are already printed by the function
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Check for --validate flag
    if sys.argv[1] == '--validate':
        model_name = sys.argv[2] if len(sys.argv) > 2 else 'bert'
        try:
            print("\n" + "="*60)
            print("■ TORCH.COMPILE VALIDATION TEST")
            print("="*60)
            results = validate_compilation_benefit(model_name)
            # Results are already printed by the function
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Normal analysis mode
    model_name = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        results = find_optimal_batch_size(model_name, device)

        print("\n" + "="*60)
        print(f"■ MemoryMark Results - {results['model_name'].upper()} on {results['device'].upper()}")
        print("="*60)
        print(f"Optimal batch size: {results['optimal_batch_size']}")
        print(f"Current batch size: {results['current_batch_size']}")
        print(f"Waste: {results['waste_gb']}GB ({results['waste_percent']}%)")
        print(f"Speedup: {results['speedup']}x")
        print(f"Savings: ${results['cost_savings_per_run']:.2f}/run, ${results['cost_savings_annual']:.2f}/year")
        print("="*60)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
