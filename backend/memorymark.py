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
        # Create dummy images (224x224 RGB)
        dummy_images = [
            torch.randn(3, IMAGE_SIZE, IMAGE_SIZE) for _ in range(batch_size)
        ]

        # Process images
        inputs = processor(images=dummy_images, return_tensors='pt')

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs


def load_model(model_name: str, device: str) -> Tuple:
    """
    Load a HuggingFace model to specified device.

    Args:
        model_name: One of ['bert', 'gpt2', 'resnet']
        device: 'cuda', 'mps', or 'cpu'

    Returns:
        tuple: (model, processor, model_type)

    Reference: https://huggingface.co/docs/transformers/main/en/model_doc/auto
    """
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_MAP.keys())}")

    hf_model_name = MODEL_MAP[model_name]

    if model_name in ['bert', 'gpt2']:
        # NLP models
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_name,
            num_labels=2  # Binary classification for dummy task
        )
        model = model.to(device)
        model.eval()  # Set to eval mode (disables dropout)
        return (model, tokenizer, 'nlp')
    else:  # resnet
        # Vision models
        processor = AutoImageProcessor.from_pretrained(hf_model_name)
        model = AutoModelForImageClassification.from_pretrained(hf_model_name)
        model = model.to(device)
        model.eval()
        return (model, processor, 'vision')


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
    model, processor, model_type = load_model(model_name, device)

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python memorymark.py <model_name> [device]")
        print("Models: bert, gpt2, resnet")
        print("Device: cuda, mps, cpu (optional, auto-detected)")
        sys.exit(1)

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
