#!/usr/bin/env python3
"""
ONNX Model INT8 Dynamic Quantization Script
-------------------------------------------
This script performs Dynamic INT8 Quantization on any FP32 ONNX model 
to reduce its RAM footprint and accelerate CPU inference.

Usage:
    python quantize_onnx_int8.py --input <path_to_model.onnx> --output <path_to_quantized_model.onnx>
"""

import os
import sys
import argparse

try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("Error: 'onnxruntime' package is required. Please install it using:")
    print("  pip install onnxruntime onnx")
    sys.exit(1)


def quantize_onnx_model(input_model_path: str, output_model_path: str):
    """
    Applies dynamic quantization to weights to convert them to signed INT8,
    keeping activations as FLOAT32. Highly efficient for LSTMs and RNNs,
    and generally reduces weight file size by 4x.
    """
    if not os.path.exists(input_model_path):
        print(f"[-] Error: Input model file '{input_model_path}' not found.")
        sys.exit(1)

    print(f"[+] Loading FP32 ONNX model from: {input_model_path}")
    print(f"[+] Quantization Type: Dynamic INT8 (QuantType.QInt8)")
    
    try:
        # Perform dynamic weight quantization
        # Dynamic quantization is a great zero-data calibration method to reduce RAM
        quantize_dynamic(
            model_input=input_model_path,
            model_output=output_model_path,
            weight_type=QuantType.QInt8
        )
        
        orig_size = os.path.getsize(input_model_path) / (1024 * 1024)
        quant_size = os.path.getsize(output_model_path) / (1024 * 1024)
        
        print(f"[+] Quantization successful!")
        print(f"  - Original Model Size : {orig_size:.2f} MB")
        print(f"  - Quantized Model Size: {quant_size:.2f} MB (~{orig_size/quant_size:.1f}x compression)")
        print(f"  - Saved to            : {output_model_path}")
        
        print("\n[!] IMPORTANT AI ENGINEER NOTE FOR PRODUCTION DEPLOYMENT:")
        print("  - Dynamic Quantization converts convolution/linear weights to INT8 and does activations at runtime.")
        print("  - For CNN-heavy models (e.g. PP-OCRv5 Mobile Detection), dynamic quantization creates ConvInteger nodes.")
        print("  - Standard ONNX Runtime CPU Execution Provider might not have highly-optimized kernels for ConvInteger nodes,")
        print("    which could lead to unsupported op errors (e.g. 'Could not find an implementation for ConvInteger') or slow performance.")
        print("  - For CNN models on CPU, using FP32/FP16 or applying Static Quantization (QLinearConv) is recommended instead.")
        
    except Exception as e:
        print(f"[-] Quantization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ONNX models to INT8 dynamically.")
    parser.add_argument("-i", "--input", required=True, help="Path to input FP32 ONNX model")
    parser.add_argument("-o", "--output", required=True, help="Path to output quantized INT8 ONNX model")
    
    args = parser.parse_args()
    quantize_onnx_model(args.input, args.output)
