import argparse
import os
import torch
import torch.onnx
from models.mobilenetv3 import get_cloud_model

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX for Jetson Nano (TensorRT) deployment")
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='Path to trained model')
    parser.add_argument('--out_onnx', type=str, default='checkpoints/cloud_model.onnx', help='Output ONNX file path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference (usually 1 for edge devices)')
    parser.add_argument('--channels', type=int, default=4, choices=[3, 4], help='Number of input channels')
    parser.add_argument('--patch_size', type=int, default=256, help='Input crop/window size')
    parser.add_argument('--dynamic_batch', action='store_true', help='Export ONNX with a dynamic batch dimension')
    parser.add_argument('--allow_untrained_export', action='store_true', help='Export an untrained model structure when checkpoint is missing')
    args = parser.parse_args()

    if args.patch_size <= 0:
        raise ValueError('patch_size must be greater than zero')
    if args.batch_size <= 0:
        raise ValueError('batch_size must be greater than zero')

    # Load model
    print("Loading PyTorch model...")
    model = get_cloud_model(pretrained=False, num_channels=args.channels)

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        print(f"Weights loaded from {args.model_path}")
    elif args.allow_untrained_export:
        print(f"Warning: {args.model_path} not found. Exporting untrained model structure.")
    else:
        raise FileNotFoundError(
            f"Model checkpoint not found: {args.model_path}. "
            "Use --allow_untrained_export only when exporting a structure test model."
        )
    
    model.eval()

    # Create dummy input with exact shape expected by the model
    # Jetson Nano usually processes 1 patch at a time, or small batches
    dummy_input = torch.randn(args.batch_size, args.channels, args.patch_size, args.patch_size, device='cpu')

    # Export to ONNX
    print(f"Exporting to ONNX: {args.out_onnx}...")
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} if args.dynamic_batch else None
    torch.onnx.export(
        model, 
        dummy_input, 
        args.out_onnx, 
        export_params=True,
        opset_version=11, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    print("Export successful!")
    print("\n--- Next Steps for Jetson Nano ---")
    print("1. Copy the .onnx file to your Jetson Nano.")
    print("2. Convert ONNX to TensorRT engine using trtexec:")
    if args.dynamic_batch:
        print(f"   trtexec --onnx={args.out_onnx} --saveEngine=cloud_model.engine --fp16 --minShapes=input:1x{args.channels}x{args.patch_size}x{args.patch_size} --optShapes=input:{args.batch_size}x{args.channels}x{args.patch_size}x{args.patch_size} --maxShapes=input:{args.batch_size}x{args.channels}x{args.patch_size}x{args.patch_size}")
    else:
        print(f"   trtexec --onnx={args.out_onnx} --saveEngine=cloud_model.engine --fp16")
    print("   (Using --fp16 gives a huge speed boost on Jetson Nano's Maxwell GPU)")
    print("3. Use the TensorRT Python API (tensorrt module) to run inference on the .engine file.")

if __name__ == '__main__':
    main()
