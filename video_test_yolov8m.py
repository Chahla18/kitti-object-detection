import os
import argparse
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import numpy as np
from tqdm import tqdm
import yaml
import random

# Your custom class names
CUSTOM_CLASS_NAMES = [
    "Car",
    "Cyclist", 
    "Misc",
    "Pedestrian",
    "Person_sitting",
    "Tram",
    "Truck",
    "Van",
]

# Define a consistent color for each class (B,G,R format)
CLASS_COLORS = {
    "Car": (0, 0, 255),           # Red
    "Cyclist": (0, 255, 255),      # Yellow
    "Misc": (128, 0, 128),        # Purple
    "Pedestrian": (0, 255, 0),     # Green
    "Person_sitting": (0, 165, 255), # Orange
    "Tram": (255, 0, 0),          # Blue
    "Truck": (255, 0, 255),       # Magenta
    "Van": (255, 255, 0),         # Cyan
}

def create_data_yaml(custom_classes):
    """Create a YAML file with custom class configuration"""
    try:
        yaml_content = {
            "path": "./datasets/kitti",
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(custom_classes),
            "names": custom_classes
        }
        
        yaml_path = "custom_kitti.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"Created data YAML at {yaml_path}")
        return yaml_path
    except Exception as e:
        print(f"Error creating YAML: {e}")
        return None

def download_model_from_hf(repo_id, model_file=None, local_dir="model"):
    """Download the model from Hugging Face Hub"""
    try:
        print(f"Listing available files in {repo_id}...")
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo_id)
        
        # Filter for PyTorch model files
        pt_files = [f for f in files if f.endswith('.pt')]
        
        if not pt_files:
            print(f"No .pt files found in the repository {repo_id}")
            return None
        
        print(f"Found {len(pt_files)} model files:")
        for i, file in enumerate(pt_files):
            print(f"  {i+1}. {file}")
        
        # If a specific model file was requested
        if model_file and model_file in pt_files:
            selected_file = model_file
            print(f"Using requested model file: {selected_file}")
        else:
            # Look for checkpoint files with epoch information
            checkpoint_files = [f for f in pt_files if "_epoch_" in f]
            
            if checkpoint_files:
                # Try to get the checkpoint with the highest epoch number
                def extract_epoch(filename):
                    try:
                        parts = filename.split("_epoch_")
                        if len(parts) > 1:
                            epoch_num = int(parts[1].replace(".pt", ""))
                            return epoch_num
                        return 0
                    except:
                        return 0
                
                # Sort by epoch number in descending order
                checkpoint_files.sort(key=extract_epoch, reverse=True)
                selected_file = checkpoint_files[0]  # This is the file with highest epoch
                print(f"Using latest checkpoint: {selected_file} (Epoch {extract_epoch(selected_file)})")
            else:
                # If no checkpoint files, use whatever is available
                selected_file = pt_files[0]
                print(f"No checkpoint files found. Using first available model: {selected_file}")
        
        # Download the selected model file
        print(f"Downloading {selected_file}...")
        os.makedirs(local_dir, exist_ok=True)
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=selected_file,
            local_dir=local_dir
        )
        print(f"Downloaded model to {model_path}")
        return model_path
    
    except Exception as e:
        print(f"Error during model download: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_video_with_color_coding(model, video_path, output_path="output_color_coded.mp4", conf_threshold=0.25):
    """Process video with color-coded bounding boxes for each class"""
    try:
        # Define mapping from COCO classes to our custom classes
        coco_to_custom = {
            2: 0,   # COCO car -> our Car
            1: 1,   # COCO bicycle -> our Cyclist
            7: 6,   # COCO truck -> our Truck
            3: 1,   # COCO motorcycle -> could also be mapped to Cyclist
            5: 0,   # COCO bus -> our car (or could be mapped differently)
            0: 3,   # COCO person -> our Pedestrian
            5: 5,   # COCO bus -> could be mapped to Tram
            7: 6,   # COCO truck -> our Truck
            8: 7,   # COCO boat -> could be mapped to Van
        }
        
        # Create a reverse mapping of indices to our class names
        idx_to_name = {i: name for i, name in enumerate(CUSTOM_CLASS_NAMES)}
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process video frame by frame
        frame_count = 0
        detection_counts = {name: 0 for name in CUSTOM_CLASS_NAMES}
        
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference with the model
                results = model(frame, conf=conf_threshold)
                
                # Get the detection results
                result = results[0]
                
                # Process and map detections to our custom classes
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    # Create a clean copy of the frame for drawing
                    annotated_frame = frame.copy()
                    
                    # Track what classes were detected in this frame
                    frame_detections = {}
                    
                    # Process each detection and map to custom classes
                    for box in boxes:
                        # Get the COCO class ID
                        coco_cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                        
                        # Only process if this COCO class has a mapping to our custom classes
                        if coco_cls_id in coco_to_custom:
                            # Map to our custom class
                            custom_cls_id = coco_to_custom[coco_cls_id]
                            custom_cls_name = idx_to_name[custom_cls_id]
                            
                            # Get bounding box, confidence
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = box.conf[0].item() if box.conf is not None else 0
                            
                            # Get color for this class
                            color = CLASS_COLORS.get(custom_cls_name, (0, 255, 0))
                            
                            # Draw bounding box with class-specific color
                            cv2.rectangle(annotated_frame, 
                                          (int(x1), int(y1)), 
                                          (int(x2), int(y2)), 
                                          color, 2)
                            
                            # Create a background for the text
                            text_size = cv2.getTextSize(f"{custom_cls_name} {conf:.2f}", 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 
                                                       0.5, 2)[0]
                            cv2.rectangle(annotated_frame, 
                                         (int(x1), int(y1)-20), 
                                         (int(x1) + text_size[0], int(y1)), 
                                         color, -1)
                            
                            # Add label with custom class name and confidence
                            cv2.putText(annotated_frame, 
                                       f"{custom_cls_name} {conf:.2f}", 
                                       (int(x1), int(y1)-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                       (255, 255, 255), 2)
                            
                            # Track this detection
                            frame_detections[custom_cls_name] = frame_detections.get(custom_cls_name, 0) + 1
                            detection_counts[custom_cls_name] += 1
                    
                    # Show detection counts for this frame
                    if frame_count % 20 == 0 and frame_detections:
                        print(f"Frame {frame_count}: " + 
                              ", ".join([f"{count} {name}" for name, count in frame_detections.items()]))
                
                else:
                    # If no detections or no mapping to custom classes, use the original frame
                    annotated_frame = frame.copy()
                
                # Add legend to the frame
                legend_y = 30
                for i, (class_name, color) in enumerate(CLASS_COLORS.items()):
                    if i < 4:  # First row of legend
                        legend_x = 10 + i * 160
                        y_pos = 30
                    else:      # Second row of legend
                        legend_x = 10 + (i-4) * 160
                        y_pos = 60
                    
                    # Draw colored rectangle and class name
                    cv2.rectangle(annotated_frame, (legend_x, y_pos-15), (legend_x+20, y_pos+5), color, -1)
                    cv2.putText(annotated_frame, class_name, (legend_x+25, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Write the frame to output video
                out.write(annotated_frame)
                
                frame_count += 1
                pbar.update(1)
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Print detection statistics
        print("\nDetection Statistics:")
        for name, count in detection_counts.items():
            print(f"  {name}: {count} detections")
        
        print(f"\nProcessed {frame_count} frames. Output saved to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Process video with color-coded bounding boxes")
    parser.add_argument("--video_path", type=str, required=True, 
                        help="Path to local video")
    parser.add_argument("--base_model", type=str, default="/home/onyxia/work/yolov8m.pt",
                        help="Path to base YOLOv8 model")
    parser.add_argument("--output_path", type=str, help="Output video path", 
                        default="output_color_coded.mp4")
    parser.add_argument("--conf_threshold", type=float, help="Confidence threshold", 
                        default=0.25)
    
    args = parser.parse_args()
    
    # Create custom data YAML
    data_yaml_path = create_data_yaml(CUSTOM_CLASS_NAMES)
    if not data_yaml_path:
        print("Failed to create data YAML. Exiting.")
        return
    
    # Load the base model
    print(f"Loading YOLOv8 model from {args.base_model}...")
    model = YOLO(args.base_model)
    
    # Process the video with color-coded bounding boxes
    output_video = process_video_with_color_coding(
        model=model,
        video_path=args.video_path,
        output_path=args.output_path,
        conf_threshold=args.conf_threshold
    )
    
    if output_video:
        print(f"Successfully processed video. Output saved to: {output_video}")
    else:
        print("Failed to process the video.")

if __name__ == "__main__":
    main()