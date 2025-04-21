import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Kitti
from PIL import Image
import numpy as np
import yaml
import uuid
import glob
import requests
import shutil
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import wandb
from huggingface_hub import HfApi, create_repo, login, upload_folder
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter

kitti_dir = "/home/onyxia/work/datasets"

yolo_transform = transforms.Compose(
    [
        transforms.Resize((640, 640)),  # YOLOv8 default size
        transforms.ToTensor(),  # Converts to [0,1] range
    ]
)

# Download the dataset with YOLOv8 preprocessing
try:
    print("Downloading training data with YOLOv8 preprocessing...")
    kitti_train = Kitti(
        root=kitti_dir, train=True, download=True, transform=yolo_transform
    )

    print(f"Successfully downloaded KITTI dataset with YOLOv8 preprocessing!")
    print(f"Training samples: {len(kitti_train)}")

    # Define a custom collate function to handle variable-sized targets
    def collate_fn(batch):
        images, targets = zip(*batch)
        # Stack images into a single tensor
        images = torch.stack(images)
        # Keep targets as a list (don't try to stack them)
        return images, targets

    # Create DataLoaders with the custom collate function
    train_loader = DataLoader(
        kitti_train,
        batch_size=16,  # YOLOv8 typical batch size
        shuffle=True,
        num_workers=2,  # For parallel preprocessing
        collate_fn=collate_fn,  # Add this to handle variable-sized annotations
    )

    # Verify we can get a sample
    sample_img, sample_targets = next(iter(train_loader))
    print("\nSuccessfully loaded a sample batch from the dataset")
    print(f"Batch shape: {sample_img.shape}")  # Should be [batch_size, 3, 640, 640]
    print(
        f"Pixel value range: [{sample_img.min():.2f}, {sample_img.max():.2f}]"
    )  # Should be [0, 1]
    print(f"Number of targets in first image: {len(sample_targets[0])}")

except Exception as e:
    print(f"Error: {e}")
    print(
        "If the automatic download fails, you may need to register on the KITTI website and download manually."
    )


# Authentication tokens
HF_TOKEN = "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"  # Hugging Face token
WANDB_API_KEY = "dda7d259bece87388377901ab094ac808377eda3"  # W&B API key
HF_USERNAME = "chahla"  # Username on Hugging Face

experiment_config = {
    # Model and technique selection
    "model_type": "yolov8",
    "model_size": "m",  # n, s, m, l, or x
    "techniques": ["focal_loss", "rotation_aug"],
    # Training parameters
    "task": "detect",
    "epochs": 50,
    "batch_size": 32,
    "img_size": 640,
    "optimizer": "Adam",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 0.05,
    "cls": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 45.0,  # Rotational augmentation for cars
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    # Dataset and output settings
    "kitti_dir": "data/Kitti",
    "output_dir": "runs/detect",
    "checkpoint_dir": "checkpoints",
    "save_period": 1,  # Save and upload checkpoint every X epochs
}

# Generate unique identifiers
experiment_id = str(uuid.uuid4())[:8]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_huggingface():
    """Set up Hugging Face authentication and create repository"""

    # Login to Hugging Face
    print("Logging in to Hugging Face...")
    login(token=HF_TOKEN, add_to_git_credential=True)

    # Generate repository name based on experiment configuration
    model_name = f"{experiment_config['model_type']}-{experiment_config['model_size']}"
    techniques = "-".join(experiment_config["techniques"])
    repo_name = f"kitti-{model_name}-{techniques}"

    # Full repository name including username
    full_repo_name = f"{HF_USERNAME}/{repo_name}"

    # Check if repository already exists
    api = HfApi()
    try:
        api.model_info(repo_id=full_repo_name)
        print(f"Repository {full_repo_name} already exists")
    except Exception:
        # Create repository if it doesn't exist
        print(f"Creating new repository: {full_repo_name}")
        create_repo(
            repo_id=repo_name,
            token=HF_TOKEN,
            private=False,
            repo_type="model",
            exist_ok=True,
        )

    # Store repository name in experiment config
    experiment_config["hf_repo_name"] = full_repo_name
    print(f"Hugging Face setup complete. Repository: {full_repo_name}")

    return full_repo_name


def setup_wandb():
    """Set up W&B authentication and project"""

    # Login to W&B
    print("Logging in to Weights & Biases...")
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

    # Set up W&B project name
    project_name = "kitti-object-detection"

    # Check if project exists by making API request
    headers = {"Authorization": f"Bearer {WANDB_API_KEY}"}
    response = requests.get(
        f"https://api.wandb.ai/graphql",
        json={"query": f'{{ project(name: "{project_name}") {{ name }} }}'},
        headers=headers,
    )

    project_exists = False
    if response.status_code == 200:
        data = response.json()
        if data.get("data", {}).get("project"):
            project_exists = True

    if project_exists:
        print(f"W&B project '{project_name}' already exists")
    else:
        print(f"W&B project '{project_name}' will be created on first run")

    # Create a unique run name
    techniques_str = "-".join(experiment_config["techniques"])
    run_name = f"{experiment_config['model_type']}-{experiment_config['model_size']}-{techniques_str}_{timestamp}_{experiment_id}"

    # Store W&B info in experiment config
    experiment_config["wandb_project"] = project_name
    experiment_config["wandb_run_name"] = run_name

    print(f"W&B setup complete. Project: {project_name}, Run: {run_name}")

    return project_name, run_name


def setup_experiment():
    """Set up the entire experiment"""
    print("Setting up experiment...")

    # Create necessary directories
    os.makedirs(experiment_config["output_dir"], exist_ok=True)
    os.makedirs(experiment_config["checkpoint_dir"], exist_ok=True)

    # Set up Hugging Face
    hf_repo_name = setup_huggingface()

    # Set up W&B
    wandb_project, wandb_run_name = setup_wandb()

    # Create descriptive experiment name and description
    model_name = f"{experiment_config['model_type']}-{experiment_config['model_size']}"
    techniques = ", ".join(experiment_config["techniques"])

    experiment_config["experiment_name"] = f"{model_name}_{timestamp}_{experiment_id}"
    experiment_config["experiment_desc"] = (
        f"{model_name.upper()} with {techniques} on KITTI dataset"
    )

    print("\nExperiment setup complete!")
    print(f"Experiment name: {experiment_config['experiment_name']}")
    print(f"HF Repository: {hf_repo_name}")
    print(f"W&B Project: {wandb_project}")
    print(f"W&B Run name: {wandb_run_name}")

    # Save experiment config to YAML file for reference
    with open(f"experiment_config_{timestamp}.yaml", "w") as f:
        yaml.dump(experiment_config, f)

    return experiment_config


# Class mapping - defined at module level for accessibility
CLASS_MAPPING = {
    "Car": 0,
    "Cyclist": 1,
    "DontCare": 2,
    "Misc": 3,
    "Pedestrian": 4,
    "Person_sitting": 5,
    "Tram": 6,
    "Truck": 7,
    "Van": 8,
}


# Define worker functions at module level so they can be pickled
def analyze_label_file(basename, label_dir):
    """Analyze a single label file to count object classes"""
    label_file = os.path.join(label_dir, f"{basename}.txt")
    if not os.path.exists(label_file):
        return basename, {}

    class_counts = Counter()

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                class_name = parts[0]
                if class_name in CLASS_MAPPING:
                    class_counts[class_name] += 1

    return basename, dict(class_counts)


def process_file(data):
    """Process a single file (convert and copy)"""
    basename, src_dirs, dst_dirs, is_train = data

    src_img_dir, src_label_dir = src_dirs
    train_img_dst, train_label_dst, val_img_dst, val_label_dst = dst_dirs

    src_img = os.path.join(src_img_dir, f"{basename}.png")
    src_label = os.path.join(src_label_dir, f"{basename}.txt")

    if is_train:
        dst_img = os.path.join(train_img_dst, f"{basename}.png")
        dst_label = os.path.join(train_label_dst, f"{basename}.txt")
    else:
        dst_img = os.path.join(val_img_dst, f"{basename}.png")
        dst_label = os.path.join(val_label_dst, f"{basename}.txt")

    # Copy image
    shutil.copy(src_img, dst_img)

    if not os.path.exists(src_label):
        return 0

    # Get image dimensions for normalization
    with Image.open(src_img) as img:
        img_width, img_height = img.size

    conversion_count = 0

    # Convert KITTI format to YOLO format
    with open(src_label, "r") as f_in, open(dst_label, "w") as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            class_name = parts[0]
            if class_name not in CLASS_MAPPING:
                continue

            # Extract bounding box coordinates
            x1, y1, x2, y2 = (
                float(parts[4]),
                float(parts[5]),
                float(parts[6]),
                float(parts[7]),
            )

            # Convert to YOLO format
            class_id = CLASS_MAPPING[class_name]
            center_x = ((x1 + x2) / 2) / img_width
            center_y = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # Write YOLO format line
            f_out.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
            conversion_count += 1

    return conversion_count


def reorganize_kitti_dataset_optimized():
    """Reorganize KITTI dataset with optimized processing and class-balanced split"""

    # Base directories
    kitti_dir = "/home/onyxia/work/datasets/Kitti"
    yolo_dir = "/home/onyxia/work/datasets/Kitti_YOLO"

    # Source directories
    train_img_src = os.path.join(kitti_dir, "raw", "training", "image_2")
    train_label_src = os.path.join(kitti_dir, "raw", "training", "label_2")

    # Target directories
    train_img_dst = os.path.join(yolo_dir, "train", "images")
    train_label_dst = os.path.join(yolo_dir, "train", "labels")
    val_img_dst = os.path.join(yolo_dir, "val", "images")
    val_label_dst = os.path.join(yolo_dir, "val", "labels")

    # Check if dataset is already organized
    if (
        os.path.exists(train_img_dst)
        and os.path.exists(train_label_dst)
        and os.path.exists(val_img_dst)
        and os.path.exists(val_label_dst)
        and len(os.listdir(train_img_dst)) > 0
        and len(os.listdir(train_label_dst)) > 0
    ):
        print("Dataset already organized. Using existing structure.")
        return os.path.join(yolo_dir, "data.yaml")

    # Create directories
    os.makedirs(train_img_dst, exist_ok=True)
    os.makedirs(train_label_dst, exist_ok=True)
    os.makedirs(val_img_dst, exist_ok=True)
    os.makedirs(val_label_dst, exist_ok=True)

    # Get all image files
    all_img_files = glob.glob(os.path.join(train_img_src, "*.png"))
    all_basenames = [os.path.basename(f).replace(".png", "") for f in all_img_files]

    # Use parallel processing for class analysis
    print("Analyzing class distribution in parallel...")
    image_class_counts = {}
    num_workers = max(1, multiprocessing.cpu_count() - 1)

    # Prepare arguments for parallel processing
    analyze_args = [(basename, train_label_src) for basename in all_basenames]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(analyze_label_file, *args) for args in analyze_args]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Analyzing labels"
        ):
            basename, counts = future.result()
            image_class_counts[basename] = counts

    # Strategy for class-balanced split:
    # 1. Count minority classes
    minority_classes = ["Cyclist", "Person_sitting", "Tram"]

    def count_minority_classes(basename):
        counts = image_class_counts[basename]
        return sum(counts.get(cls, 0) for cls in minority_classes)

    sorted_images = sorted(
        image_class_counts.keys(), key=count_minority_classes, reverse=True
    )

    # 2. Ensure even distribution by taking every 5th image for validation
    train_images = []
    val_images = []

    for i, basename in enumerate(sorted_images):
        if i % 5 == 0:  # Every 5th image goes to validation (20%)
            val_images.append(basename)
        else:
            train_images.append(basename)

    print(
        f"Class-balanced split: {len(train_images)} training images, {len(val_images)} validation images"
    )

    # Source and destination directories to pass to process_file
    src_dirs = (train_img_src, train_label_src)
    dst_dirs = (train_img_dst, train_label_dst, val_img_dst, val_label_dst)

    # Process files in parallel
    print(f"Using {num_workers} CPU cores for parallel conversion")

    # Process training images
    print("Processing training images...")
    train_args = [(basename, src_dirs, dst_dirs, True) for basename in train_images]
    train_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file, arg) for arg in train_args]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Converting training data"
        ):
            train_count += future.result()

    # Process validation images
    print("Processing validation images...")
    val_args = [(basename, src_dirs, dst_dirs, False) for basename in val_images]
    val_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file, arg) for arg in val_args]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Converting validation data"
        ):
            val_count += future.result()

    # Create YAML file
    data_yaml = {
        "path": yolo_dir,
        "train": "train/images",
        "val": "val/images",
        "nc": 9,
        "names": [
            "Car",
            "Cyclist",
            "DontCare",
            "Misc",
            "Pedestrian",
            "Person_sitting",
            "Tram",
            "Truck",
            "Van",
        ],
    }

    yaml_path = os.path.join(yolo_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    # Print summary statistics
    print(f"\nDataset reorganized successfully!")
    print(f"Training images: {len(os.listdir(train_img_dst))}")
    print(f"Training labels: {len(os.listdir(train_label_dst))}")
    print(f"Validation images: {len(os.listdir(val_img_dst))}")
    print(f"Validation labels: {len(os.listdir(val_label_dst))}")
    print(f"Converted {train_count} object annotations for training")
    print(f"Converted {val_count} object annotations for validation")
    print(f"\nYAML file created at: {yaml_path}")

    return yaml_path


def upload_to_hf(local_folder, commit_message):
    """Upload content to Hugging Face Hub with correct authentication method"""
    try:
        # Create a new branch for each run to avoid conflicts
        branch = f"run-{experiment_id}"

        print(
            f" Attempting upload to Hugging Face repo: {experiment_config['hf_repo_name']}, branch: {branch}"
        )
        print(f" Local folder to upload: {local_folder}")

        # Check folder contents
        if os.path.exists(local_folder):
            files = os.listdir(local_folder)
            file_sizes = [
                os.path.getsize(os.path.join(local_folder, f)) / (1024 * 1024)
                for f in files
            ]
            print(f" Files found: {len(files)} files")
            for i, (file, size) in enumerate(zip(files, file_sizes)):
                print(f"   {i+1}. {file} - {size:.2f} MB")
        else:
            print(f" Error: Local folder {local_folder} does not exist")
            return False

        # Verify HF token
        print(
            f"🔑 Using token: {HF_TOKEN[:5]}...{HF_TOKEN[-5:]} (length: {len(HF_TOKEN)})"
        )

        # Use the HF API with correct authentication
        api = HfApi()
        # Note: HfApi doesn't need explicit token setting in newer versions

        print(" Verifying Hugging Face account access...")
        try:
            user_info = api.whoami(token=HF_TOKEN)
            print(f" Authenticated as: {user_info}")
        except Exception as e:
            print(f" Authentication error: {e}")
            return False

        # Try uploading individual files
        print(f" Starting upload process...")
        uploaded_files = []

        for file in files:
            file_path = os.path.join(local_folder, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)

            print(f"   Uploading {file} ({file_size:.2f} MB)...")
            try:
                # Use upload_file for individual files
                url = api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file,
                    repo_id=experiment_config["hf_repo_name"],
                    token=HF_TOKEN,
                    repo_type="model",
                )
                uploaded_files.append(file)
                print(f"    Uploaded {file} successfully to {url}")
            except Exception as e:
                print(f"    Failed to upload {file}: {e}")

        print(f" Upload summary: {len(uploaded_files)}/{len(files)} files uploaded")
        return len(uploaded_files) > 0
    except Exception as e:
        print(f" Error in upload process: {e}")
        import traceback

        traceback.print_exc()
        return False


def find_latest_checkpoint():
    """Find the latest checkpoint with proper error handling"""

    checkpoint_dir = experiment_config["checkpoint_dir"]
    local_checkpoints = []

    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.endswith(".pt") and "_epoch_" in file:
                # Extract epoch number from filename
                try:
                    epoch_str = file.split("_epoch_")[-1].replace(".pt", "")
                    epoch = int(epoch_str)
                    local_checkpoints.append(
                        (file, epoch, os.path.join(checkpoint_dir, file))
                    )
                except ValueError:
                    pass

    if not local_checkpoints:
        # Try finding in Hugging Face repo if no local checkpoints
        try:
            print("No local checkpoints found. Checking Hugging Face repo...")
            api = HfApi()
            repo_id = experiment_config["hf_repo_name"]
            files = api.list_repo_files(repo_id=repo_id, token=HF_TOKEN)

            hf_checkpoints = []
            for file in files:
                if file.endswith(".pt") and "_epoch_" in file:
                    try:
                        epoch_str = file.split("_epoch_")[-1].replace(".pt", "")
                        epoch = int(epoch_str)
                        # Download the file
                        local_path = os.path.join(
                            checkpoint_dir, os.path.basename(file)
                        )
                        api.hf_hub_download(
                            repo_id=repo_id,
                            filename=file,
                            local_dir=checkpoint_dir,
                            token=HF_TOKEN,
                        )
                        hf_checkpoints.append(
                            (os.path.basename(file), epoch, local_path)
                        )
                    except Exception as e:
                        print(f"Error downloading {file}: {e}")

            if hf_checkpoints:
                print(f"Downloaded {len(hf_checkpoints)} checkpoints from Hugging Face")
                local_checkpoints = hf_checkpoints
        except Exception as e:
            print(f"Error checking Hugging Face for checkpoints: {e}")

    if not local_checkpoints:
        return None, 0

    # Use the latest checkpoint
    latest_file, latest_epoch, latest_path = max(local_checkpoints, key=lambda x: x[1])
    print(f"Found latest checkpoint: {latest_file} (Epoch {latest_epoch})")
    return latest_path, latest_epoch


def load_checkpoint(checkpoint_path):
    """Load checkpoint with more robust error handling"""
    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        # Try loading with YOLO's built-in method first
        model = YOLO(checkpoint_path)
        print(f"Successfully loaded checkpoint using YOLO()")
        return model, True
    except Exception as e:
        print(f"Error loading with YOLO(): {e}")

        # Try loading as a PyTorch model
        try:
            print("Attempting to load as a PyTorch model...")
            # First create a base model
            model_size = experiment_config["model_size"]
            pretrained_path = f"yolov8{model_size}.pt"
            model = YOLO(pretrained_path)

            # Then load the checkpoint state dictionary
            import torch

            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model" in checkpoint and "model_state_dict" in checkpoint["model"]:
                    model.model.load_state_dict(checkpoint["model"]["model_state_dict"])
                    print(
                        "Loaded state dict from checkpoint['model']['model_state_dict']"
                    )
                elif "model" in checkpoint:
                    model.model.load_state_dict(checkpoint["model"])
                    print("Loaded state dict from checkpoint['model']")
                elif "state_dict" in checkpoint:
                    model.model.load_state_dict(checkpoint["state_dict"])
                    print("Loaded state dict from checkpoint['state_dict']")
                elif all(
                    k.startswith(("model.", "0.", "1.", "2."))
                    for k in checkpoint.keys()
                ):
                    # Handle raw state dict
                    model.model.load_state_dict(checkpoint)
                    print("Loaded raw state dictionary")
                else:
                    print(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")
                    return model, False
            else:
                print("Checkpoint is not a dictionary, using pretrained model")
                return model, False

            return model, True
        except Exception as e2:
            print(f"Error loading checkpoint as PyTorch model: {e2}")
            # Fall back to pretrained
            print(f"Falling back to pretrained weights")
            return YOLO(f"yolov8{experiment_config['model_size']}.pt"), False


def save_model_checkpoint(model, epoch, run_name):
    """Save model checkpoint in the proper format for resuming"""
    try:
        checkpoint_dir = experiment_config["checkpoint_dir"]
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_file = f"{checkpoint_dir}/{run_name}_epoch_{epoch}.pt"

        # Use YOLO's built-in save method for consistent format
        model.save(checkpoint_file)

        print(f"Model saved to {checkpoint_file}")
        return checkpoint_file
    except Exception as e:
        print(f"Error saving model: {e}")
        import traceback

        traceback.print_exc()
        return None


def init_wandb():
    """Initialize W&B for tracking"""

    run = wandb.init(
        project=experiment_config["wandb_project"],
        name=experiment_config["wandb_run_name"],
        config=experiment_config,
        notes=experiment_config["experiment_desc"],
        tags=[
            experiment_config["model_type"],
            experiment_config["model_size"],
            *experiment_config["techniques"],
        ],
        resume="allow",
    )

    print(f"Initialized W&B run: {experiment_config['wandb_run_name']}")
    return run.id


def train_yolov8_model():
    """Train the YOLOv8 model with all configurations"""

    # Create KITTI YAML file
    data_yaml_path = reorganize_kitti_dataset_optimized()
    experiment_config["data_yaml_path"] = data_yaml_path

    # Initialize W&B
    wandb_run_id = init_wandb()

    # Find latest checkpoint
    checkpoint_path, start_epoch = find_latest_checkpoint()

    # Model initialization
    if checkpoint_path:
        print(f"Found checkpoint for epoch {start_epoch}: {checkpoint_path}")
        model, loaded_successfully = load_checkpoint(checkpoint_path)
        if loaded_successfully:
            print(f"Successfully loaded checkpoint from epoch {start_epoch}")
        else:
            print("Using pretrained model as checkpoint loading failed")
    else:
        print("Starting new training run from pretrained weights")
        pretrained_path = f"yolov8{experiment_config['model_size']}.pt"
        model = YOLO(pretrained_path)

    # Early stopping callback
    def add_early_stopping(model, patience: int = 6):
        def on_epoch_end(trainer):
            # get current mAP50 from the metrics dict
            current = trainer.metrics.get("metrics/mAP50(B)", None)
            if current is None:
                return
            # initialize on first epoch
            if not hasattr(trainer, "best_map"):
                trainer.best_map = current
                trainer.early_stop_counter = 0
                print(f"\n[EarlyStopping] Initialized mAP50 = {current:.4f}")
                return
            # check improvement
            if current > trainer.best_map:
                trainer.best_map = current
                trainer.early_stop_counter = 0
                print(f"\n[EarlyStopping] New best mAP50: {current:.4f}")
            else:
                trainer.early_stop_counter += 1
                print(
                    f"\n[EarlyStopping] No improvement for {trainer.early_stop_counter}/{patience} epochs (best = {trainer.best_map:.4f})"
                )
            if trainer.early_stop_counter >= patience:
                print(f"\n[EarlyStopping] Patience exceeded; stopping training.")
                trainer.stop_training = True

        model.add_callback("on_train_epoch_end", on_epoch_end)
        print(f"Added EarlyStopping (monitor=mAP50, patience={patience})")
        return model

    model = add_early_stopping(model, patience=6)

    # Training-end callback: safe W&B logging and checkpointing
    def on_train_epoch_end(trainer):
        epoch = trainer.epoch + 1
        log_data = {}

        # pull mAP50 and mAP50-95 from the metrics dict
        m50 = trainer.metrics.get("metrics/mAP50(B)", None)
        m5095 = trainer.metrics.get("metrics/mAP50-95(B)", None)
        if m50 is not None:
            log_data["train_mAP50"] = float(m50)
        if m5095 is not None:
            log_data["train_mAP50-95"] = float(m5095)

        # learning rate
        try:
            lr = trainer.optimizer.param_groups[0]["lr"]
            log_data["train_lr"] = lr
        except Exception:
            pass

        if log_data:
            wandb.log(log_data)

        # checkpoint save
        if (
            epoch % experiment_config["save_period"] == 0
            or epoch == experiment_config["epochs"]
        ):
            os.makedirs(experiment_config["checkpoint_dir"], exist_ok=True)
            ckpt_file = (
                f"{experiment_config['checkpoint_dir']}/"
                f"{experiment_config['wandb_run_name']}_epoch_{epoch}.pt"
            )
            if hasattr(trainer, "model"):
                save_fn = getattr(trainer.model, "save", None)
                if callable(save_fn):
                    save_fn(ckpt_file)
                else:
                    import torch

                    torch.save(trainer.model.model.state_dict(), ckpt_file)
                print(f"Saved checkpoint to {ckpt_file}")
                upload_to_hf(
                    experiment_config["checkpoint_dir"], f"checkpoint epoch {epoch}"
                )
            else:
                print("Warning: trainer.model not found; skipping save")

    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # Validation-end callback: safe per-class and IoU logging
    def on_val_end(validator):
        # validator.metrics is a DetMetrics
        m = validator.metrics
        class_names = [
            "Car", "Cyclist", "DontCare", "Misc", "Pedestrian",
            "Person_sitting", "Tram", "Truck", "Van"
        ]

        # Overall metrics
        overall = {
            "val_mAP50": float(m.box.map50),
            "val_mAP50-95": float(m.box.map),
        }
        if hasattr(m.box, "iou"):
            overall["val_IoU"] = float(m.box.iou)
        wandb.log(overall)

        # Per‑class AP50
        ap50_per_class = getattr(m.box, "ap50", None)
        if ap50_per_class is not None:
            for idx, name in enumerate(class_names):
                wandb.log({f"val_AP50_{name}": float(ap50_per_class[idx])})

    # def on_val_end(validator):
    #     metrics = validator.metrics
    #     class_names = [
    #         "Car",
    #         "Cyclist",
    #         "DontCare",
    #         "Misc",
    #         "Pedestrian",
    #         "Person_sitting",
    #         "Tram",
    #         "Truck",
    #         "Van",
    #     ]
    #     # Per-class AP50
    #     for idx, name in enumerate(class_names):
    #         ap50 = metrics.get(f"metrics/mAP50(B)/{idx}", None)
    #         if ap50 is not None:
    #             wandb.log({f"val_AP50_{name}": ap50})
    #     # IoU
    #     iou = metrics.get("metrics/IoU(B)", None)
    #     if iou is not None:
    #         wandb.log({"val_IoU": iou})

    model.add_callback("on_val_end", on_val_end)

    # Set up training arguments
    run_name = experiment_config["wandb_run_name"]
    train_args = {
        "data": experiment_config["data_yaml_path"],
        "epochs": experiment_config["epochs"],
        "batch": experiment_config["batch_size"],
        "imgsz": experiment_config["img_size"],
        "optimizer": experiment_config["optimizer"],
        "lr0": experiment_config["lr0"],
        "lrf": experiment_config["lrf"],
        "momentum": experiment_config["momentum"],
        "weight_decay": experiment_config["weight_decay"],
        "warmup_epochs": experiment_config["warmup_epochs"],
        "warmup_momentum": experiment_config["warmup_momentum"],
        "warmup_bias_lr": experiment_config["warmup_bias_lr"],
        "box": experiment_config["box"],
        "cls": experiment_config["cls"],
        "hsv_h": experiment_config["hsv_h"],
        "hsv_s": experiment_config["hsv_s"],
        "hsv_v": experiment_config["hsv_v"],
        "degrees": experiment_config["degrees"],
        "translate": experiment_config["translate"],
        "scale": experiment_config["scale"],
        "shear": experiment_config["shear"],
        "perspective": experiment_config["perspective"],
        "flipud": experiment_config["flipud"],
        "fliplr": experiment_config["fliplr"],
        "mosaic": experiment_config["mosaic"],
        "mixup": experiment_config["mixup"],
        "copy_paste": experiment_config["copy_paste"],
        "project": experiment_config["output_dir"],
        "name": run_name,
        "exist_ok": True,
        "resume": checkpoint_path is not None,  # Set resume flag if checkpoint exists
        "device": "0",  # Use GPU if available
        "workers": 8,
        "val": True,
        "save": True,
        "save_period": experiment_config["save_period"],
        "verbose": True,
    }

    # Training progress bar and start
    print(f"Starting training for {experiment_config['epochs']} epochs")
    with tqdm(
        total=experiment_config["epochs"] - start_epoch, desc="Training Progress"
    ) as pbar:

        def update_pbar(trainer):
            pbar.update(1)
            pbar.set_postfix(
                {
                    "epoch": trainer.epoch + 1,
                    "mAP50": trainer.metrics.get("metrics/mAP50(B)", 0),
                    "loss": trainer.metrics.get("train/box_loss", 0),
                }
            )

        model.add_callback("on_train_epoch_end", update_pbar)
        results = model.train(**train_args)

    def enhance_wandb_logging(results, model):
        """Add comprehensive visualizations to W&B"""
        print("Enhancing W&B logging with additional visualizations...")

        # Define class names directly in the function to avoid KeyError
        class_names = [
            "Car",
            "Cyclist",
            "DontCare",
            "Misc",
            "Pedestrian",
            "Person_sitting",
            "Tram",
            "Truck",
            "Van",
        ]
        class_dict = {i: name for i, name in enumerate(class_names)}

        try:
            # # 1. Add confusion matrix
            # if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
            #     conf_matrix = results.confusion_matrix

            #     # Use the locally defined class names instead of trying to access experiment_config
            #     wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            #         probs=None,
            #         y_true=list(range(len(class_names))),
            #         preds=list(range(len(class_names))),
            #         class_names=class_names
            #     )})

            # # 2. Add PR curves
            # for i, class_name in enumerate(class_names):
            #     try:
            #         if hasattr(model, 'val') and hasattr(model.val, 'get_stats'):
            #             pr_data = model.val.get_stats().get(f'PR_{i}')
            #             if pr_data is not None:
            #                 precision, recall = pr_data

            #                 # Create table for PR curve
            #                 pr_table = wandb.Table(columns=["precision", "recall", "class"])
            #                 for p, r in zip(precision, recall):
            #                     pr_table.add_data(p, r, class_name)

            #                 wandb.log({
            #                     f"PR_curve_{class_name}": wandb.plot.line(
            #                         pr_table, "recall", "precision",
            #                         title=f"Precision-Recall Curve - {class_name}"
            #                     )
            #                 })
            #     except Exception as e:
            #         print(f"Could not generate PR curve for class {class_name}: {e}")

            # # 3. Add class distribution analysis
            # class_counts = {name: 0 for name in class_names}

            # # Count class instances in validation set
            # if hasattr(model, 'val') and hasattr(model.val, 'dataset'):
            #     val_dataset = model.val.dataset
            #     try:
            #         for batch_i, batch in enumerate(val_dataset):
            #             if batch_i >= 100:  # Limit to 100 batches for speed
            #                 break

            #             if len(batch) >= 2:  # Make sure the batch has labels
            #                 _, labels, _ = batch

            #                 # Process labels to count classes
            #                 if labels.shape[0] > 0:
            #                     for label in labels:
            #                         if len(label) > 0:  # Make sure label has elements
            #                             cls_id = int(label[0])  # Class ID is typically first column
            #                             if 0 <= cls_id < len(class_names):
            #                                 class_name = class_names[cls_id]
            #                                 class_counts[class_name] += 1
            #     except Exception as e:
            #         print(f"Error counting validation classes: {e}")

            # # Create table for class distribution
            # class_dist_table = wandb.Table(columns=["class", "count"])
            # for cls_name, count in class_counts.items():
            #     class_dist_table.add_data(cls_name, count)

            # wandb.log({
            #     "validation_class_distribution": wandb.plot.bar(
            #         class_dist_table, "class", "count",
            #         title="Class Distribution in Validation Set"
            #     )
            # })

            # 4. Add model summary with parameters and FLOPs
            try:
                # Log model architecture
                if hasattr(model, "model"):
                    wandb.log({"model_summary": wandb.Text(str(model.model))})

                    # Calculate and log model size
                    total_params = sum(p.numel() for p in model.model.parameters())
                    trainable_params = sum(
                        p.numel() for p in model.model.parameters() if p.requires_grad
                    )

                    wandb.log(
                        {
                            "total_parameters": total_params,
                            "trainable_parameters": trainable_params,
                        }
                    )
            except Exception as e:
                print(f"Could not log model architecture details: {e}")

            # # 5. Add performance metrics table
            # if hasattr(results, 'box'):
            #     metrics_table = wandb.Table(columns=["Metric", "Value"])

            #     # Add overall metrics - make sure they are scalar values
            #     metrics_table.add_data("mAP50", float(results.box.map50))
            #     metrics_table.add_data("mAP50-95", float(results.box.map))

            #     # Handle potentially array metrics by using mean values
            #     if isinstance(results.box.p, np.ndarray):
            #         metrics_table.add_data("Precision (mean)", float(np.mean(results.box.p)))
            #     else:
            #         metrics_table.add_data("Precision", float(results.box.p))

            #     if isinstance(results.box.r, np.ndarray):
            #         metrics_table.add_data("Recall (mean)", float(np.mean(results.box.r)))
            #     else:
            #         metrics_table.add_data("Recall", float(results.box.r))

            #     if isinstance(results.box.f1, np.ndarray):
            #         metrics_table.add_data("F1-Score (mean)", float(np.mean(results.box.f1)))
            #     else:
            #         metrics_table.add_data("F1-Score", float(results.box.f1))

            #     # Add per-class metrics to a separate table
            #     if len(class_names) > 0:
            #         class_metrics = wandb.Table(columns=["Class", "AP50"])
            #         for i, class_name in enumerate(class_names):
            #             if hasattr(results.box, f'ap50_{i}'):
            #                 ap50 = float(getattr(results.box, f'ap50_{i}'))
            #                 class_metrics.add_data(class_name, ap50)

            #         wandb.log({"per_class_metrics": class_metrics})
            #     else:
            #         # Log main metrics table
            #         wandb.log({"performance_metrics": metrics_table})

            print("Enhanced W&B logging complete")

        except Exception as e:
            print(f"Error during W&B logging enhancement: {e}")
            import traceback

            traceback.print_exc()

    # Enhance the W&B logging
    enhance_wandb_logging(results, model)

    # Final upload to Hugging Face
    print("Training complete. Uploading final model to Hugging Face...")
    final_results_path = f"{experiment_config['output_dir']}/{run_name}"
    upload_to_hf(final_results_path, "Final model after training")

    # Close W&B run
    wandb.finish()

    return results


def analyze_model_performance():
    """Analyze model performance with detailed metrics and proper data path"""

    # Load the trained model
    run_name = experiment_config["wandb_run_name"]
    model_path = f"{experiment_config['output_dir']}/{run_name}/weights/best.pt"

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Best model not found at {model_path}, looking for last model...")
        model_path = f"{experiment_config['output_dir']}/{run_name}/weights/last.pt"

        if not os.path.exists(model_path):
            print(
                f"Last model not found either. Trying to use the latest checkpoint..."
            )
            checkpoint_path, _ = find_latest_checkpoint()
            if checkpoint_path:
                model_path = checkpoint_path
            else:
                print("No model found for evaluation. Skipping analysis.")
                return None

    print(f"Evaluating model: {model_path}")

    # Use the improved loading method
    model, loaded_successfully = load_checkpoint(model_path)
    if not loaded_successfully:
        print("Warning: Model may not have loaded correctly")

    # Do NOT initialize a new W&B run - these lines are removed

    print("Evaluating model on test set...")

    # Ensure proper path to data YAML file
    data_yaml_path = experiment_config.get("data_yaml_path")
    if not data_yaml_path or not os.path.exists(data_yaml_path):
        print(f"Data YAML file not found at {data_yaml_path}")
        data_yaml_path = "/content/drive/My Drive/MOSEF/DL/Kitti_YOLO/data.yaml"  # Fallback to known path
        if not os.path.exists(data_yaml_path):
            print("Cannot find data YAML file. Aborting evaluation.")
            return None

    print(f"Using data YAML file: {data_yaml_path}")

    try:
        # Full test set evaluation
        results = model.val(data=data_yaml_path)

        # Print results to console
        print("\nAnalysis complete. Results:")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")

        # Handle potentially array metrics
        p_value = results.box.p
        r_value = results.box.r
        f1_value = results.box.f1

        print(
            f"Precision: {float(np.mean(p_value)) if isinstance(p_value, np.ndarray) else float(p_value):.4f}"
        )
        print(
            f"Recall: {float(np.mean(r_value)) if isinstance(r_value, np.ndarray) else float(r_value):.4f}"
        )
        print(
            f"F1-Score: {float(np.mean(f1_value)) if isinstance(f1_value, np.ndarray) else float(f1_value):.4f}"
        )

        # Print per-class metrics if available
        if hasattr(results.box, "classes") and results.box.classes:
            class_names = [
                "Car",
                "Cyclist",
                "DontCare",
                "Misc",
                "Pedestrian",
                "Person_sitting",
                "Tram",
                "Truck",
                "Van",
            ]
            print("\nPer-class results:")
            for i, class_name in enumerate(class_names):
                if i < len(class_names):
                    class_map = getattr(results.box, f"map50_{i}", 0)
                    class_precision = (
                        p_value[i]
                        if isinstance(p_value, np.ndarray) and i < len(p_value)
                        else 0
                    )
                    class_recall = (
                        r_value[i]
                        if isinstance(r_value, np.ndarray) and i < len(r_value)
                        else 0
                    )
                    print(
                        f"{class_name}: mAP50={class_map:.4f}, Precision={class_precision:.4f}, Recall={class_recall:.4f}"
                    )

        return results

    except Exception as e:
        print(f"Error during model evaluation: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Main execution function"""

    # Step 1: Set up the experiment (HF repo, W&B project)
    setup_experiment()

    # Step 2: Train the model with enhanced monitoring
    training_results = train_yolov8_model()

    # Step 3: Analyze model performance
    analysis_results = analyze_model_performance()

    print("Experiment complete!")
    print(f"Model saved to HF repository: {experiment_config['hf_repo_name']}")
    print(f"Results logged to W&B project: {experiment_config['wandb_project']}")
    print(f"Run name: {experiment_config['wandb_run_name']}")

    # Print key performance metrics - handle None result case
    print("\nModel Performance Summary:")
    if analysis_results is not None and hasattr(analysis_results, "box"):
        print(f"mAP50: {analysis_results.box.map50:.4f}")
        print(f"mAP50-95: {analysis_results.box.map:.4f}")
        print(
            f"Precision: {float(np.mean(analysis_results.box.p)) if isinstance(analysis_results.box.p, np.ndarray) else analysis_results.box.p:.4f}"
        )
        print(
            f"Recall: {float(np.mean(analysis_results.box.r)) if isinstance(analysis_results.box.r, np.ndarray) else analysis_results.box.r:.4f}"
        )
        print(
            f"F1-Score: {float(np.mean(analysis_results.box.f1)) if isinstance(analysis_results.box.f1, np.ndarray) else analysis_results.box.f1:.4f}"
        )
    else:
        # Use the training results instead if available
        if training_results is not None and hasattr(training_results, "box"):
            print(f"mAP50: {training_results.box.map50:.4f}")
            print(f"mAP50-95: {training_results.box.map:.4f}")
            print(
                f"Precision: {float(np.mean(training_results.box.p)) if isinstance(training_results.box.p, np.ndarray) else training_results.box.p:.4f}"
            )
            print(
                f"Recall: {float(np.mean(training_results.box.r)) if isinstance(training_results.box.r, np.ndarray) else training_results.box.r:.4f}"
            )
            print(
                f"F1-Score: {float(np.mean(training_results.box.f1)) if isinstance(training_results.box.f1, np.ndarray) else training_results.box.f1:.4f}"
            )
        else:
            print(
                "No performance metrics available. Check logs for validation results."
            )


if __name__ == "__main__":
    main()
