import os
import torch
from torchvision.datasets import Kitti
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import uuid
import glob
import requests
import shutil
import random
from datetime import datetime
from pathlib import Path
import wandb
from huggingface_hub import HfApi, create_repo, login
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.transforms import functional as F
from torchvision.utils import make_grid
from torchvision.transforms import Resize, RandomAffine
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import math

# Authentication tokens
HF_TOKEN = "...."  # Hugging Face token
WANDB_API_KEY = "..."  # W&B API key
HF_USERNAME = "..."  # Username on Hugging Face

experiment_config = {
    # Model and technique selection
    "model_type": "faster_rcnn",
    "backbone": "resnet50_fpn",
    "techniques": ["augmentation"],
    # Training parameters
    "epochs": 40,
    "batch_size": 16,
    "img_size": 800,  # R-CNN can handle larger images
    "lr0": 0.005,
    "lrf": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "warmup_epochs": 5.0,
    "optimizer": "Adam",
    # Keep only relevant augmentations
    "scale": 0.2,  # Scale augmentation can be useful
    "translate": 0.1,  # Translation is useful
    # Dataset and output settings
    "kitti_dir": "/home/onyxia/work/datasets",
    "rcnn_dir": "/home/onyxia/work/datasets/Kitti_RCNN",  # Add this to point to processed data
    "output_dir": "runs/detect",
    "checkpoint_dir": "checkpoints",
    "save_period": 1,
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
    model_name = f"{experiment_config['model_type']}-{experiment_config['backbone']}"
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
    run_name = f"{experiment_config['model_type']}-{experiment_config['backbone']}-{techniques_str}_{timestamp}_{experiment_id}"

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
    model_name = f"{experiment_config['model_type']}-{experiment_config['backbone']}"
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
    "car": 1,
    "cyclist": 2,
    "misc": 3,
    "pedestrian": 4,
    "person_sitting": 5,
    "tram": 6,
    "truck": 7,
    "van": 8,
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


def process_file_rcnn(data):
    """Process a single file (convert and copy) for Faster R-CNN"""
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

    # Copy image without resizing
    shutil.copy(src_img, dst_img)

    if not os.path.exists(src_label):
        return 0

    conversion_count = 0

    # Convert KITTI format to R-CNN format (absolute coordinates)
    with open(src_label, "r") as f_in, open(dst_label, "w") as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            class_name = parts[0].lower()
            if class_name not in CLASS_MAPPING:
                continue

            # Extract bounding box coordinates (already in absolute format in KITTI)
            x1, y1, x2, y2 = (
                float(parts[4]),
                float(parts[5]),
                float(parts[6]),
                float(parts[7]),
            )

            # Get class ID
            class_id = CLASS_MAPPING[class_name]

            # Write in format class_id x1 y1 x2 y2
            f_out.write(f"{class_id} {x1} {y1} {x2} {y2}\n")
            conversion_count += 1

    return conversion_count


def save_batch_grid(loader, path):
    """Save a grid of sample images from a data loader"""
    # Get first batch of images and targets
    images, _ = next(iter(loader))
    # Take up to 16 images to display
    imgs = images[:16]

    # Find the maximum dimensions
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)
    
    # Create a new list of uniformly sized tensors
    uniform_imgs = []
    
    for img in imgs:
        # Create a new tensor of the max size
        c, h, w = img.shape
        new_img = torch.zeros((c, max_h, max_w), dtype=img.dtype)
        
        # Copy the original image into the new tensor
        new_img[:, :h, :w] = img
        
        # Add to our list
        uniform_imgs.append(new_img)
    
    # Now all tensors are the same size, we can stack safely
    grid = make_grid(torch.stack(uniform_imgs), nrow=4, normalize=True)
    plt.imsave(path, grid.permute(1, 2, 0).cpu().numpy())


def download_kitti_dataset():
    """Download KITTI dataset if not already downloaded"""
    raw_training_dir = os.path.join(
        experiment_config["kitti_dir"], "Kitti", "raw", "training"
    )
    if os.path.exists(raw_training_dir):
        print(f"KITTI dataset already exists at {raw_training_dir}")
        return

    print("Downloading KITTI dataset...")
    os.makedirs(experiment_config["kitti_dir"], exist_ok=True)
    _ = Kitti(
        root=experiment_config["kitti_dir"],
        train=True,
        download=True,
     )
    print(f"Downloaded KITTI into {experiment_config['kitti_dir']}")


def reorganize_kitti_dataset_for_rcnn():
    """Reorganize KITTI dataset with optimized processing for Faster R-CNN"""

    # Base directories
    kitti_dir = experiment_config["kitti_dir"]
    rcnn_dir  = experiment_config["rcnn_dir"]

    # Source directories
    train_img_src = os.path.join(kitti_dir, "Kitti", "raw", "training", "image_2")
    train_label_src = os.path.join(kitti_dir, "Kitti", "raw", "training", "label_2")

    # Target directories
    train_img_dst = os.path.join(rcnn_dir, "train", "images")
    train_label_dst = os.path.join(rcnn_dir, "train", "labels")
    val_img_dst = os.path.join(rcnn_dir, "val", "images")
    val_label_dst = os.path.join(rcnn_dir, "val", "labels")

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
        return os.path.join(rcnn_dir, "data.yaml")

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
    minority_classes = ["cyclist", "person_sitting", "tram", "misc", "truck"]

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
        futures = [executor.submit(process_file_rcnn, arg) for arg in train_args]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Converting training data"
        ):
            train_count += future.result()

    # Process validation images
    print("Processing validation images...")
    val_args = [(basename, src_dirs, dst_dirs, False) for basename in val_images]
    val_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file_rcnn, arg) for arg in val_args]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Converting validation data"
        ):
            val_count += future.result()

    # Create YAML file
    data_yaml = {
        "path": rcnn_dir,
        "train": "train/images",
        "val": "val/images",
        "nc": 8,
        "names": [
            "car",
            "cyclist",
            "misc",
            "pedestrian",
            "person_sitting",
            "tram",
            "truck",
            "van",
        ],
    }

    yaml_path = os.path.join(rcnn_dir, "data.yaml")
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


def load_rcnn_checkpoint(checkpoint_path, model, optimizer=None, lr_scheduler=None):
    """Load checkpoint for Faster R-CNN model"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint.get("epoch", 0)
        print(f"Loaded checkpoint from epoch {epoch}")
        return epoch
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0


def init_wandb():
    """Initialize W&B for tracking"""

    run = wandb.init(
        project=experiment_config["wandb_project"],
        name=experiment_config["wandb_run_name"],
        config=experiment_config,
        notes=experiment_config["experiment_desc"],
        tags=[
            experiment_config["model_type"],
            experiment_config["backbone"],
            *experiment_config["techniques"],
        ],
        resume="allow",
    )

    print(f"Initialized W&B run: {experiment_config['wandb_run_name']}")
    return run.id


# Create model function
def create_faster_rcnn_model(num_classes):
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier with a new one for your number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# Custom dataset class for Faster R-CNN
class KittiRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(img_dir)))
        self.labels = [p.replace(".png", ".txt") for p in self.imgs]

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        img = Image.open(img_path).convert("RGB")

        # Load labels and convert to Faster R-CNN format
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) >= 5:
                        class_id = int(data[0])
                        # Read directly as absolute coordinates (no conversion needed)
                        x1, y1, x2, y2 = map(float, data[1:5])

                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ResizeWithBBox:
    """
    Resize the PIL image to `size` and scale all boxes in target accordingly.
    `size` can be a (h, w) tuple.
    """
    def __init__(self, size):
        # torchvision’s Resize takes (h, w)
        self.size = size
        self.resize = Resize(size)

    def __call__(self, image, target):
        # original size
        w0, h0 = image.size  # PIL gives (width, height)
        # resize image
        image = self.resize(image)
        # new size
        h1, w1 = self.size   # note: torchvision Resize(size) uses (h, w)

        # scale boxes if present
        if "boxes" in target and len(target["boxes"]) > 0:
            # convert to tensor if needed
            boxes = target["boxes"]
            # compute scale factors
            x_scale = w1 / w0
            y_scale = h1 / h0
            # scale [x1,y1,x2,y2]
            boxes = boxes * torch.tensor([x_scale, y_scale, x_scale, y_scale])
            target["boxes"] = boxes

        return image, target


class RandomAffineWithBBox:
    """
    Apply the same RandomAffine to the PIL image and the target boxes.
    Only supports degrees=0 (no rotation), but scale+translate.
    """
    def __init__(self, degrees, translate, scale):
        self.degrees   = degrees
        self.translate = translate
        self.scale     = scale

    def __call__(self, image, target):
        # Sample affine params the same way torchvision does internally:
        # angle, translations (tx, ty), scale, shear
        angle, (tx, ty), scale, shear = RandomAffine.get_params(
            degrees=( -self.degrees, self.degrees ),
            translate=self.translate,
            scale_ranges=(1.0 - self.scale, 1.0 + self.scale),
            shears=None,
            img_size=image.size  # (W, H)
        )

        # 1) warp the image
        image = F.affine(image, angle=angle, translate=(tx, ty),
                         scale=scale, shear=shear)

        # 2) transform each box
        if "boxes" in target:
            boxes = target["boxes"]  # tensor of shape [N,4] in [x1,y1,x2,y2]
            # build the 2×3 affine matrix
            theta = math.radians(angle)
            a = scale * math.cos(theta)
            b = scale * math.sin(theta)
            # F.affine uses the matrix [[ a, -b, tx ], [ b,  a, ty ]]
            matrix = torch.tensor([[ a, -b, tx ],
                                   [ b,  a, ty ]], dtype=torch.float32)

            new_boxes = []
            for box in boxes:
                # corner points in homogeneous coords
                x1, y1, x2, y2 = box
                corners = torch.tensor([[x1,y1,1],
                                        [x1,y2,1],
                                        [x2,y1,1],
                                        [x2,y2,1]], dtype=torch.float32)  # (4,3)
                warped = (matrix @ corners.T).T  # (4,2)
                xs = warped[:,0]; ys = warped[:,1]
                x1n, x2n = xs.min(), xs.max()
                y1n, y2n = ys.min(), ys.max()
                new_boxes.append([x1n, y1n, x2n, y2n])

            target["boxes"] = torch.stack([torch.tensor(b) for b in new_boxes])

        return image, target

# Training function
def train_rcnn_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, image, target):
            for t in self.transforms:
                image, target = t(image, target)
            return image, target

    class ToTensor:
        def __call__(self, image, target):
            image = F.to_tensor(image)
            return image, target

    class RandomHorizontalFlip:
        def __init__(self, prob=0.5):
            self.prob = prob

        def __call__(self, image, target):
            if random.random() < self.prob:
                image = F.hflip(image)
                width = image.shape[-1]

                # Flip boxes
                if "boxes" in target:
                    boxes = target["boxes"]
                    boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                    target["boxes"] = boxes

            return image, target

    # Create transforms
    transforms = Compose([
        # 1) resize to square using your img_size
        ResizeWithBBox((experiment_config["img_size"], experiment_config["img_size"])),
        # 2) random scale & translate
        RandomAffineWithBBox(
            degrees=0,
            scale=experiment_config["scale"],
            translate=(experiment_config["translate"], experiment_config["translate"])
        ),
        # 3) convert to tensor
        ToTensor(),
        # 4) horizontal flip
        RandomHorizontalFlip(0.5),
    ])


    # Create datasets
    train_dataset = KittiRCNNDataset(
        img_dir=os.path.join(experiment_config["rcnn_dir"], "train", "images"),
        label_dir=os.path.join(experiment_config["rcnn_dir"], "train", "labels"),
        transforms=transforms,
    )

    val_dataset = KittiRCNNDataset(
        img_dir=os.path.join(experiment_config["rcnn_dir"], "val", "images"),
        label_dir=os.path.join(experiment_config["rcnn_dir"], "val", "labels"),
        transforms=transforms,
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=experiment_config["batch_size"],
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=4,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=experiment_config["batch_size"],
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=4,
    )

    # Create model
    model = create_faster_rcnn_model(num_classes=len(CLASS_MAPPING)+1)
    model.to(device)


    # Optimizer selection from config
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = experiment_config["optimizer"].lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=experiment_config["lr0"],
            momentum=experiment_config["momentum"],
            weight_decay=experiment_config["weight_decay"],
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=experiment_config["lr0"],
            weight_decay=experiment_config["weight_decay"],
        )
    else:
        raise ValueError(f"Unsupported optimizer: {experiment_config['optimizer']}")


    # Warm-up scheduler for the first N epochs, then cosine thereafter
    warmup_epochs = experiment_config["warmup_epochs"]
    total_warmup_iters = int(warmup_epochs * len(train_loader))

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=total_warmup_iters,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=experiment_config["epochs"] - warmup_epochs,
        eta_min=experiment_config["lr0"] * experiment_config["lrf"],
    )
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


    # Initialize W&B
    wandb_run_id = init_wandb()

    all_logs = []

    # Training loop
    for epoch in range(experiment_config["epochs"]):
        # Set model to training mode
        model.train()

        # Track metrics
        train_loss = 0.0
        train_box_loss = 0.0
        train_cls_loss = 0.0

        with tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{experiment_config['epochs']}"
        ) as pbar:
            for images, targets in pbar:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Backward and optimize
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                # Track losses
                train_loss += losses.item()
                train_box_loss += (
                    loss_dict["loss_box_reg"].item()
                    if "loss_box_reg" in loss_dict
                    else 0
                )
                train_cls_loss += (
                    loss_dict["loss_classifier"].item()
                    if "loss_classifier" in loss_dict
                    else 0
                )

                pbar.set_postfix(loss=losses.item())

        # Update learning rate
        lr_scheduler.step()

        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_train_box_loss = train_box_loss / len(train_loader)
        avg_train_cls_loss = train_cls_loss / len(train_loader)

        # Evaluate on validation set
        val_metrics = evaluate_rcnn(model, val_loader, device)

        # now log exactly the same names YOLO uses
        log_dict = {
            "train_box_loss": avg_train_box_loss,
            "train_cls_loss": avg_train_cls_loss,
            "train_lr":       optimizer.param_groups[0]["lr"],
            "val_mAP50":      val_metrics["mAP50"],
            "val_mAP50-95":   val_metrics["mAP50-95"],
        }

        # dump each per-class AP50 under val_AP50_<classname>
        for cls, ap50 in val_metrics["per_class_AP50"].items():
            log_dict[f"val_AP50_{cls}"] = ap50

        wandb.log(log_dict)

        all_logs.append({"epoch": epoch+1, **log_dict})

        # Save checkpoint
        if (epoch + 1) % experiment_config["save_period"] == 0:
            checkpoint_file = f"{experiment_config['checkpoint_dir']}/{experiment_config['wandb_run_name']}_epoch_{epoch+1}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                },
                checkpoint_file,
            )

            # Upload to HF
            upload_to_hf(
                experiment_config["checkpoint_dir"], f"checkpoint epoch {epoch+1}"
            )


    # --- SAVE CSV ---
    out_dir = Path(experiment_config["output_dir"])
    out_dir.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(all_logs)
    csv_path = out_dir/"results.csv"
    df.to_csv(csv_path, index=False)

    wandb.finish()

    # --- PLOT AND SAVE METRIC CURVES ---
    # 1) train vs val losses & mAP50 / mAP50-95
    fig, axes = plt.subplots(2, 3, figsize=(15,10))
    df.plot(x="epoch", y="train_box_loss", ax=axes[0,0], marker="o",    title="train_box_loss")
    df.plot(x="epoch", y="train_cls_loss", ax=axes[0,1], marker="o",    title="train_cls_loss")
    df.plot(x="epoch", y="train_lr",       ax=axes[0,2], marker="o",    title="train_lr")
    df.plot(x="epoch", y="val_mAP50",      ax=axes[1,1], marker="o",    title="val_mAP50")
    df.plot(x="epoch", y="val_mAP50-95",   ax=axes[1,2], marker="o",    title="val_mAP50-95")
    axes[1,0].axis("off")
    plt.tight_layout()
    fig.savefig(out_dir/"epoch_metrics.png")

    # 2) Per‑class AP50 over epochs (one curve per class)
    classes = list(val_metrics["per_class_AP50"].keys())
    fig, ax = plt.subplots(figsize=(8,6))
    for cls in classes:
        col = f"val_AP50_{cls}"
        df.plot(x="epoch", y=col, ax=ax, lw=1, label=cls)
    ax.set_title("Per‑class val AP50")
    ax.set_xlabel("epoch")
    ax.set_ylabel("AP50")
    ax.legend()
    fig.savefig(out_dir/"per_class_AP50.png")

    # --- SAVE IMAGE GRIDS ---
    save_batch_grid(train_loader, out_dir/"train_batch.png")
    save_batch_grid(val_loader,   out_dir/"val_batch.png")

    # --- UPLOAD FINAL ARTIFACTS ---
    upload_to_hf(str(out_dir), "final training images, CSV & plots")

    return model, val_metrics


def evaluate_rcnn(model, data_loader, device):
    model.eval()
    predictions = []
    gt_annotations = []
    ann_id = 0
    image_id = 0

    with torch.no_grad():
        for images, batch_targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, batch_targets):
                # assign this image a global ID
                img_id = image_id
                image_id += 1

                # collect detections
                boxes  = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    predictions.append({
                        "image_id":    img_id,
                        "category_id": int(label),
                        "bbox":        [float(box[0]), float(box[1]),
                                        float(box[2] - box[0]), float(box[3] - box[1])],
                        "score":       float(score),
                    })

                # collect ground truth
                gt_boxes  = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                for box, label in zip(gt_boxes, gt_labels):
                    gt_annotations.append({
                        "id":           ann_id,
                        "image_id":     img_id,
                        "category_id":  int(label),
                        "bbox":         [float(box[0]), float(box[1]),
                                         float(box[2] - box[0]), float(box[3] - box[1])],
                        "area":         float((box[2] - box[0]) * (box[3] - box[1])),
                        "iscrowd":      0,
                    })
                    ann_id += 1

    # build minimal COCO-style dataset
    coco_gt = COCO()
    coco_gt.dataset = {
        "images":      [{"id": i} for i in range(image_id)],
        "categories": [{"id": CLASS_MAPPING[name], "name": name}
                       for name in CLASS_MAPPING.keys()],
        "annotations": gt_annotations
    }
    coco_gt.createIndex()

    # load detections
    coco_dt = coco_gt.loadRes(predictions) if predictions else coco_gt

    # run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # per-class AP50
    per_class = {}
    for idx, class_name in enumerate(CLASS_MAPPING.keys()):
        pr = coco_eval.eval['precision'][0, :, idx, 0, -1]
        valid = pr[pr >= 0]
        per_class[class_name] = float(valid.mean()) if len(valid) else 0.0

    return {
        "mAP50":     float(coco_eval.stats[1]),
        "mAP50-95":  float(coco_eval.stats[0]),
        "precision": float(np.mean(coco_eval.eval["precision"][0, :, :, 0, -1])),
        "recall":    float(np.mean(coco_eval.eval["recall"][0, :, 0, -1])),
        "per_class_AP50": per_class,
    }



# Helper function to calculate IoU between two bounding boxes
def calculate_box_iou(box1, box2):
    """
    Calculate IoU between box1 and box2
    Boxes are in format [x1, y1, x2, y2]
    """
    # Calculate intersection area
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area

    
def main():
    """Main execution function"""
    # Step 1: Set up the experiment
    setup_experiment()

    # Step 2: Reorganize dataset for R-CNN
    download_kitti_dataset()
    data_yaml_path = reorganize_kitti_dataset_for_rcnn()
    experiment_config["data_yaml_path"] = data_yaml_path

    # Step 3: Train the model
    model, training_metrics = train_rcnn_model()

    # Step 4: Print metrics
    print("Experiment complete!")
    print(f"Model saved to HF repository: {experiment_config['hf_repo_name']}")
    print(f"Results logged to W&B project: {experiment_config['wandb_project']}")
    print(f"Run name: {experiment_config['wandb_run_name']}")

    print("\nModel Performance Summary:")
    print(f"mAP50: {training_metrics['mAP50']:.4f}")
    print(f"mAP50-95: {training_metrics['mAP50-95']:.4f}")
    print(f"Precision: {training_metrics['precision']:.4f}")
    print(f"Recall: {training_metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
