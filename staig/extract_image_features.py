"""Extract image embeddings from a Space Ranger Visium directory.

The main entry point is :func:`extract_image_features`, which can be imported
by another Python module without starting any processing at import time.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import scanpy as sc
import torch
import torchvision.transforms as T
from byol_pytorch import BYOL
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


PathLike = Union[str, os.PathLike[str]]
PATCH_SIZE = int(3.5 * 144)
TRAINING_EPOCHS = 1


def create_custom_mask(
    image_shape: tuple[int, int],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """Create a rectangular frequency-domain mask."""
    rows, cols = image_shape
    mask = np.zeros((rows, cols), np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def process_image(
    filename: str,
    path: PathLike,
    output_path: PathLike,
    gaussian_blur: bool,
    lower: int,
    upper: int,
) -> np.ndarray:
    """Filter one patch and save the filtered patch."""
    image_path = Path(path) / filename
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image patch: {image_path}")

    original_height, original_width = image.shape[:2]
    resize_needed = (original_height != 512 or original_width != 512)
    if resize_needed:
        image = cv2.resize(image, (512, 512))

    if gaussian_blur:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    shifted = np.fft.fftshift(np.fft.fft2(image))
    mask = create_custom_mask(image.shape, lower, lower, upper, upper)
    filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(shifted * mask)))

    if gaussian_blur:
        filtered = cv2.GaussianBlur(filtered, (15, 15), 0)

    filtered_rgb = cv2.cvtColor(np.float32(filtered), cv2.COLOR_GRAY2RGB)
    mae_patch = cv2.resize(filtered_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    if resize_needed:
        filtered_rgb = cv2.resize(filtered_rgb, (original_width, original_height))

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path / filename), filtered_rgb):
        raise OSError(f"Could not write filtered image patch: {output_path / filename}")
    return mae_patch


class ImagePatchDataset(Dataset):
    """Dataset for numbered PNG patches."""

    def __init__(self, root_dir: PathLike, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_list = sorted(
            self.root_dir.glob("*.png"),
            key=lambda path: int(path.stem),
        )

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int):
        image = Image.open(self.image_list[index]).convert("RGB")
        return self.transform(image) if self.transform else image


def _set_deterministic_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


def _find_tissue_image(input_dir: Path) -> Path:
    spatial_dir = input_dir / "spatial"
    candidates = sorted(
        path
        for path in spatial_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not candidates:
        raise FileNotFoundError(f"No TIFF tissue image found in {spatial_dir}")
    full_image = [path for path in candidates if "fullres" in path.stem.lower()]
    return full_image[0] if full_image else candidates[0]


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _build_augmentations() -> torch.nn.Sequential:
    class RandomApply(nn.Module):
        def __init__(self, fn, probability: float):
            super().__init__()
            self.fn = fn
            self.probability = probability

        def forward(self, tensor):
            return self.fn(tensor) if random.random() <= self.probability else tensor

    return torch.nn.Sequential(
        RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), probability=0.3),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), probability=0.2),
        T.RandomRotation(degrees=(0, 360)),
        T.RandomResizedCrop((256, 256)),
        T.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32),
            std=torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32),
        ),
    )


def extract_image_features(
    input_dir: PathLike,
    output_dir: PathLike,
    work_dir: PathLike,
    img_enbedding_dir: PathLike,
) -> np.ndarray:
    """Extract and save image embeddings for a Visium dataset.

    Args:
        input_dir: Space Ranger directory containing the Visium files.
        output_dir: Directory where the trained BYOL checkpoint is stored.
        work_dir: Directory for temporary extracted and filtered patches.
        img_enbedding_dir: Directory where ``embeddings.npy`` is stored.

    Returns:
        The extracted embeddings as an ``(n_patches, embedding_size)`` array.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    work_dir = Path(work_dir)
    img_enbedding_dir = Path(img_enbedding_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    img_enbedding_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = work_dir / "clip_image"
    filtered_patch_dir = work_dir / "clip_image_filter"
    _reset_directory(patch_dir)
    _reset_directory(filtered_patch_dir)

    adata = sc.read_visium(
        path=str(input_dir),
        count_file="filtered_feature_bc_matrix.h5",
        load_images=True,
    )
    adata.var_names_make_unique()

    image = cv2.imread(str(_find_tissue_image(input_dir)), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError("The Visium TIFF could not be read.")

    for index, coord in tqdm(
        enumerate(adata.obsm["spatial"]),
        total=len(adata.obsm["spatial"]),
        desc="Extracting image patches",
    ):
        left = int(coord[0] - PATCH_SIZE / 2)
        top = int(coord[1] - PATCH_SIZE / 2)
        patch = image[top : top + PATCH_SIZE, left : left + PATCH_SIZE]
        if patch.size == 0:
            raise ValueError(f"Empty image patch at spatial coordinate {coord}.")
        resized_patch = cv2.resize(patch, (512, 512), interpolation=cv2.INTER_LINEAR)
        if not cv2.imwrite(str(patch_dir / f"{index}.png"), resized_patch):
            raise OSError(f"Could not write image patch {index}.")

    png_files = [path.name for path in patch_dir.glob("*.png")]
    with Pool(processes=os.cpu_count() or 1) as pool:
        pool.starmap(
            process_image,
            [(name, patch_dir, filtered_patch_dir, True, 245, 275) for name in png_files],
        )

    _set_deterministic_seed()
    dataset = ImagePatchDataset(
        filtered_patch_dir,
        transform=transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
        ),
    )
    if not len(dataset):
        raise ValueError("No image patches were created.")
    data_loader = DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augmentations = _build_augmentations()
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    if torch.cuda.device_count() > 1:
        backbone = nn.DataParallel(backbone)
        hidden_layer = "module.avgpool"
    else:
        hidden_layer = "avgpool"

    learner = BYOL(
        backbone,
        image_size=256,
        hidden_layer=hidden_layer,
        augment_fn=augmentations,
    ).to(device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)
    for _ in range(TRAINING_EPOCHS):
        for images in tqdm(data_loader, desc="Training image encoder", leave=False):
            loss = learner(images.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()

    checkpoint_path = output_dir / "learner.pth"
    clean_state_dict = {
        key.replace(".module.", "."): value for key, value in learner.state_dict().items()
    }
    torch.save(clean_state_dict, checkpoint_path)
    del learner, backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_backbone = models.resnet50(weights=None).to(device)
    eval_learner = BYOL(
        eval_backbone,
        image_size=256,
        hidden_layer="avgpool",
        augment_fn=augmentations,
    ).to(device)
    eval_learner.load_state_dict(torch.load(checkpoint_path, map_location=device))
    eval_learner.eval()
    eval_loader = DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
    )
    embeddings = []
    with torch.no_grad():
        for image_batch in tqdm(eval_loader, desc="Extracting image embeddings"):
            _, embedding = eval_learner(
                image_batch.to(device), return_embedding=True
            )
            embeddings.append(embedding.cpu().numpy())

    result = np.vstack(embeddings)
    np.save(img_enbedding_dir / "embeddings.npy", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("work_dir")
    parser.add_argument("img_enbedding_dir")
    args = parser.parse_args()
    extract_image_features(
        args.input_dir,
        args.output_dir,
        args.work_dir,
        args.img_enbedding_dir,
    )


if __name__ == "__main__":
    main()
