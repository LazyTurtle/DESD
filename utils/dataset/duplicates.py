"""
Near-duplicate rate for an image dataset, computed two ways:
  1. Perceptual hash (pHash)  -> near-pixel-identical / literal-copy redundancy
  2. DINOv2 embeddings        -> structural/semantic redundancy (robust to
                                  lighting, minor viewpoint, texture changes)

Install:
    pip install torch torchvision transformers pillow imagehash faiss-cpu scipy numpy tqdm

Usage:
    Edit IMAGE_DIR, HASH_THRESH, and DINO_THRESH below (see the note on
    calibrating thresholds via the nearest-neighbor histogram method), then:
        python near_duplicate_rate.py
"""

import argparse
import faiss
import imagehash
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from transformers import AutoImageProcessor, AutoModel, logging

# ----------------------------------------------------------------------
# Config — calibrate HASH_THRESH / DINO_THRESH using the nearest-neighbor
# histogram approach (look at each image's distance/similarity to its
# single nearest neighbor; pick the threshold in the valley of the
# resulting bimodal distribution) before trusting the reported rates.
# ----------------------------------------------------------------------
IMAGE_DIR = "path/to/your/images"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")

HASH_SIZE = 8          # 16 for 256-bit hash; use 8 (64-bit) for smaller/less homogeneous datasets
HASH_THRESH = 10        # max Hamming distance (out of HASH_SIZE**2 bits) to count as near-dup

DINO_MODEL = "facebook/dinov2-base"   # or dinov2-small / dinov2-large / dinov2-giant
DINO_THRESH = 0.95      # min cosine similarity to count as near-duplicate
BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def near_dup_rate(values, idx, n, threshold, is_distance, verbose=True):
    """Shared rate definition for both representations.

    is_distance=True  -> a pair counts as a hit if value <= threshold (Hamming distance)
    is_distance=False -> a pair counts as a hit if value >= threshold (cosine similarity)

    Builds a graph over hits and returns 1 - (connected components / n): the
    fraction of images that are "extra" beyond one representative per
    near-duplicate group.
    """
    rows, cols = [], []
    for i in tqdm(range(n), disable=not verbose):
        for pos in range(1, idx.shape[1]):          # skip self-match at pos 0
            j, v = idx[i, pos], values[i, pos]
            hit = (v <= threshold) if is_distance else (v >= threshold)
            if hit:
                rows.append(i)
                cols.append(j)
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_clusters, _ = connected_components(graph, directed=False)
    return 1 - n_clusters / n


def compute_hash_rate(image_paths:list[Path], hash_size, threshold, verbose=True):
    
    hashes = list()
    for image in tqdm(image_paths, 'Calculate hashes', disable=not verbose):
        img = Image.open(image)        
        hash = imagehash.phash(img.convert("RGB"), hash_size=hash_size).hash.flatten()
        hashes.append(hash)
    
    bits = np.array(hashes).astype(np.uint8)
    packed = np.packbits(bits, axis=1)
    index = faiss.IndexBinaryFlat(bits.shape[1])
    index.add(packed)
    dist, pos = index.search(packed, 11)
    return near_dup_rate(dist, pos, len(image_paths), threshold, is_distance=True, verbose=verbose)


@torch.no_grad()
def extract_dino(image_paths, model_name, batch_size, device, verbose = True):
    if not verbose:
        logging.disable_progress_bar()

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size), disable=not verbose):
        batch = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch_size]]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        cls_tokens = model(**inputs).last_hidden_state[:, 0]      # CLS token = global descriptor
        cls_tokens = torch.nn.functional.normalize(cls_tokens, dim=-1)
        embeddings.append(cls_tokens.cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype("float32")


def compute_dino_rate(image_paths, model_name, threshold, batch_size, device, verbose=True):
    emb = extract_dino(image_paths, model_name, batch_size, device, verbose)
    index = faiss.IndexFlatIP(emb.shape[1])       # inner product = cosine sim (embeddings normalized)
    index.add(emb)
    sims, pos = index.search(emb, 11)
    return near_dup_rate(sims, pos, len(image_paths), threshold, is_distance=False, verbose=verbose)

def compute_near_duplicates(dataset:str|Path, verbose = True):
    dataset = Path(dataset)
    images_folder = dataset / 'images'
    images = images_folder.glob('*.*')
    images = [i for i in images if i.suffix.lower() in IMAGE_EXTS]

    if verbose:
        print(f"{dataset.stem}: Found {len(images)} images")

    rate_hash = compute_hash_rate(
        image_paths = images,
        hash_size = HASH_SIZE,
        threshold= HASH_THRESH,
        verbose=verbose)
    if verbose:
        print(f"hash near-duplicate rate: {rate_hash:.1%}")

    rate_dino = compute_dino_rate(images, DINO_MODEL, DINO_THRESH, BATCH_SIZE, DEVICE, verbose)
    if verbose:
        print(f"DINO near-duplicate rate: {rate_dino:.1%}")

    return (rate_hash, rate_dino)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, default=None, help="Root directory of a yolo dataset.")
    args = parser.parse_args()
    compute_near_duplicates(args.dataset)
