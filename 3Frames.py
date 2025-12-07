import os
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageDraw

# =========================
# CONFIG
# =========================

DATA_ROOT = Path(r"C:\Users\sebas\King's College London\Oya Celiktutan - SAP Final Project -Dataset")

TRAIN_CSV = DATA_ROOT / "annotations" / "train_set_labels.csv"
TRAIN_VIDEO_DIR = DATA_ROOT / "train_set"
TEST_VIDEO_DIR = DATA_ROOT / "test_set"

BATCH_SIZE = 8          # a bit smaller, because 3 frames per sample
NUM_EPOCHS = 17
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
RANDOM_SEED = 42

NUM_ACTION_CLASSES = 30

ACTION_LABELS = [
    "DeliverObject",                          # 1
    "MoveBackwardsWhileDrilling",            # 2
    "MoveBackwardsWhilePolishing",           # 3
    "MoveDiagonallyBackwardLeftWithDrill",   # 4
    "MoveDiagonallyBackwardLeftWithPolisher",# 5
    "MoveDiagonallyBackwardRightWithDrill",  # 6
    "MoveDiagonallyBackwardRightWithPolisher",#7
    "MoveDiagonallyForwardLeftWithDrill",    # 8
    "MoveDiagonallyForwardLeftWithPolisher", # 9
    "MoveDiagonallyForwardRightWithDrill",   # 10
    "MoveDiagonallyForwardRightWithPolisher",# 11
    "MoveForwardWhileDrilling",              # 12
    "MoveForwardWhilePolishing",             # 13
    "MoveLeftWhileDrilling",                 # 14
    "MoveLeftWhilePolishing",                # 15
    "MoveRightWhileDrilling",                # 16
    "MoveRightWhilePolishing",               # 17
    "NoCollaborativeWithDrill",              # 18
    "NoCollaborativeWithPolisher",           # 19
    "PickUpDrill",                           # 20
    "PickUpPolisher",                        # 21
    "PickUpTheObject",                       # 22
    "PutDownDrill",                          # 23
    "PutDownPolisher",                       # 24
    "UsingTheDrill",                         # 25
    "UsingThePolisher",                      # 26
    "Walking",                               # 27
    "WalkingWithObject",                     # 28
    "WalkingWithDrill",                      # 29
    "WalkingWithPolisher"                    # 30
]


# =========================
# DATASET: 3 frames per video
# =========================

class ActionDataset3Frames(Dataset):
    """
    One sample = one video.
    We load THREE frames: first, middle, last.
    """

    def __init__(self, df: pd.DataFrame, video_root: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.video_root = video_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _load_three_frames(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Video has no frames: {video_path}")

        # choose indices
        first_idx = 0
        mid_idx = frame_count // 2
        last_idx = frame_count - 1

        frames = []
        for idx in [first_idx, mid_idx, last_idx]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                raise RuntimeError(f"Could not read frame {idx} from: {video_path}")
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        return frames  # list of 3 HxWx3 arrays

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row["video_id"]
        label_id = int(row["class_id"]) - 1  # 0..29

        video_path = self.video_root / f"{video_id}.avi"
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        frames = self._load_three_frames(video_path)

        processed_frames = []
        for f in frames:
            if self.transform is not None:
                img = self.transform(f)
            else:
                img = transforms.ToTensor()(f)
            processed_frames.append(img)

        # shape: (3, C, H, W)
        frames_tensor = torch.stack(processed_frames, dim=0)

        return frames_tensor, label_id


# =========================
# UTILS
# =========================

def load_train_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["video_id", "class_name", "class_id"],
    )
    print("[INFO] Loaded train CSV with shape:", df.shape)
    print(df.head())
    return df


def make_splits(df: pd.DataFrame, val_split=0.2, seed=42):
    np.random.seed(seed)
    perm = np.random.permutation(len(df))
    val_size = int(len(df) * val_split)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    print(f"[INFO] Train samples: {len(df_train)}, Val samples: {len(df_val)}")
    return df_train, df_val


# =========================
# TRAINING (3-frame averaging)
# =========================

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    df_full = load_train_dataframe(TRAIN_CSV)
    df_train, df_val = make_splits(df_full, val_split=VAL_SPLIT, seed=RANDOM_SEED)

    # transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = ActionDataset3Frames(df_train, TRAIN_VIDEO_DIR, transform=train_transform)
    val_dataset   = ActionDataset3Frames(df_val,   TRAIN_VIDEO_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_ACTION_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")

        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for frames, labels in train_loader:
            # frames: (B, 3, C, H, W)
            bsz, num_f, C, H, W = frames.shape
            frames = frames.to(device)          # (B,3,C,H,W)
            labels = labels.to(device)          # (B,)

            # reshape to treat each frame as separate sample
            frames_flat = frames.view(bsz * num_f, C, H, W)   # (B*3,C,H,W)

            optimizer.zero_grad()
            outputs_flat = model(frames_flat)                 # (B*3, num_classes)

            # reshape back and average over 3 frames
            outputs = outputs_flat.view(bsz, num_f, NUM_ACTION_CLASSES).mean(dim=1)  # (B,num_classes)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bsz
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += bsz

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for frames, labels in val_loader:
                bsz, num_f, C, H, W = frames.shape
                frames = frames.to(device)
                labels = labels.to(device)

                frames_flat = frames.view(bsz * num_f, C, H, W)
                outputs_flat = model(frames_flat)
                outputs = outputs_flat.view(bsz, num_f, NUM_ACTION_CLASSES).mean(dim=1)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * bsz
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += bsz

        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    out_path = Path("action_model_resnet18_3frame.pth")
    torch.save(model.state_dict(), out_path)
    print("[INFO] Saved model to", out_path)

    return model


# =========================
# TEST PREDICTION (3-frame averaging)
# =========================

def predict_on_test_set(model_path="action_model_resnet18_3frame.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Loading model from:", model_path)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_ACTION_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    def load_three_frames(video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Video has no frames: {video_path}")

        # choose indices at ~20%, 50%, 80% of the video
        idx1 = int(frame_count * 0.2)
        idx2 = int(frame_count * 0.5)
        idx3 = int(frame_count * 0.8)

        indices = [
            max(0, min(frame_count - 1, idx1)),
            max(0, min(frame_count - 1, idx2)),
            max(0, min(frame_count - 1, idx3)),
        ]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                raise RuntimeError(f"Could not read frame {idx} from {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        return frames

    results = []

    for vid_file in sorted(TEST_VIDEO_DIR.iterdir()):
        if vid_file.suffix.lower() != ".avi":
            continue

        try:
            frames = load_three_frames(vid_file)
        except Exception as e:
            print(f"[WARN] Skipping {vid_file.name}: {e}")
            continue

        processed = []
        for f in frames:
            x = val_transform(f)
            processed.append(x)

        frames_tensor = torch.stack(processed, dim=0)  # (3,C,H,W)
        frames_tensor = frames_tensor.unsqueeze(0).to(device)  # (1,3,C,H,W)

        with torch.no_grad():
            bsz, num_f, C, H, W = frames_tensor.shape
            frames_flat = frames_tensor.view(bsz * num_f, C, H, W)
            outputs_flat = model(frames_flat)
            outputs = outputs_flat.view(bsz, num_f, NUM_ACTION_CLASSES).mean(dim=1)
            _, pred = torch.max(outputs, 1)

            label_idx = int(pred.item())   # 0..29
            class_id = label_idx + 1       # 1..30
            class_name = ACTION_LABELS[label_idx]

        results.append((vid_file.stem, class_id, class_name))
        print(f"{vid_file.stem}: {class_id} → {class_name}")

    out_csv = Path("test_action_predictions_3frame.csv")
    pd.DataFrame(
        results,
        columns=["video_id", "predicted_class_id", "predicted_class_name"]
    ).to_csv(out_csv, index=False)
    print("[INFO] Saved predictions to", out_csv)


# =========================
# OPTIONAL PREVIEW (uses middle frame)
# =========================

def generate_action_preview(predictions_csv="test_action_predictions_3frame.csv",
                            output_dir="test_action_preview_3frame"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(predictions_csv)

    for _, row in df.iterrows():
        video_id = row["video_id"]
        class_name = row["predicted_class_name"]

        video_path = TEST_VIDEO_DIR / f"{video_id}.avi"
        if not video_path.exists():
            print(f"[WARN] Missing video: {video_id}.avi")
            continue

        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"[WARN] Could not read frame from {video_path}")
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        draw = ImageDraw.Draw(img)
        text = f"{video_id} → {class_name}"
        draw.text((10, 10), text, fill="yellow")

        img.save(output_dir / f"{video_id}.jpg")

    print(f"[INFO] Previews saved to {output_dir}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    model = train_model()
    predict_on_test_set("action_model_resnet18_3frame.pth")
    generate_action_preview("test_action_predictions_3frame.csv")
