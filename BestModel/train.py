import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from sklearn.utils import resample

# ======================
# CONFIG
# ======================
MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
OVERSAMPLE_FACTOR = 2.5
DATA_DIR = "data"
CHECKPOINT_DIR = "BestModel/checkpoint_stage2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ======================
# LOAD DATA
# ======================
print("Loading dataset...")

df = pd.read_csv(
    os.path.join(DATA_DIR, "dontpatronizeme_pcl.tsv"),
    sep="\t",
    skiprows=4,
    header=None,
    names=["par_id","art_id","keyword","country","text","orig_label"]
)

df["orig_label"] = pd.to_numeric(df["orig_label"], errors="coerce").fillna(0).astype(int)
df["label"] = df["orig_label"].apply(lambda x: 0 if x in (0,1) else 1)
df["text"] = df["text"].astype(str)

train_ids = set(pd.read_csv(os.path.join(DATA_DIR, "train_semeval_parids-labels.csv"))["par_id"])
dev_ids = set(pd.read_csv(os.path.join(DATA_DIR, "dev_semeval_parids-labels.csv"))["par_id"])

train_df = df[df["par_id"].isin(train_ids)].reset_index(drop=True)
dev_df = df[df["par_id"].isin(dev_ids)].reset_index(drop=True)

print("Train:", len(train_df))
print("Dev:", len(dev_df))

# ======================
# OVERSAMPLING (Stage 2)
# ======================
pcl = train_df[train_df["label"] == 1]
extra = resample(
    pcl,
    replace=True,
    n_samples=int(len(pcl)*(OVERSAMPLE_FACTOR-1)),
    random_state=42
)
train_df = pd.concat([train_df, extra]).sample(frac=1, random_state=42).reset_index(drop=True)

print("After oversampling:", len(train_df))

# ======================
# DATASET
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class PCLDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k,v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = PCLDataset(train_df["text"].tolist(), train_df["label"].tolist())
dev_dataset = PCLDataset(dev_df["text"].tolist(), dev_df["label"].tolist())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

# ======================
# MODEL
# ======================
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

# Class weights
counts = train_df["label"].value_counts().sort_index()
n0, n1 = counts[0], counts[1]
weight_0 = 1.0
weight_1 = n0 / n1
class_weights = torch.tensor([weight_0, weight_1]).float().to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ======================
# TRAIN LOOP
# ======================
print("\nStarting training...")

best_f1 = 0
best_threshold = 0.5

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = {k: v.to(DEVICE) for k,v in batch.items()}

        outputs = model(**batch)
        loss = criterion(outputs.logits, batch["labels"])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"\nEpoch {epoch+1}")
    print("Train Loss:", total_loss / len(train_loader))

    # ======================
    # VALIDATION
    # ======================
    model.eval()
    probs = []
    true = []

    with torch.no_grad():
        for batch in dev_loader:
            batch = {k: v.to(DEVICE) for k,v in batch.items()}
            outputs = model(**batch)
            p = torch.softmax(outputs.logits, dim=1)[:,1]
            probs.extend(p.cpu().numpy())
            true.extend(batch["labels"].cpu().numpy())

    # Threshold tuning
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (np.array(probs) >= t).astype(int)
        f1 = f1_score(true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))

    print("Best Dev F1 so far:", best_f1)
    print("Best Threshold so far:", best_threshold)

print("\nTraining complete.")
print("Final Best Dev F1:", best_f1)

# ======================
# LOAD BEST MODEL
# ======================
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt")))
model.eval()

# ======================
# WRITE DEV.TXT
# ======================
final_preds = (np.array(probs) >= best_threshold).astype(int)

with open("BestModel/dev.txt", "w") as f:
    for p in final_preds:
        f.write(str(p) + "\n")

print("dev.txt written.")

# ======================
# TEST PREDICTIONS
# ======================
print("Generating test predictions...")

test_df = pd.read_csv(
    os.path.join(DATA_DIR, "PCL_test.tsv"),
    sep="\t",
    skiprows=4,
    header=None,
    names=["par_id","art_id","keyword","country","text"]
)

test_dataset = PCLDataset(test_df["text"].astype(str).tolist())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

test_probs = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(DEVICE) for k,v in batch.items()}
        outputs = model(**batch)
        p = torch.softmax(outputs.logits, dim=1)[:,1]
        test_probs.extend(p.cpu().numpy())

test_preds = (np.array(test_probs) >= best_threshold).astype(int)

with open("BestModel/test.txt", "w") as f:
    for p in test_preds:
        f.write(str(p) + "\n")

print("test.txt written.")
print("Done.")