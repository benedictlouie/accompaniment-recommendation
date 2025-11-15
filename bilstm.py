import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from constants import MELODY_NOTES_PER_BEAT, NUM_CLASSES, LEARNING_RATE, DROPOUT_RATE, WEIGHT_DECAY, BATCH_SIZE
from torch.utils.tensorboard import SummaryWriter
import os

# ---------------------------
# 1. Load data
# ---------------------------
def load_data_from_npz(path):
    data = np.load(path)
    return data["inputs"], data["targets"]  # adjust keys if different

# ---------------------------
# 2. Dataset class
# ---------------------------
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # integer class labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------
# 3. Bidirectional LSTM model
# ---------------------------
class BiLSTMSeq2Seq(nn.Module):
    def __init__(self, input_dim=1+MELODY_NOTES_PER_BEAT, hidden_dim=64, num_layers=2, num_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT_RATE
        )
        self.fc = nn.Linear(hidden_dim*2, num_classes)  # output logits for each class

    def forward(self, x):
        out, _ = self.lstm(x)      # (B, 32, hidden*2)
        out = self.fc(out)         # (B, 32, num_classes)
        return out

# ---------------------------
# 4. Count parameters
# ---------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------
# 5. Save checkpoint
# ---------------------------
def save_checkpoint(model, optimizer, epoch, path="checkpoints/bilstm.pth"):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, path)
    print(f"Checkpoint saved at epoch {epoch} -> {path}")

# ---------------------------
# 6. Load checkpoint
# ---------------------------
def load_checkpoint(model, optimizer, path="checkpoints/bilstm.pth"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No checkpoint found. Starting from scratch.")
    return start_epoch

# ---------------------------
# 7. Training function
# ---------------------------
def train_model(train_path, val_path, num_classes, epochs=10, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                hidden_dim=64, num_layers=2, checkpoint_path="checkpoints/bilstm.pth"):
    # Load data
    X_train, y_train = load_data_from_npz(train_path)
    X_val, y_val     = load_data_from_npz(val_path)

    train_ds = SeqDataset(X_train, y_train)
    val_ds   = SeqDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, loss, optimizer
    model = BiLSTMSeq2Seq(input_dim=X_train.shape[2], hidden_dim=hidden_dim,
                          num_layers=num_layers, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    print("Number of trainable parameters:", count_parameters(model))

    # Resume from checkpoint if exists
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    # Training loop
    for epoch in range(start_epoch, epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        for batch_idx, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)  # y: (B, 32)

            optimizer.zero_grad()
            y_pred = model(input)                  # y_pred: (B, 32, num_classes)
            loss = criterion(y_pred.view(-1, num_classes), target.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # optional gradient clipping
            optimizer.step()
            train_loss += loss.item()

            if (batch_idx + 1) % 500 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for input, target in val_loader:
                input, target = input.to(device), target.to(device)
                y_pred = model(input)
                loss = criterion(y_pred.view(-1, num_classes), target.view(-1))
                val_loss += loss.item()

                predicted = y_pred.argmax(dim=-1)  # (B,32)
                correct += (predicted == target).sum().item()
                total += target.numel()

        acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Acc: {acc:.4f}")
        
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", acc, epoch)

        save_checkpoint(model, optimizer, epoch, checkpoint_path)

    return model

# ---------------------------
# 8. Example usage
# ---------------------------
if __name__ == "__main__":

    writer = SummaryWriter()

    model = train_model(
        train_path="data_train_smooth.npz",
        val_path="data_val_smooth.npz",
        num_classes=NUM_CLASSES,
        epochs=20,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        hidden_dim=64,
        num_layers=2
    )
