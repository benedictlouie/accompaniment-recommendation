import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from constants import DEVICE, CHORD_CLASSES, NUM_CLASSES, MELODY_NOTES_PER_BEAT, CHORD_TO_TETRAD, INPUT_DIM, CHORD_EMBEDDING_LENGTH, LEARNING_RATE, DROPOUT_RATE, WEIGHT_DECAY, LEARNING_RATE, DROPOUT_RATE, WEIGHT_DECAY, BATCH_SIZE
from FifthsCircleLoss import FifthsCircleLoss
from torch.utils.tensorboard import SummaryWriter

# -------------------------------------------------------
#               MODEL DEFINITION (GRU)
# -------------------------------------------------------

class SequenceToChordGRU(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, num_layers=2, num_classes=NUM_CLASSES, dropout=DROPOUT_RATE):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=DROPOUT_RATE)
        self.fc = nn.Linear(hidden_dim, num_classes)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{total_params:,} trainable parameters")

    def forward(self, x):
        # x: [B, MEMORY, input_dim]
        out, _ = self.gru(x)  # out: [B, MEMORY, hidden_dim]
        out = out[:, -1, :]   # take the last timestep
        logits = self.fc(out)
        return logits

# -------------------------------------------------------
#               DATASET & HELPERS
# -------------------------------------------------------

class ChordDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def load_data_from_npz(npz_path):
    data = np.load(npz_path)
    inputs = torch.tensor(data['inputs'], dtype=torch.float32)
    targets = torch.tensor(data['targets'].squeeze(), dtype=torch.long)
    print(f"Loaded {len(inputs)} samples from {npz_path}")
    return inputs, targets

# -------------------------------------------------------
#               CHECKPOINTING
# -------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, batch_idx, loss, save_path='checkpoints/gru.pth'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'batch': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path)

def load_checkpoint(model, optimizer=None, save_path='checkpoints/gru.pth', device='cuda'):
    if not os.path.exists(save_path):
        print(f"No checkpoint found at {save_path}, starting fresh.")
        return model, optimizer, 0, 0
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {save_path} (epoch {checkpoint['epoch']}, batch {checkpoint['batch']})")
    return model, optimizer, checkpoint['epoch'], checkpoint['batch']

# -------------------------------------------------------
#               CORRUPTION LOGIC
# -------------------------------------------------------

def corrupt_history_with_model(x, model, K, wrong_prob, device):
    if torch.rand(1).item() > wrong_prob:
        return x

    batch, timestamp, feature = x.shape
    assert timestamp > K, "T must be greater than K"

    prefix = x[:, :-K, :].to(device)
    pad = torch.zeros((batch, K, feature), device=device)
    pad[:, :, 1:1+MELODY_NOTES_PER_BEAT] = -1
    pad[:, :, -1] = NUM_CLASSES - 1
    model_input = torch.cat([prefix, pad], dim=1)

    model.eval()
    with torch.no_grad():
        logits = model(model_input)
        preds = logits.argmax(dim=-1)  # [B]

    x_corr = x.clone().to(device)
    prev_chords = [CHORD_CLASSES[pred] for pred in preds]
    for i, chord in enumerate(prev_chords):
        tetrads = np.array(CHORD_TO_TETRAD[chord]).reshape(1, -1).repeat(K, axis=0)
        x_corr[i, -K:, -CHORD_EMBEDDING_LENGTH:] = torch.from_numpy(tetrads).to(DEVICE, dtype=x_corr.dtype)
    
    return x_corr

# -------------------------------------------------------
#               TRAINING / EVALUATION
# -------------------------------------------------------

def evaluate(model, dataloader, criterion, device='cuda'):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
    return total_loss / len(dataloader), total_correct / total_samples

def train_model(model, train_dataset, val_dataset, num_epochs=30, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                device='cuda', checkpoint_path='checkpoints/gru.pth'):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    criterion = FifthsCircleLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=LEARNING_RATE/10)  # LR schedule

    model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, save_path=checkpoint_path, device=device)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train_loss, total_correct, total_samples = 0, 0, 0

        # Corruption schedule: starts at 0, ramps to max_wrong_prob gradually
        max_wrong_prob = 0.9
        wrong_history_prob = max_wrong_prob * (epoch / (num_epochs - 1))

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            x_corr = corrupt_history_with_model(x, model, K=4, wrong_prob=wrong_history_prob, device=device)

            optimizer.zero_grad()
            logits = model(x_corr)
            loss = criterion(logits, y)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_train_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

            if (batch_idx + 1) % 500 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        scheduler.step()  # update LR

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = total_correct / total_samples
        val_loss, val_acc = evaluate(model, val_loader, criterion, device=device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        save_checkpoint(model, optimizer, epoch, 0, avg_train_loss, save_path=checkpoint_path)


# -------------------------------------------------------
#               MAIN ENTRY
# -------------------------------------------------------

if __name__ == "__main__":

    writer = SummaryWriter()

    train_inputs, train_targets = load_data_from_npz('data_train.npz')
    val_inputs, val_targets = load_data_from_npz('data_val.npz')

    train_dataset = ChordDataset(train_inputs, train_targets)
    val_dataset = ChordDataset(val_inputs, val_targets)

    model = SequenceToChordGRU(input_dim=INPUT_DIM)
    train_model(model, train_dataset, val_dataset, num_epochs=30, batch_size=BATCH_SIZE, device=DEVICE)
