import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.constants import MEMORY, DEVICE, BATCH_SIZE, NUM_CLASSES, LEARNING_RATE, MELODY_NOTES_PER_BEAT

os.makedirs("checkpoints", exist_ok=True)

# --------------------------
# Data loading
# --------------------------
def load_data_from_npz(path):
    data = np.load(path)
    X = data['inputs'].astype(np.float32)
    y = data['targets'].astype(np.int64)
    return X, y

X_train, y_train = load_data_from_npz("data/data_train_smooth.npz")
X_val, y_val     = load_data_from_npz("data/data_val_smooth.npz")

class MusicDataset(Dataset):
    def __init__(self, X, Y):
        self.melody = X.astype(np.float32)
        self.chords = Y.astype(np.int64).squeeze(-1)
    def __len__(self):
        return len(self.melody)
    def __getitem__(self, idx):
        return torch.tensor(self.melody[idx], dtype=torch.float32), torch.tensor(self.chords[idx], dtype=torch.long)

train_loader = DataLoader(MusicDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(MusicDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

# --------------------------
# Positional Encoding
# --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MEMORY):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    def forward(self, x):
        return x + self.pe.to(x.device)

# --------------------------
# Transformer Model
# --------------------------
class ChordTransformer(nn.Module):
    def __init__(self, num_chords=NUM_CLASSES, feature_dim=1+MELODY_NOTES_PER_BEAT, d_model=64, nhead=2, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        # Encoder
        self.encoder_input = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Decoder
        self.chord_emb = nn.Embedding(num_chords, d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.output_layer = nn.Linear(d_model, num_chords)
        self.d_model = d_model

    def forward(self, melody, chords=None, teacher_forcing_ratio=1.0):
        """
        melody: (B, T, feature_dim)
        chords: (B, T) integers (previous chords, 0=start token)
        teacher_forcing_ratio: probability to use ground-truth token instead of prediction
        """
        B, T, _ = melody.shape
        enc = self.encoder_input(melody) * np.sqrt(self.d_model)
        enc = self.pos_encoder(enc)
        enc = self.encoder(enc.transpose(0,1))  # (T, B, d_model)

        # Decoder input initialization
        outputs = torch.zeros(B, T, NUM_CLASSES, device=melody.device)
        input_tokens = torch.zeros(B, dtype=torch.long, device=melody.device)  # start token

        for t in range(T):
            tgt_emb = self.chord_emb(input_tokens).unsqueeze(1) * np.sqrt(self.d_model)  # (B,1,d_model)
            tgt_emb = self.pos_decoder(tgt_emb)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(melody.device)
            dec_out = self.decoder(tgt_emb.transpose(0,1), enc, tgt_mask=tgt_mask)
            logits = self.output_layer(dec_out.transpose(0,1)[:,0,:])  # (B, num_chords)
            outputs[:,t,:] = logits

            # Scheduled sampling
            if chords is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_tokens = chords[:,t]
            else:
                input_tokens = logits.argmax(dim=-1)

        return outputs

# --------------------------
# Checkpoint functions
# --------------------------
def save_checkpoint(model, optimizer, epoch, batch_idx, loss, save_path='checkpoints/full.pth'):
    torch.save({
        'epoch': epoch,
        'batch': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path)

def load_checkpoint(model, optimizer=None, save_path='checkpoints/full.pth', device='cuda'):
    if not os.path.exists(save_path):
        print(f"No checkpoint found at {save_path}, starting fresh.")
        return model, optimizer, 0
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {save_path}")
    return model, optimizer, checkpoint['epoch']

# --------------------------
# Training loop
# --------------------------
def train_model():
    model = ChordTransformer().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    EPOCHS = 20

    model, optimizer, start_epoch = load_checkpoint(model, optimizer, save_path="checkpoints/full.pth", device=DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    for epoch in range(start_epoch+1, EPOCHS+1):
        model.train()
        total_loss = 0
        # progressively reduce teacher forcing
        teacher_forcing_ratio = max(0.5, 1.0 - epoch*0.05)

        for batch_idx, (melody, chords) in enumerate(train_loader, start=1):
            melody, chords = melody.to(DEVICE), chords.to(DEVICE)
            optimizer.zero_grad()
            output = model(melody, chords, teacher_forcing_ratio)
            loss = criterion(output.view(-1, NUM_CLASSES), chords.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for melody, chords in val_loader:
                melody, chords = melody.to(DEVICE), chords.to(DEVICE)
                output = model(melody, chords, teacher_forcing_ratio=teacher_forcing_ratio)
                loss = criterion(output.view(-1, NUM_CLASSES), chords.view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch} | Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        save_checkpoint(model, optimizer, epoch, batch_idx, loss, save_path="checkpoints/full.pth")

    writer.close()

if __name__ == "__main__":
    train_model()
