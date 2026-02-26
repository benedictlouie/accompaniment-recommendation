import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Subset
from utils.constants import DEVICE, INPUT_DIM, MEMORY, LEARNING_RATE, NUM_CLASSES_ALL, TEMPERATURE

# -------------------------
# Constants
# -------------------------
OUTPUT_DIM = NUM_CLASSES_ALL
MAX_LEN = MEMORY + 1

# -------------------------
# Transformer Model (Parallel Teacher Forcing)
# -------------------------
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim

        # Embeddings
        self.feature_to_embedding = nn.Linear(input_dim, d_model)
        self.embedding_output = nn.Embedding(output_dim, d_model) 

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, MEMORY, d_model))

        # Encoder & Decoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_decoder_layers
        )

        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, input_seq, target_seq=None):
        """
        - Training: provide target_seq → teacher forcing
        - Inference: target_seq=None → autoregressive generation
        """

        B, device = input_seq.size(0), input_seq.device

        # ---- Encoder ----
        src_emb = self.feature_to_embedding(
            torch.where(input_seq > 10, input_seq % 12, input_seq)
        ) + self.pos_encoder[:, :input_seq.size(1), :]
        memory = self.encoder(src_emb)

        if target_seq is not None:
            # ---- Teacher forcing (training) ----
            if target_seq.dim() == 3:
                target_seq = target_seq[:, :, 0]

            T = target_seq.size(1)
            tgt_emb = self.embedding_output(target_seq)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
            out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            logits = self.fc_out(out)
            return logits  # shape (B, T, OUTPUT_DIM)

        else:
            # ---- Autoregressive generation (inference) ----
            output_logits = []  # list to store logits at each timestep
            current_tokens = torch.zeros(B, 1, dtype=torch.long, device=device)  # start token

            for t in range(MAX_LEN):
                tgt_emb = self.embedding_output(current_tokens)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(device)
                out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
                logits = self.fc_out(out[:, -1, :])  # logits for last token
                
                # Save logits for this timestep
                output_logits.append(logits.unsqueeze(1))  # shape (B, 1, OUTPUT_DIM)

                # Sample next token
                probs = torch.softmax(logits / TEMPERATURE, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                current_tokens = torch.cat([current_tokens, next_tok], dim=1)

            # Concatenate along timestep dimension → (B, MAX_LEN, OUTPUT_DIM)
            output_logits = torch.cat(output_logits, dim=1)
            return output_logits

# -------------------------
# Dataset
# -------------------------
class MusicDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.inputs = torch.tensor(data['inputs'], dtype=torch.float32)[:, :MEMORY, :INPUT_DIM]
        self.targets = torch.tensor(data['targets'], dtype=torch.long)  # integer class indices

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# -------------------------
# Training function
# -------------------------
def train(model, train_loader, val_loader, optimizer, num_epochs=10):
    criterion = nn.CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(log_dir='runs/transformer_model')

    model.to(DEVICE)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (input_seq, target_seq) in enumerate(train_loader):
            input_seq, target_seq = input_seq.to(DEVICE), target_seq.to(DEVICE)

            optimizer.zero_grad()

            # Teacher forcing: feed target sequence shifted right
            logits = model(input_seq, target_seq[:, :])
            targets = target_seq[:, :]

            loss = criterion(
                logits.reshape(-1, OUTPUT_DIM),
                targets.reshape(-1)
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:
                # Training loss
                avg_train_loss = running_loss / (i + 1)

                # Validation loss
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_input, val_target in val_loader:
                        val_input, val_target = val_input.to(DEVICE), val_target.to(DEVICE)
                        val_logits = model(val_input, val_target[:, :])
                        val_targets = val_target[:, :]
                        v_loss = criterion(
                            val_logits.reshape(-1, OUTPUT_DIM),
                            val_targets.reshape(-1)
                        )
                        val_loss += v_loss.item()
                avg_val_loss = val_loss / len(val_loader)

                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}", f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                writer.add_scalar('Loss/train', avg_train_loss, epoch * len(train_loader) + i)
                writer.add_scalar('Loss/val', avg_val_loss, epoch * len(train_loader) + i)
                model.train()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Train Loss: {epoch_loss:.4f}")
        writer.add_scalar('Loss/epoch', epoch_loss, epoch)

        # Save checkpoint
        torch.save(model.state_dict(), "checkpoints/transformer_model.pth")

    writer.close()

# -------------------------
# Load checkpoint
# -------------------------
def load_model_checkpoint(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        return True
    else:
        print("No checkpoint found, starting from scratch.")
        return False

if __name__ == "__main__":
    d_model = 128
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    num_epochs = 10
    batch_size = 32

    model = TransformerModel(INPUT_DIM, OUTPUT_DIM, d_model, nhead,
                             num_encoder_layers, num_decoder_layers)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Load training dataset
    train_dataset = MusicDataset("data/data_train.npz")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load validation dataset
    val_dataset = MusicDataset("data/data_val.npz")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint_path = "checkpoints/transformer_model.pth"
    load_model_checkpoint(model, checkpoint_path)

    # Train with separate validation dataset
    train(model, train_loader, val_loader, optimizer, num_epochs=num_epochs)