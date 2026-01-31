import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Subset
from utils.constants import DEVICE, INPUT_DIM, MEMORY, LEARNING_RATE, NUM_CLASSES
from utils.FifthsCircleLoss import FifthsCircleLoss

# -------------------------
# Constants
# -------------------------
OUTPUT_DIM = NUM_CLASSES
MAX_LEN = MEMORY + 1
BOS_TOKEN_INDEX = NUM_CLASSES

# -------------------------
# Transformer Model
# -------------------------
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim

        # Embeddings
        self.feature_to_embedding = nn.Linear(input_dim, d_model)
        self.embedding_output = nn.Embedding(output_dim, d_model)  # use integer class indices

        # Positional encoding (learned, batch-first)
        self.pos_encoder = nn.Parameter(torch.randn(1, MAX_LEN, d_model))

        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_encoder_layers
        )

        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_decoder_layers
        )

        # Final linear layer
        self.fc_out = nn.Linear(d_model, output_dim)

    def _causal_mask(self, T, device):
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def forward(self, input_seq):
        """
        Closed-loop autoregressive generation (no teacher forcing)

        input_seq: [B, MEMORY, INPUT_DIM]
        returns:   [B, MAX_LEN, OUTPUT_DIM] logits
        """
        B = input_seq.size(0)
        device = input_seq.device

        # ----- Encoder -----
        x = self.feature_to_embedding(input_seq.float())  # [B, MEMORY, d_model]
        x = x + self.pos_encoder[:, :x.size(1)]
        memory = self.encoder(x)  # [B, MEMORY, d_model]

        # ----- Autoregressive decoder -----
        # Start with BOS token
        generated_tokens = torch.full(
            (B, 1),
            BOS_TOKEN_INDEX,  # BOS token index
            dtype=torch.long,
            device=device
        )

        output_logits = torch.zeros(
            B, MAX_LEN, OUTPUT_DIM, device=device
        )

        for t in range(MAX_LEN):
            # Embed generated tokens so far
            decoder_emb = self.embedding_output(generated_tokens)  # [B, t+1, d_model]
            decoder_emb = decoder_emb + self.pos_encoder[:, :decoder_emb.size(1)]

            # Causal mask
            tgt_mask = self._causal_mask(decoder_emb.size(1), device)

            # Decode
            dec_out = self.decoder(
                tgt=decoder_emb,
                memory=memory,
                tgt_mask=tgt_mask
            )

            # Predict next token
            logits = self.fc_out(dec_out[:, -1])  # [B, OUTPUT_DIM]
            output_logits[:, t] = logits

            # CLOSED LOOP: feed model output back in
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

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
def train(model, train_loader, optimizer, num_epochs=10):
    criterion = FifthsCircleLoss()  # expects [B*MAX_LEN, OUTPUT_DIM] logits and [B*MAX_LEN] targets

    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(log_dir='runs/transformer_model')

    model.to(DEVICE)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (input_seq, target_seq) in enumerate(train_loader):
            input_seq, target_seq = input_seq.to(DEVICE), target_seq.to(DEVICE)
            optimizer.zero_grad()

            output = model(input_seq)  # [B, MAX_LEN, OUTPUT_DIM]

            # Flatten for loss
            logits = output.view(-1, OUTPUT_DIM)         # [B*MAX_LEN, OUTPUT_DIM]
            targets = target_seq.view(-1)                # [B*MAX_LEN]

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
                print(f"Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/epoch', avg_loss, epoch)

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

# -------------------------
# Main
# -------------------------
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

    train_data = MusicDataset("data/data_train.npz")

    num_samples = 200000
    indices = torch.randperm(len(train_data))[:num_samples]
    train_data = Subset(train_data, indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint_path = "checkpoints/transformer_model.pth"
    load_model_checkpoint(model, checkpoint_path)

    train(model, train_loader, optimizer, num_epochs=num_epochs)
