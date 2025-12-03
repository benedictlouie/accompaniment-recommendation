import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from constants import CHORD_CLASSES

class FifthsCircleLoss(nn.Module):
    def __init__(self, num_chords=25, factor_minor=0.8, k=99.0, no_chord_penalty=3, eta=0.2):
        """
        num_chords: number of chord classes (default 25)
        factor_minor: scale factor for minor chords
        k_N: sharpness for the 'N' chord sigmoid
        """
        super().__init__()
        self.num_chords = num_chords
        self.factor_minor = factor_minor
        self.k = k
        self.eta = eta
        self.no_chord_penalty = torch.tensor(no_chord_penalty)

        # precompute chord coordinates
        coords = torch.stack([self.map_to_circle(torch.tensor(i, dtype=torch.float32)) 
                              for i in range(num_chords)])  # [num_chords, 2]
        self.register_buffer("chord_coords", coords)

    def map_to_circle(self, x):
        """
        Differentiable mapping from chord index x -> 2D coordinates
        """
        factor = self.factor_minor

        is_N = torch.sigmoid(self.k * (x - (self.num_chords - 1.5)))
        is_minor = torch.sin(math.pi / 2 * x) ** 2

        dist_major = x / 2 * 7
        dist_minor = (x + 5) / 2 * 7

        angle_major = 2 * math.pi * dist_major / 12
        angle_minor = 2 * math.pi * dist_minor / 12
        coord_major = torch.stack([torch.cos(angle_major), torch.sin(angle_major), self.no_chord_penalty])
        coord_minor = factor * torch.stack([torch.cos(angle_minor), torch.sin(angle_minor), self.no_chord_penalty])
        coords = is_N * torch.zeros(3) + (1 - is_N) * ((1 - is_minor) * coord_major + is_minor * coord_minor)
        return coords

    def forward(self, logits, target_idx):
        """
        logits: [batch_size, num_chords]
        target_idx: [batch_size] long tensor of chord indices
        """
        probs = torch.softmax(logits, dim=-1)  # [B, num_chords]
    
        pred_coords = probs @ self.chord_coords  # [B, 3]
        target_coords = self.chord_coords[target_idx]  # [B, 3]
        coord_loss = torch.norm(pred_coords - target_coords, dim=1).mean()
        cat_loss = F.cross_entropy(logits, target_idx)
        loss = coord_loss + self.eta * cat_loss
        return loss

if __name__ == "__main__":
    fcl = FifthsCircleLoss()
    x = np.zeros((0,3))
    for i in range(25):
        res = fcl.map_to_circle(torch.tensor(i)).numpy()
        x = np.vstack((x, res))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c='r', marker='o')

    # Label each point with its index
    for i, (xi, yi, zi) in enumerate(x):
        ax.text(xi, yi, zi, CHORD_CLASSES[i], color='blue')

    # Optional: set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()