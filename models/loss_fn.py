'''
more loss functions
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeDistinctLoss(nn.Module):
    """
    Prototype Distinct Loss to encourage prototypes to be different from each other.
    Works for input shape (num_prototypes, 1, 1, feature_dim).
    """
    def __init__(self, distance_type="cosine", margin=0.2, isAbs=True):
        """
        Args:
            distance_type (str): "cosine" for cosine distance, "euclidean" for Euclidean distance.
            margin (float): Minimum distance margin to separate prototypes.
        """
        super().__init__()
        assert distance_type in ["cosine", "euclidean"], "distance_type must be 'cosine' or 'euclidean'"
        self.distance_type = distance_type
        self.margin = margin  # Encourages prototypes to be at least `margin` distance apart
        self.isAbs = isAbs

    def forward(self, prototypes):
        """
        Compute the Prototype Distinct Loss.

        Args:
            prototypes (Tensor): Tensor of shape (num_prototypes, 1, 1, feature_dim).

        Returns:
            loss (Tensor): Scalar loss value encouraging distinct prototypes.
        """
        num_prototypes = prototypes.shape[0]
        if num_prototypes < 2:
            return torch.tensor(0.0, device=prototypes.device)  # No loss if only one prototype
        
        # Reshape from (num_prototypes, 1, 1, feature_dim) → (num_prototypes, feature_dim)
        prototypes = prototypes.view(num_prototypes, -1)  # Flatten

        # Compute pairwise distances
        if self.distance_type == "cosine":
            prototypes = F.normalize(prototypes, p=2, dim=1)  # Normalize for cosine similarity
            similarity_matrix = torch.matmul(prototypes, prototypes.T)  # Cosine similarity
            if self.isAbs:
                # abs the similarity matrix to avoid negative values, as the negative high similarity also means the prototypes are semantically close to each other (in the opposite direction)
                similarity_matrix = torch.abs(similarity_matrix)
            distance_matrix = 1 - similarity_matrix  # Convert similarity to distance
        else:  # Euclidean distance
            distance_matrix = torch.cdist(prototypes, prototypes, p=2)  # (num_prototypes, num_prototypes)

        mask = torch.eye(num_prototypes, device=prototypes.device).bool()
        distance_matrix = distance_matrix.masked_fill(mask, float("inf"))  # Ignore diagonal (self-distances)

        # Compute loss: encourage prototypes to be at least `margin` apart
        min_distances = torch.min(distance_matrix, dim=1)[0]  # Get closest prototype for each
        loss = torch.mean(F.relu(self.margin - min_distances))  # Push apart if too close

        return loss
