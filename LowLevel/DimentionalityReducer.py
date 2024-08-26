import torch
from torch import nn
import umap
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA


class DimensionalityReducer: 
    """
    AutoEncoder acts on data representation space: (B, R) F is large (1024)
    Other methods work on target property space: (B, T)

    """
    def __init__(self, target_dim=1, method='umap', encoder_path=None):
        self.target_dim = target_dim
        self.method = method
        self.reducer = None
        self.encoder_path = encoder_path

    def load_encoder(self, input_dim=1024):
        """Load and return a wrapper around the pre-trained encoder."""
        if self.encoder_path is None:
            raise ValueError("Encoder path must be provided for 'autoencoder' method.")
        
        # Create an instance of the encoder model
        # Adjust based on your input dimensions
        encoder = AutoEncoder(input_dim=input_dim, encoding_dim=self.target_dim).get_encoder()
        
        # Load the encoder state
        encoder.load_state_dict(torch.load(self.encoder_path))
        encoder.eval()
        
        # Return a wrapper around the encoder
        return EncoderWrapper(encoder)

    def fit_transform(self, data):
        """Reduce the dimensionality of data to the target dimension using the specified method"""
        if self.method == 'umap':
            self.reducer = umap.UMAP(n_components=self.target_dim)
        elif self.method == 'pca':
            self.reducer = PCA(n_components=self.target_dim)
        elif self.method == 'tsne':
            self.reducer = TSNE(n_components=self.target_dim)
        elif self.method == 'isomap':
            self.reducer = Isomap(n_components=self.target_dim)
        elif self.method == 'autoencoder':
            self.reducer = self.load_autoencoder()
        else:
            raise ValueError("Method must be 'umap', 'pca', 'tsne', 'isomap', 'autoencoder', ")
        
        return self.reducer.fit_transform(data)


class AutoEncoder(nn.Module):
    """Train data representation compression"""
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2), 

            nn.Linear(128, 256),  # Corrected the input size here
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.2), 

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_encoder(self):
        return self.encoder
    
class EncoderWrapper:
    """Wrapper for encoder to provide fit_transform method."""
    def __init__(self, encoder):
        self.encoder = encoder

    def fit_transform(self, data):
        """Transforms data using the encoder part of the autoencoder."""
        # Ensure data is in the correct format (Torch Tensor)
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        # Transform the data using the encoder
        with torch.no_grad():
            reduced_data = self.encoder(data)
        
        return reduced_data.numpy()