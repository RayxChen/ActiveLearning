from AutoEncoders import AutoEncoder
import umap
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA


class DimensionalityReducer: 
    """
    AutoEncoder acts on data representation space: (B, R) F is large (1024)
    Other methods work on target property space: (B, T)

    """
    def __init__(self, target_dim=1, method='umap', encoder_params=None):
        self.target_dim = target_dim
        self.method = method
        self.reducer = None
        self.encoder_params = encoder_params # dict

    def load_encoder(self):
        """Load and return a wrapper around the pre-trained encoder."""
        if self.encoder_path is None:
            raise ValueError("Encoder params must be provided for 'autoencoder' method.")
        
        encoder = AutoEncoder().load_encoder(**self.encoder_params)
        
        return encoder

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
            self.reducer = self.load_encoder()
        else:
            raise ValueError("Method must be 'umap', 'pca', 'tsne', 'isomap', 'autoencoder', ")
        
        return self.reducer.fit_transform(data)


