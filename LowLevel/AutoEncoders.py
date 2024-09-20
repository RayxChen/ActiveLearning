import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class AutoEncoder(pl.LightningModule):
    """Unified AutoEncoder/Variational AutoEncoder with flexible architecture."""
    def __init__(self, input_dim=1024, base_hidden_dim=512, num_layers=3, latent_dim=32, vae=False, scaling_factor=0.5, learning_rate=1e-3):
        super(AutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.base_hidden_dim = base_hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.vae = vae
        self.scaling_factor = scaling_factor
        self.learning_rate = learning_rate

        # Create hidden dimensions dynamically based on the number of layers and scaling factor
        hidden_dims = [int(base_hidden_dim * (scaling_factor ** i)) for i in range(num_layers)]

        # Build the Encoder
        self.encoder = self.build_network(input_dim, hidden_dims, latent_dim, is_encoder=True)
        
        # Build the Decoder
        self.decoder = self.build_network(latent_dim, hidden_dims, input_dim, is_encoder=False)
    
    def build_network(self, start_dim, hidden_dims, end_dim, is_encoder=True):
        """Builds either the encoder or decoder part of the AutoEncoder."""
        layers = []
        dims = hidden_dims if is_encoder else hidden_dims[::-1]  # Reverse for decoder
        
        prev_dim = start_dim
        for h_dim in dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(True))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, end_dim))
        if not is_encoder:
            layers.append(nn.Sigmoid())  # Apply Sigmoid in decoder's last layer
        
        return nn.Sequential(*layers)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = self.encoder(x)
        
        if self.vae:
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            z = self.reparameterize(mu, logvar)
            return self.decoder(z), mu, logvar
        else:
            z = self.fc_latent(x)
            return self.decoder(z)
    
    def get_latent_vector(self, x):
        """Returns the latent vector (encoded representation)."""
        x = self.encoder(x)
        if self.vae:
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            return self.reparameterize(mu, logvar)
        else:
            return self.fc_latent(x)
    
    def loss_function(self, recon_x, x, mu=None, logvar=None):
        if self.vae:
            MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return MSE + KLD, MSE, KLD
        else:
            return nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    def training_step(self, batch, batch_idx):
        batch = batch[0]
        if self.vae:
            recon_batch, mu, logvar = self.forward(batch)
            loss, MSE, KLD = self.loss_function(recon_batch, batch, mu, logvar)
        else:
            recon_batch = self.forward(batch)
            loss = self.loss_function(recon_batch, batch)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('KLD_loss', KLD, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = batch[0]
        if self.vae:
            recon_batch, mu, logvar = self.forward(batch)
            loss, MSE, KLD = self.loss_function(recon_batch, batch, mu, logvar)
        else:
            recon_batch = self.forward(batch)
            loss = self.loss_function(recon_batch, batch)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def save_encoder(self, filepath):
        """Save the encoder part of the AutoEncoder."""
        torch.save(self.encoder.state_dict(), filepath)
        print(f"Encoder has been saved successfully to {filepath}.")

    def load_encoder(self, input_dim=1024, base_hidden_dim=512, num_layers=3, latent_dim=8, encoder_path=None):
        """Load the encoder part of the AutoEncoder."""
        hidden_dims = [int(base_hidden_dim * (0.5 ** i)) for i in range(num_layers)]
        encoder = self.build_network(input_dim, hidden_dims, latent_dim, is_encoder=True)
        
        if encoder_path is not None:
            encoder.load_state_dict(torch.load(encoder_path))
            encoder.eval()
        
        return EncoderWrapper(encoder)
    
    

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