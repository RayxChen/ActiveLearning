import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=3):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = x
        for i in range(len(self.layers)):
            h = torch.relu(self.bn_layers[i](self.layers[i](h)))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=3):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(latent_dim, hidden_dim))
        self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = z
        for i in range(len(self.layers)):
            h = torch.relu(self.bn_layers[i](self.layers[i](h)))

        return torch.sigmoid(self.fc_output(h)).clamp(0, 1)

class VAE(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, latent_dim, learning_rate=1e-5):
        super(VAE, self).__init__()
        self.learning_rate = learning_rate
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        
        self.save_hyperparameters()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss (BCE)
        # BCE = nn.functional.binary_cross_entropy(recon_x, x)#, reduction='sum')
        MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        # Kullback-Leibler divergence loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD, MSE, KLD

    def training_step(self, batch, batch_idx):
        batch = batch[0]  # Unpack batch if it's a tuple (as from TensorDataset)
        recon_batch, mu, logvar = self.forward(batch)
        loss, bce, kld = self.loss_function(recon_batch, batch, mu, logvar)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_bce', bce, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_kld', kld, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = batch[0]
        recon_batch, mu, logvar = self.forward(batch)
        loss, bce, kld = self.loss_function(recon_batch, batch, mu, logvar)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_bce', bce, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_kld', kld, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.encoder.fc2_mu.out_features).to(self.device)
        return self.decoder(z)

    def generate(self, x):
        recon_x, _, _ = self.forward(x)
        return recon_x
