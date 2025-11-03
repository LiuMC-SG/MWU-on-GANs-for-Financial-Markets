import math
from typing import Dict, List, Optional, Tuple, Union
import logging
import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Utility: weight initialization
# -----------------------------
def weights_init(m: nn.Module) -> None:
    """Initialize weights for better training stability"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# -----------------------------
# Generator (Sequential CNN → LSTM)
# -----------------------------
class Generator(nn.Module):
    def __init__(
            self,
            latent_dim: int = 100,
            sequence_length: int = 64,
            features: int = 32,
            lstm_hidden_size: int = 128,
            lstm_num_layers: int = 2,
            cnn_out_channels: List[int] = [64, 128, 256, 512],
            leaky_relu_slope: float = 0.2,
            dropout: float = 0.3,
        ):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.features = features
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # Initial projection to create feature maps
        self.initial_projection = nn.Linear(latent_dim, sequence_length * features)
        
        # CNN layers - Feature extraction stage
        cnn_layers = []
        in_channels = features
        for out_channels in cnn_out_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(leaky_relu_slope),
                nn.Dropout1d(dropout * 0.5)  # Light dropout for generator
            ])
            in_channels = out_channels
        
        self.cnn_seq = nn.Sequential(*cnn_layers)
        
        # LSTM layers - Sequential modeling stage
        # Input to LSTM will be (batch, seq_len, cnn_out_channels[-1])
        self.lstm = nn.LSTM(
            cnn_out_channels[-1],  # Input size from CNN output
            lstm_hidden_size, 
            lstm_num_layers, 
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        
        # Final output projection
        self.final_layers = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.BatchNorm1d(lstm_hidden_size // 2),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(dropout * 0.5),
            nn.Linear(lstm_hidden_size // 2, features),
        )
        self.final_activation = nn.Tanh()

        self.apply(weights_init)

    def forward(self, z):
        batch_size = z.size(0)
        
        # Stage 1: Initial projection
        x = self.initial_projection(z)  # (batch, seq_len * features)
        x = x.view(batch_size, self.features, self.sequence_length)  # (batch, features, seq_len)
        
        # Stage 2: CNN feature extraction
        x = self.cnn_seq(x)  # (batch, cnn_out_channels[-1], seq_len)
        
        # Stage 3: LSTM sequential modeling
        # Transpose for LSTM: (batch, seq_len, channels)
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_out_channels[-1])
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden_size)
        
        # Stage 4: Final output generation
        # Apply final layers to each timestep
        lstm_out_reshaped = lstm_out.contiguous().view(-1, self.lstm_hidden_size)  # (batch*seq_len, lstm_hidden_size)
        output = self.final_layers(lstm_out_reshaped)  # (batch*seq_len, features)
        output = output.view(batch_size, self.sequence_length, self.features)  # (batch, seq_len, features)
        
        # Transpose back to (batch, features, seq_len) and apply activation
        output = output.transpose(1, 2)  # (batch, features, seq_len)
        output = self.final_activation(output)
        
        return output

# -----------------------------
# Discriminator (Sequential CNN → LSTM)
# -----------------------------
class Discriminator(nn.Module):
    def __init__(
            self,
            sequence_length: int = 64,
            features: int = 32,
            lstm_hidden_size: int = 128,
            lstm_num_layers: int = 2,
            cnn_out_channels: List[int] = [64, 128, 256, 512],
            leaky_relu_slope: float = 0.2,
            dropout: float = 0.3,
        ):
        super(Discriminator, self).__init__()
        self.sequence_length = sequence_length
        self.features = features
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # CNN layers - Feature extraction stage
        cnn_layers = []
        in_channels = features
        for out_channels in cnn_out_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(leaky_relu_slope),
                nn.Dropout1d(dropout)
            ])
            in_channels = out_channels
        
        self.cnn_seq = nn.Sequential(*cnn_layers)
        
        # LSTM layers - Sequential analysis stage
        # Input to LSTM will be (batch, seq_len, cnn_out_channels[-1])
        self.lstm = nn.LSTM(
            cnn_out_channels[-1],  # Input size from CNN output
            lstm_hidden_size, 
            lstm_num_layers, 
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better discrimination
        )
        
        # Classification layers
        # Account for bidirectional LSTM (2 * hidden_size)
        lstm_output_size = lstm_hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 4, 1),
        )
        
        self.final_activation = nn.Sigmoid()
    
    def forward(self, x):
        # Input: (batch, features, sequence_length)
        batch_size = x.size(0)
        
        # Stage 1: CNN feature extraction
        x = self.cnn_seq(x)  # (batch, cnn_out_channels[-1], seq_len)
        
        # Stage 2: LSTM sequential analysis
        # Transpose for LSTM: (batch, seq_len, channels)
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_out_channels[-1])
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, lstm_hidden_size * 2)
        
        # Stage 3: Final classification
        # Use the final hidden state for classification
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # For bidirectional LSTM, concatenate final forward and backward hidden states
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, lstm_hidden_size * 2)
        
        # Classification
        output = self.classifier(final_hidden)
        output = self.final_activation(output)

        return output

# -----------------------------
# LSTM-CNN GAN
# -----------------------------
class LSTMCNNGANExpert:
    def __init__(
        self,
        latent_dim: int = 128,
        seq_len: int = 25,
        features: int = 1,
        resample_z: int = 3,               # used only at eval time
        lstm_hidden_size_G: int = 256,
        lstm_num_layers_G: int = 2,
        lstm_hidden_size_D: int = 256,
        lstm_num_layers_D: int = 2,
        cnn_out_channels_G: List[int] = [64, 128, 256, 512],
        cnn_out_channels_D: List[int] = [64, 128, 256, 512],
        negative_slope_G: float = 0.2,
        negative_slope_D: float = 0.2,
        dropout_G: float = 0.2,
        dropout_D: float = 0.2,
        device: Optional[torch.device] = None,
        lambda_anom: float = 0.9,         # used only for detection
        lr_G: float = 1.5e-4,
        lr_D: float = 1e-5,
        lr_anomaly: float = 0.05,
        backprop_steps: int = 50,
        batch_sizes: int = 64,
        epochs: int = 200,
        scaler_type: str = 'robust',  # 'minmax', 'robust', 'standard'
        model_save_path: str = './models',  # Added missing attribute
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = max(1, int(latent_dim))
        self.features = max(1, int(features))
        self.sequence_length = max(1, int(seq_len))
        self.resample_z = max(1, int(resample_z))
        self.lstm_hidden_size_G = max(1, int(lstm_hidden_size_G))
        self.lstm_num_layers_G = max(1, int(lstm_num_layers_G))
        self.lstm_hidden_size_D = max(1, int(lstm_hidden_size_D))
        self.lstm_num_layers_D = max(1, int(lstm_num_layers_D))
        self.cnn_out_channels_G = cnn_out_channels_G
        self.cnn_out_channels_D = cnn_out_channels_D
        self.negative_slope_G = float(negative_slope_G)
        self.negative_slope_D = float(negative_slope_D)
        self.dropout_G = float(dropout_G)
        self.dropout_D = float(dropout_D)
        self.lambda_anom = float(lambda_anom)
        self.lr_G = float(lr_G)
        self.lr_D = float(lr_D)
        self.lr_anomaly = float(lr_anomaly)
        self.backprop_steps = max(1, int(backprop_steps))
        self.batch_sizes = max(1, int(batch_sizes))
        self.epochs = max(1, int(epochs))
        self.model_save_path = model_save_path
        self.scaler_type = scaler_type

        # Models
        self._build_models()

        # Starred learning rates
        self.g_opt = torch.optim.Adam(self.G.parameters(), lr=lr_G, betas=(0.5, 0.999))
        self.d_opt = torch.optim.Adam(self.D.parameters(), lr=lr_D, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss().to(self.device)

        # Initialize scaler (robust for financial outliers)
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        # Training history
        self.training_history = {
            'g_loss': [],
            'd_loss_real': [],
            'd_loss_fake': [],
            'd_acc': [],
            'tanogan_loss': []
        }
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        log_every: int = 50,
        verbose: int = 1
    ):
        self.G.train(); self.D.train()
        step = 0

        # Data preprocessing with financial-specific handling
        X_train = self._preprocess_financial_data(X_train)
        X_val = self._preprocess_financial_data(X_val) if X_val is not None else None
        
        # Convert to PyTorch tensors
        train_tensor = torch.FloatTensor(X_train).to(self.device)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_sizes, shuffle=True)

        logger.info("Starting adversarial training...")

        # Initialize early stopping variables
        best_g_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):            
            epoch_d_loss_real = []
            epoch_d_loss_fake = []
            epoch_g_loss = []
            epoch_d_acc = []

            for i, (real_sequences,) in enumerate(train_loader):  # Fixed: unpacking tuple
                current_batch_size = real_sequences.size(0)

                # Labels
                valid_labels = torch.ones(current_batch_size, 1).to(self.device)
                fake_labels = torch.zeros(current_batch_size, 1).to(self.device)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.d_opt.zero_grad()
                
                # Generate fake sequences
                z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                fake_sequences = self.G(z)

                # Real data loss
                real_pred = self.D(real_sequences)
                d_loss_real = self.criterion(real_pred, valid_labels)
                
                # Fake data loss
                fake_pred = self.D(fake_sequences.detach())  # Detach to avoid generator gradients
                d_loss_fake = self.criterion(fake_pred, fake_labels)
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                
                d_loss.backward()
                self.d_opt.step()
                
                # -----------------
                #  Train Generator
                # -----------------
                self.g_opt.zero_grad()
                
                # Generate new fake sequences
                z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                generated_sequences = self.G(z)

                # Generator loss (wants discriminator to classify as real)
                validity = self.D(generated_sequences)
                g_loss = self.criterion(validity, valid_labels)

                g_loss.backward()
                self.g_opt.step()

                # Calculate accuracy
                real_acc = (real_pred > 0.5).float().mean()
                fake_acc = (fake_pred < 0.5).float().mean()
                d_acc = (real_acc + fake_acc) / 2

                # Store losses
                epoch_d_loss_real.append(d_loss_real.item())
                epoch_d_loss_fake.append(d_loss_fake.item())
                epoch_g_loss.append(g_loss.item())
                epoch_d_acc.append(d_acc.item())

                if (step % log_every) == 0:
                    print(f"[Epoch {epoch:03d}] step {step:06d} | d_loss {d_loss.item():.4f} | g_loss {g_loss.item():.4f}")
                
                step += 1
            
            # Calculate epoch metrics
            avg_d_loss_real = np.mean(epoch_d_loss_real)
            avg_d_loss_fake = np.mean(epoch_d_loss_fake)
            avg_g_loss = np.mean(epoch_g_loss)
            avg_d_acc = np.mean(epoch_d_acc)
            
            # Store history
            self.training_history['d_loss_real'].append(avg_d_loss_real)
            self.training_history['d_loss_fake'].append(avg_d_loss_fake)
            self.training_history['g_loss'].append(avg_g_loss)
            self.training_history['d_acc'].append(avg_d_acc)

            # Validation evaluation
            if X_val is not None and epoch % 10 == 0:
                val_tensor = torch.FloatTensor(X_val).to(self.device)
                val_tanogan_loss = self._evaluate_tanogan_loss(val_tensor)
                val_tanogan_loss_mean = val_tanogan_loss.mean().item()
                self.training_history['tanogan_loss'].append(val_tanogan_loss_mean)
            
            # Print progress
            if verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.epochs}")
                logger.info(f"  D loss real: {avg_d_loss_real:.4f}")
                logger.info(f"  D loss fake: {avg_d_loss_fake:.4f}")
                logger.info(f"  G loss: {avg_g_loss:.4f}")
                if X_val is not None:
                    logger.info(f"  TanoGAN loss: {val_tanogan_loss_mean:.4f}")
            
            # Early stopping
            if avg_g_loss < best_g_loss:
                best_g_loss = avg_g_loss
                patience_counter = 0
                self.save_models()
            else:
                patience_counter += 1
            
            if patience_counter >= 30:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        self.is_fitted = True
        self.G.eval(); self.D.eval()

        logger.info("GAN training completed successfully!")
    
    def _evaluate_tanogan_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Fixed to work with torch tensors directly and handle RNN backward passes.
        
        Args:
            x: Input tensor of shape (batches, features, sequence)
            
        Returns:
            torch.Tensor: Loss tensor of shape (batches,)
        """
        # Store original training states
        g_training = self.G.training
        d_training = self.D.training
        
        # Keep models in training mode for RNN backward compatibility
        # but we'll disable parameter updates via requires_grad=False
        self.G.train()
        self.D.train()

        B = x.size(0)

        # Disable param grads to save memory but keep graph for z.
        g_flags = [p.requires_grad for p in self.G.parameters()]
        d_flags = [p.requires_grad for p in self.D.parameters()]
        for p in self.G.parameters(): p.requires_grad_(False)
        for p in self.D.parameters(): p.requires_grad_(False)

        Dx = self.D(x)  # fixed reference for the discrimination difference

        all_losses = []  # Will store losses for each resample

        try:
            for r in range(self.resample_z):
                z = torch.randn(B, self.latent_dim, device=self.device, requires_grad=True)
                opt = torch.optim.Adam([z], lr=self.lr_anomaly)
                
                for _ in range(self.backprop_steps):
                    # Both models in training mode for RNN backward passes
                    # but parameter gradients are disabled above
                    x_hat = self.G(z)
                    Dx_hat = self.D(x_hat)
                    
                    LR = self.residual_L1(x, x_hat)                # (B,)
                    LD = self.discrimination_output_L1(Dx, Dx_hat) # (B,)
                    L_per_sample = (1.0 - self.lambda_anom) * LR + self.lambda_anom * LD  # (B,) - per sample loss
                    L_scalar = L_per_sample.mean()  # scalar for backprop

                    opt.zero_grad(set_to_none=True)
                    L_scalar.backward()
                    opt.step()
                
                # Store the final per-sample losses for this resample
                with torch.no_grad():
                    x_hat = self.G(z)
                    Dx_hat = self.D(x_hat)
                    LR = self.residual_L1(x, x_hat)
                    LD = self.discrimination_output_L1(Dx, Dx_hat)
                    final_loss = (1.0 - self.lambda_anom) * LR + self.lambda_anom * LD  # (B,)
                    all_losses.append(final_loss)
                    
        finally:
            # Restore flags
            for p, f in zip(self.G.parameters(), g_flags): p.requires_grad_(f)
            for p, f in zip(self.D.parameters(), d_flags): p.requires_grad_(f)
            
            # Restore original training states
            self.G.train(g_training)
            self.D.train(d_training)

        # Stack all losses: (resample_z, B) then transpose to (B, resample_z)
        all_losses_tensor = torch.stack(all_losses, dim=0).T  # (B, resample_z)
        
        # Get minimum loss across resamples for each batch item
        min_losses, _ = torch.min(all_losses_tensor, dim=1)  # (B,)
        
        return min_losses

    def residual_L1(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Equation (2) without the conditioning notation; returns per-sample (B,)."""
        return (x - x_hat).abs().mean(dim=(1, 2))

    def discrimination_output_L1(self, Dx: torch.Tensor, Dxh: torch.Tensor) -> torch.Tensor:
        """Equation (3) using the discriminator outputs; returns per-sample (B,)."""
        return (Dx - Dxh).abs().view(-1)

    def _preprocess_financial_data(self, X: np.ndarray) -> np.ndarray:
        """Preprocess financial time series data with robust handling."""
        logger.info(f"Input data shape: {X.shape}")
        
        # Ensure we have the expected dimensionality (B, seq_len, features)
        if len(X.shape) == 2:
            # If 2D, assume (B, seq_len) and add features dimension
            X = X[:, :, np.newaxis]
            logger.info(f"Reshaped 2D input to: {X.shape}")
        elif len(X.shape) == 3:
            # If 3D, transpose to (B, seq_len, features) if needed
            if X.shape[1] != self.sequence_length:
                if X.shape[2] == self.sequence_length:
                    X = X.transpose(0, 2, 1)
                    logger.info(f"Transposed 3D input to: {X.shape}")
        else:
            raise ValueError(f"Unsupported input shape: {X.shape}")
        
        # Update features count if needed
        if X.shape[2] != self.features:
            logger.warning(f"Input features ({X.shape[2]}) don't match model features ({self.features})")
            self.features = X.shape[2]
            # Rebuild models with correct feature count
            self._build_models()
        
        # Handle missing values and outliers
        X_processed = X.copy()
        
        # Fill NaN values with forward fill then backward fill
        for i in range(X_processed.shape[0]):
            for j in range(X_processed.shape[2]):
                series = X_processed[i, :, j]
                # Forward fill
                mask = ~np.isnan(series)
                if mask.any():
                    series[~mask] = np.interp(np.where(~mask)[0], np.where(mask)[0], series[mask])
                X_processed[i, :, j] = series
        
        # Scale the data
        original_shape = X_processed.shape
        X_reshaped = X_processed.reshape(-1, X_processed.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(original_shape)
        
        # Clip extreme outliers for stable training
        X_scaled = np.clip(X_scaled, -5, 5)
        
        # Transpose to (B, features, seq_len) for CNN
        X_final = X_scaled.transpose(0, 2, 1)
        logger.info(f"Final processed data shape: {X_final.shape}")
        
        return X_final

    def _build_models(self):
        """Build generator and discriminator models."""
        self.G = Generator(
            latent_dim=self.latent_dim, 
            sequence_length=self.sequence_length, 
            features=self.features, 
            lstm_hidden_size=self.lstm_hidden_size_G,
            lstm_num_layers=self.lstm_num_layers_G,
            dropout=self.dropout_G,
            cnn_out_channels=self.cnn_out_channels_G,
            leaky_relu_slope=self.negative_slope_G
        ).to(self.device)
        
        self.D = Discriminator(
            sequence_length=self.sequence_length, 
            features=self.features, 
            lstm_hidden_size=self.lstm_hidden_size_D,
            lstm_num_layers=self.lstm_num_layers_D,
            dropout=self.dropout_D,
            cnn_out_channels=self.cnn_out_channels_D,
            leaky_relu_slope=self.negative_slope_D
        ).to(self.device)

        logger.info(f"Generator parameters: {sum(p.numel() for p in self.G.parameters()):,}")
        logger.info(f"Discriminator parameters: {sum(p.numel() for p in self.D.parameters()):,}")

    def save_models(self, path: Optional[str] = None):
        """Save trained models."""
        save_path = path or self.model_save_path
        os.makedirs(save_path, exist_ok=True)

        if self.G is not None:
            torch.save(self.G.state_dict(), os.path.join(save_path, 'generator.pth'))
        if self.D is not None:
            torch.save(self.D.state_dict(), os.path.join(save_path, 'discriminator.pth'))
        
        # Save model configuration and scaler
        config = {
            'latent_dim': self.latent_dim,
            'sequence_length': self.sequence_length,
            'features': self.features,
            'lstm_hidden_size_G': self.lstm_hidden_size_G,
            'lstm_num_layers_G': self.lstm_num_layers_G,
            'dropout_G': self.dropout_G,
            'cnn_out_channels_G': self.cnn_out_channels_G,
            'negative_slope_G': self.negative_slope_G,
            'lstm_hidden_size_D': self.lstm_hidden_size_D,
            'lstm_num_layers_D': self.lstm_num_layers_D,
            'dropout_D': self.dropout_D,
            'cnn_out_channels_D': self.cnn_out_channels_D,
            'negative_slope_D': self.negative_slope_D,
            'resample_z': self.resample_z,
            'lambda_anom': self.lambda_anom,
            'lr_G': self.lr_G,
            'lr_D': self.lr_D,
            'lr_anomaly': self.lr_anomaly,
            'backprop_steps': self.backprop_steps,
            'batch_sizes': self.batch_sizes,
            'epochs': self.epochs,
            'scaler_type': self.scaler_type,
        }
        
        with open(os.path.join(save_path, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        
        with open(os.path.join(save_path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Models saved to {save_path}")
    
    def load_models(self, path: Optional[str] = None):
        """Load pre-trained models."""
        load_path = path or self.model_save_path
        
        try:
            # Load configuration
            with open(os.path.join(load_path, 'config.pkl'), 'rb') as f:
                config = pickle.load(f)
            
            # Update configuration
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

            # Build models
            self._build_models()
            
            # Load model weights
            self.G.load_state_dict(torch.load(
                os.path.join(load_path, 'generator.pth'), map_location=self.device))
            self.D.load_state_dict(torch.load(
                os.path.join(load_path, 'discriminator.pth'), map_location=self.device))
            
            # Load scaler
            with open(os.path.join(load_path, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_fitted = True
            logger.info(f"Models loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def plot_training_history(self):
        """Plot comprehensive training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Generator and discriminator losses
        axes[0, 0].plot(self.training_history['g_loss'], label='Generator Loss', color='blue')
        axes[0, 0].plot(self.training_history['d_loss_real'], label='Discriminator Loss (Real)', color='red')
        axes[0, 0].plot(self.training_history['d_loss_fake'], label='Discriminator Loss (Fake)', color='orange')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Discriminator accuracy
        axes[0, 1].plot(self.training_history['d_acc'], label='Discriminator Accuracy', color='green')
        axes[0, 1].set_title('Discriminator Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reconstruction loss (if available)
        if self.training_history['reconstruction_loss']:
            axes[1, 0].plot(self.training_history['reconstruction_loss'], label='Reconstruction Loss', color='purple')
            axes[1, 0].set_title('Validation Reconstruction Loss')
            axes[1, 0].set_xlabel('Epoch (x10)')
            axes[1, 0].set_ylabel('Reconstruction Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Loss difference (generator vs discriminator)
        g_loss = np.array(self.training_history['g_loss'])
        d_loss_avg = (np.array(self.training_history['d_loss_real']) + 
                     np.array(self.training_history['d_loss_fake'])) / 2
        
        axes[1, 1].plot(g_loss - d_loss_avg, label='G_loss - D_loss', color='purple')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Loss Difference (Generator - Discriminator)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def detect_financial_anomalies(self, X: np.ndarray, threshold_percentile: float = 95.0, return_details: bool = False) -> Union[np.ndarray, Dict]:
        """
        Detect financial anomalies with specialized scoring.
        
        Args:
            X: Input financial time series
            threshold_percentile: Percentile for anomaly threshold
            return_details: Whether to return detailed analysis
        
        Returns:
            Binary anomaly labels or detailed analysis dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detecting anomalies")
        
        # Get anomaly scores
        processed_X = self._preprocess_financial_data(X)
        processed_tensor = torch.FloatTensor(processed_X).to(self.device)
        anomaly_scores = []
        
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, processed_tensor.size(0), batch_size):
            batch = processed_tensor[i:i+batch_size]
            batch_scores = self._evaluate_tanogan_loss(batch).cpu()
            if isinstance(batch_scores, (int, float)):
                anomaly_scores.extend([batch_scores] * batch.size(0))
            else:
                anomaly_scores.extend(batch_scores)
        
        anomaly_scores = np.array(anomaly_scores)

        # Determine threshold
        threshold = np.percentile(anomaly_scores, threshold_percentile)
        anomaly_labels = (anomaly_scores > threshold).astype(int)
        
        if not return_details:
            return anomaly_labels
        
        # Detailed analysis for financial context
        results = {
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': anomaly_scores,
            'threshold': threshold,
            'anomaly_rate': np.mean(anomaly_labels),
            'high_risk_indices': np.where(anomaly_scores > np.percentile(anomaly_scores, 99))[0],
            'medium_risk_indices': np.where((anomaly_scores > np.percentile(anomaly_scores, 95)) & 
                                          (anomaly_scores <= np.percentile(anomaly_scores, 99)))[0],
            'score_statistics': {
                'mean': np.mean(anomaly_scores),
                'std': np.std(anomaly_scores),
                'median': np.median(anomaly_scores),
                'q95': np.percentile(anomaly_scores, 95),
                'q99': np.percentile(anomaly_scores, 99)
            }
        }
        
        return results

    def generate_samples(self, n_samples: int = 100, return_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate synthetic samples using the trained generator.
        
        Args:
            n_samples: Number of samples to generate
            return_numpy: Whether to return numpy array (True) or torch tensor (False)
            
        Returns:
            Generated samples with shape (n_samples, features, seq_len) or (n_samples, seq_len, features)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating samples")
        
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            generated = self.G(z)  # (n_samples, features, seq_len)
            
            if return_numpy:
                generated = generated.cpu().numpy()
                # Transpose back to (n_samples, seq_len, features) for consistency
                generated = generated.transpose(0, 2, 1)
                
                # Inverse transform if scaler was fitted
                if hasattr(self.scaler, 'inverse_transform'):
                    original_shape = generated.shape
                    generated_reshaped = generated.reshape(-1, generated.shape[-1])
                    generated_scaled = self.scaler.inverse_transform(generated_reshaped)
                    generated = generated_scaled.reshape(original_shape)
                
                return generated
            else:
                return generated
