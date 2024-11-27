'''
Entire file has been converted from a jupyter notebook

Problem Statement
Modify the code so that the second layer output from the encoder is combined with the 
first layer output of the decoder and the model produces better results of reconstruction.

'''

# mount drive
from google.colab import drive
drive.mount('/content/drive')

# move around the drive directory to requried location, using !ls commands

# import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim
import math

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to the Hymenoptera dataset in Google Drive
data_path = 'hymenoptera'  # Update this to your dataset path

# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# Load data
train_data = datasets.ImageFolder(root=data_path + '/train', transform=transform)
val_data = datasets.ImageFolder(root=data_path + '/val', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)


# to observe what the train_data loader looks like
print(train_data)

'''
Dataset ImageFolder
    Number of datapoints: 244
    Root location: hymenoptera/train
    StandardTransform
Transform: Compose(
               Resize(size=(128, 128), interpolation=bilinear, max_size=None, antialias=True)
               ToTensor()
           )
'''

# alternate
''' 
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        # to add ReLU activation after each layer, for both encoder and decoder
        self.encoder_L1 = nn.Conv2d(3, 16, kernel_size= 3, padding = 1)
        
        self.encoder_L2 = nn.Conv2d(16, 32, kernel_size= 3, padding = 1)

        self.encoder_L3 = nn.Conv2d(32, 64, kernel_size= 3, padding = 1)

        self.ReLU = nn.ReLU(inplace=True)



        # Decoder
        self.decoder_L1 = nn.ConvTranspose2d(64, 32, 3, padding =1, output_padding = 1, stride = 2)
        self.decoder_L2 = nn.ConvTranspose2d(32, 16, 3, padding =1, output_padding = 1, stride = 2)
        self.decoder_L3 = nn.ConvTranspose2d(16, 3, 3, padding =1, output_padding = 1, stride = 2)

    
    # a simpler forward function
    def forward(self, inputs):

      encoder11 = nn.ReLU(self.encoder_L1(inputs))

      encoder12 = nn.ReLU(self.encoder_L2(encoder11))

      encoder13 = nn.ReLU(self.encoder_L3(encoder12))

      # decoder block
      decoder11 = nn.ReLU(self.decoder_L1(encoder13))

      residual_connections = torch.cat([decoder11, encoder12], dim =  1)
      decoder12 = nn.ReLU(self.decoder_L2(residual_connections))


      decoder13 = nn.ReLU(self.decoder_L3(decoder12))
      '''



# class for an encoder block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 2, padding=1)

        # relu is already incorporated here, so no need to add it in the model class
        self.relu = nn.ReLU(inplace=True)
    def forward(self, inputs):
      conv_output = self.conv(inputs)
      output = self.relu(conv_output)
      return conv_output, output

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(out_channels + out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, inputs, skip = None):

      output = self.upconv(inputs)
      if skip is not None:
        output = torch.cat([inputs, skip], axis = 1)

      output = self.conv(output)
      output = self.relu(output)
      return output

#
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder

        self.encoder_L1 = EncoderBlock(3, 16)
        self.encoder_L2 = EncoderBlock(16, 32)
        self.encoder_L3 = EncoderBlock(32, 64)

        # Decoder
        self.decoder_L1 = DecoderBlock(64, 32)
        self.decoder_L2 = DecoderBlock(32, 16)
        self.decoder_L3 = DecoderBlock(16, 3)

    # a simpler forward function
    def forward(self, inputs):

      # encoder part
      conv_output1, output1 = self.encoder_L1(inputs)
      conv_output2, output2 = self.encoder_L2(output1)
      conv_output3, output3 = self.encoder_L3(output2)

      # we have arrived at bottleneck

      d1 = self.decoder_L1(output3, None)
      # here we add the results of Decoder Layer 1 and Encoder Layer 2
      d2 = self.decoder_L2(d1, conv_output2)
      d3 = self.decoder_L3(d2, None)

      return d3

test_run = ConvAutoencoder()

# to see the model architecture
print(test_run)

'''
ConvAutoencoder(
  (encoder_L1): EncoderBlock(
    (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (relu): ReLU(inplace=True)
  )
  (encoder_L2): EncoderBlock(
    (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (relu): ReLU(inplace=True)
  )
  (encoder_L3): EncoderBlock(
    (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (relu): ReLU(inplace=True)
  )
  (decoder_L1): DecoderBlock(
    (upconv): ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (conv): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu2): ReLU(inplace=True)
  )
  (decoder_L2): DecoderBlock(
    (upconv): ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (conv): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu2): ReLU(inplace=True)
  )
  (decoder_L3): DecoderBlock(
    (upconv): ConvTranspose2d(16, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (conv): Conv2d(6, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu2): ReLU(inplace=True)
  )
)
'''

# Initialize model, loss function and optimizer
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Training the Autoencoder
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs, _, _ = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    train_loss /= len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}')


from skimage.metrics import structural_similarity as ssim

def compute_ssim(img1, img2):
    # Convert tensors to numpy arrays and move channel to last dimension
    img1_np = img1.squeeze().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.squeeze().cpu().numpy().transpose(1, 2, 0)

    # Determine the smallest dimension and set `win_size` accordingly
    height, width, _ = img1_np.shape
    win_size = min(height, width)
    if win_size >= 7:
        win_size = 7  # Use the default max size of 7
    elif win_size % 2 == 0:
        win_size -= 1  # Ensure win_size is odd

    # Compute SSIM with adjusted window size and channel axis
    ssim_value = ssim(img1_np, img2_np, win_size=win_size, channel_axis=-1, data_range=img2_np.max() - img2_np.min())
    return ssim_value


# Function to compute PSNR
def compute_psnr(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float('inf')
    pixel_max = 1.0  # Since images are normalized
    return 20 * math.log10(pixel_max / math.sqrt(mse))


# Modified validation function to include PSNR and SSIM
def validate_autoencoder(model, val_loader):
    model.eval()
    val_loss = 0
    ssim_scores = []
    psnr_scores = []

    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            outputs, _, _ = model(data)

            # Calculate loss
            loss = criterion(outputs, data)
            val_loss += loss.item() * data.size(0)

            # Calculate SSIM and PSNR for each image in the batch
            for i in range(data.size(0)):
                ssim_score = compute_ssim(data[i], outputs[i])
                psnr_score = compute_psnr(data[i], outputs[i])

                ssim_scores.append(ssim_score)
                psnr_scores.append(psnr_score)

    # Average the loss, SSIM, and PSNR over the validation set
    val_loss /= len(val_loader.dataset)
    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)

    print(f'Validation Loss: {val_loss:.4f}, Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.2f} dB')
    return val_loss, avg_ssim, avg_psnr

# Run the validation with metrics
val_loss, avg_ssim, avg_psnr = validate_autoencoder(model, val_loader)

'''
Validation Loss: 0.0029, Average SSIM: 0.8187, Average PSNR: 26.28 dB
'''


# Function to plot the grid with actual, regenerated, and activation maps
def plot_results(model, val_loader):
    model.eval()
    data_iter = iter(val_loader)
    images, _ = next(data_iter)
    images = images[:4].to(device)

    with torch.no_grad():
        outputs, enc_activation, dec_activation = model(images)

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    for i in range(4):
        # Original Image
        axes[i, 0].imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Reconstructed Image
        axes[i, 1].imshow(np.transpose(outputs[i].cpu().numpy(), (1, 2, 0)))
        axes[i, 1].set_title('Reconstructed Image')
        axes[i, 1].axis('off')

        # Encoder Activation Map (from 3rd layer)
        enc_act_map = enc_activation[i].cpu().numpy()
        enc_act_map = np.mean(enc_act_map, axis=0)  # Average over channels for visualization
        axes[i, 2].imshow(enc_act_map, cmap='viridis')
        axes[i, 2].set_title('Encoder Activation (Layer 3)')
        axes[i, 2].axis('off')

        # Decoder Activation Map (from 2nd layer)
        dec_act_map = dec_activation[i].cpu().numpy()
        dec_act_map = np.mean(dec_act_map, axis=0)  # Average over channels for visualization
        axes[i, 3].imshow(dec_act_map, cmap='viridis')
        axes[i, 3].set_title('Decoder Activation (Layer 2)')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()
# Display results on validation images
plot_results(model, val_loader)