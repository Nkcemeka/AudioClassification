import torch 
from torch import nn
from torch.utils.data import DataLoader 
from custom_dataset import UrbanDataset
from audio_cnn import AudioCNN
import torchaudio

# Define training params and other utilities 
batch_size = 128
num_epochs = 10 
metadata = "../UrbanSound8K/metadata/UrbanSound8K.csv"
audio_path = "../UrbanSound8K/audio/"
sample_rate = 22050
num_samp = 22050 # number of samples in the audio file


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
        Train the model for one epoch 
    """
    for imgs, targets in dataloader: 
        imgs, targets = imgs.to(device), targets.to(device)
        # Get the loss 
        preds = model(imgs) 
        loss = loss_fn(preds, targets) 

        # backpropagation of the loss and weight updates
        optimizer.zero_grad() # reset the gradients for each batch 
        loss.backward() # backpropagate the loss
        optimizer.step() # update the weights

    # Print the loss for the last batch or at the end o the epoch 
    print(f"Loss: {loss.item()}")


def train(model, dataloader, loss_fn, optimizer, device, epochs = 10):
    """
        Train the model for multiple epochs 
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print("------------------")
    print("Finished training")


if __name__ == "__main__":
    # Create dataset object 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate = sample_rate,
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64
    )
    train_data = UrbanDataset(metadata, audio_path, mel_spec, sample_rate, num_samp, device)    

    # Get data loader for the training set 
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

    model = AudioCNN().to(device)

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    # Train the model 
    train(model, train_dataloader, loss_fn, optimizer, device, num_epochs)

    # Save the model
    torch.save(model.state_dict(), "audio_cnn_model.pth")
    print("Model saved successfully")


