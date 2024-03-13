# Import necessary libraries 
import torch 
from audio_cnn import AudioCNN
from custom_dataset import UrbanDataset
import torchaudio
from train import metadata, audio_path, sample_rate, num_samp

# Define class map  
class_map = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

# Define predict function
def predict(model, img, target, class_map):
    # Set model to evaluation
    model.eval()
    with torch.no_grad():
        preds = model(img) 
        pred_idx = preds[0].argmax(0)
        pred = class_map[pred_idx]
        truth = class_map[target]
    return pred, truth


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pretrained model 
    model = AudioCNN().to(device)
    model.load_state_dict(torch.load("audio_cnn_model.pth"))

    # Get dataset
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate = sample_rate,
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64
    )

    urban_dataset = UrbanDataset(metadata, audio_path, mel_spec, sample_rate, num_samp, device)

    # Get test point for inference 
    input, target = urban_dataset[0][0], urban_dataset[0][1]
    input.unsqueeze_(0) # add batch dimension

    # make inference
    pred, truth = predict(model, input, target, class_map)
    print(f"Prediction: {pred}, Truth: {truth}")
