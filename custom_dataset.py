from torch.utils.data import Dataset 
import os
import pandas as pd
import torchaudio
import torch

class UrbanDataset(Dataset):
    """
        UrbanDataset class
    """
    
    def __init__(self, csv_file, audio_path, trans, samp_rate, num_samp, device):
        """
            Constructor
        """
        self.metadata = pd.read_csv(csv_file)
        self.audio_path = audio_path
        self.device = device
        self.trans = trans.to(self.device)
        self.sample_rate = samp_rate
        self.num_samples = num_samp 

    def __len__(self):
        """
            Number of samples in the dataset
        """
        return len(self.metadata)

    def __getitem__(self, idx):
        """
            Get a sample from the dataset
        """
        # load audio and corresponding label
        sample_audio = self._get_sample(idx)
        label = self._get_label(idx)
        audio, audio_sr = torchaudio.load(sample_audio)
        audio = audio.to(self.device) # register the audio to device
        # Transform the audio to a mel spectrogram
        audio = self._resample(audio, audio_sr) # resample the audio signals to have the same sample rate
        audio = self._audio_mono(audio) # convert stereo to audio for stereo files
        audio = self._trunc(audio) # truncate the audio signals to have the same number of samples if it is more than the desired number
        audio = self._audio_pad(audio) # pad the audio signals to have the same number of samples if it is less than the desired number
        audio_mel = self.trans(audio) # (1, n_mels, T) where T is the number of frames
        return audio_mel, label

    def _trunc(self, audio):
        """
            Truncate the audio signals
        """
        if audio.shape[1] > self.num_samples:
            audio = audio[:, :self.num_samples]
        return audio

    def _audio_pad(self, audio):
        """
            Right pad the audio signals if 
            num of samples is less than 
            desired number 
        """
        if audio.shape[1] < self.num_samples:
            samples_to_add = self.num_samples - audio.shape[1]
            right_padding = (0, samples_to_add) # (left_pad (last dim), right_pad(last_dim)), you can pad for other dimensions
            audio = torch.nn.functional.pad(audio, right_padding)
        return audio

    def _resample(self, audio, audio_sr):
        """
            Resample the audio signals
        """
        if audio_sr !=  self.sample_rate:
            resamp_obj = torchaudio.transforms.Resample(audio_sr, self.sample_rate).to(self.device)
            audio = resamp_obj(audio)
        return audio

    def _audio_mono(self, audio):
        """
            Convert stereo to mono
        """
        # audio is a tensor of (num_chans, num_samples)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        return audio

    def _get_sample(self, idx):
        """
            Get the audio sample
        """
        folder = f"fold{self.metadata.iloc[idx, 5]}"
        path = os.path.join(self.audio_path, folder, self.metadata.iloc[idx, 0])
        return path 

    def _get_label(self, idx):
        """
            Get the label
        """
        return self.metadata.iloc[idx, 6]

if __name__ == "__main__":
    metadata = "../UrbanSound8K/metadata/UrbanSound8K.csv"
    audio_path = "../UrbanSound8K/audio/"
    sample_rate = 22050
    num_samp = 22050 # number of samples in the audio file

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"DEVICE: {device}")

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate = sample_rate,
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64
    )

    urban_dataset = UrbanDataset(metadata, audio_path, mel_spec, sample_rate, num_samp, device)
    signal, label = urban_dataset[0]
    print(f"Signal shape: {signal.shape}")
    print(f"Label: {label}")

    print(f"Number of samples in the dataset: {len(urban_dataset)}")
