import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
from tqdm import tqdm


class CustomAudioTextDataset(Dataset):
    def __init__(self, audio_directory, csv_path):
        self.audio_dir = audio_directory
        self.csv_path = csv_path
        self.data = self._load_data()

    def _load_data(self):
        df = pd.read_csv(self.csv_path)
        data = []
        for _, row in tqdm(
            df.iterrows(), total=df.shape[0], desc=f"Loading {self.csv_path} data"
        ):
            audio_file = row["file_name"].split(".")[0] + ".wav"
            wav_path = os.path.join(self.audio_dir, audio_file)
            
            # symbolic link인 경우도 처리
            real_path = os.path.realpath(wav_path)
            
            if os.path.exists(real_path):
                data.append({"audio_path": wav_path, "transcript": row["text"]})
            else:
                print(f"Warning: Audio file not found: {wav_path} (real path: {real_path})")
                
        print(f"Loaded {len(data)} audio files, {len(df) - len(data)} audio files not found")

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        waveform, sample_rate = torchaudio.load(item["audio_path"])
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return {
            "audio_path": item["audio_path"],
            "waveform": waveform[0],
            "sample_rate": sample_rate,
            "transcript": item["transcript"],
        }

    def get_audio_sample_rate(self, idx):
        audio_path = self.data[idx]["audio_path"]
        metadata = torchaudio.info(audio_path)
        return metadata.sample_rate


def collate_fn(batch):
    max_length = max(item["waveform"].shape[0] for item in batch)

    waveforms = [
        F.pad(item["waveform"], (0, max_length - item["waveform"].shape[0]))
        for item in batch
    ]

    return {
        "audio_path": [item["audio_path"] for item in batch],
        "waveform": torch.stack(waveforms).float(),
        "sample_rate": [item["sample_rate"] for item in batch],
        "transcript": [item["transcript"] for item in batch],
    }