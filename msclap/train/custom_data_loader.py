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
        invalid_files = []
        
        for _, row in tqdm(
            df.iterrows(), total=df.shape[0], desc=f"Loading {self.csv_path} data"
        ):
            audio_file = row["file_name"].split(".")[0] + ".wav"
            wav_path = os.path.join(self.audio_dir, audio_file)
            real_path = os.path.realpath(wav_path)
            
            if not os.path.exists(real_path):
                invalid_files.append({"path": wav_path, "reason": "file_not_found"})
                continue
                
            try:
                metadata = torchaudio.info(real_path)
                
                if metadata.num_frames == 0:
                    invalid_files.append({"path": wav_path, "reason": "empty_audio"})
                    continue
                    
                if metadata.num_channels == 0:
                    invalid_files.append({"path": wav_path, "reason": "no_channels"})
                    continue
                
                try:
                    waveform, sample_rate = torchaudio.load(real_path)
                    if waveform.nelement() == 0:
                        invalid_files.append({"path": wav_path, "reason": "empty_waveform"})
                        continue
                except Exception as e:
                    invalid_files.append({"path": wav_path, "reason": f"load_error: {str(e)}"})
                    continue
                    
                data.append({
                    "audio_path": wav_path, 
                    "transcript": row["text"],
                })
                
            except Exception as e:
                invalid_files.append({"path": wav_path, "reason": f"process_error: {str(e)}"})
                continue
        
        print(f"\nData Loading Statistics:")
        print(f"Total files in CSV: {len(df)}")
        print(f"Successfully loaded: {len(data)}")
        print(f"Invalid files: {len(invalid_files)}")
        
        if invalid_files:
            error_types = {}
            for file in invalid_files:
                reason = file["reason"]
                error_types[reason] = error_types.get(reason, 0) + 1
            
            print("\nError types distribution:")
            for error_type, count in error_types.items():
                print(f"{error_type}: {count}")
                
            log_path = os.path.join(os.path.dirname(self.csv_path), "invalid_audio_files.log")
            with open(log_path, "w") as f:
                for file in invalid_files:
                    f.write(f"Path: {file['path']}, Reason: {file['reason']}\n")
            print(f"\nDetailed error log saved to: {log_path}")

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