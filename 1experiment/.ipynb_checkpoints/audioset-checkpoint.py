import pandas as pd
from audiotools import AudioSignal
from torch.utils.data import Dataset, DataLoader
import random

class AudioDataset(Dataset):
    def __init__(self, cfg, data_path, train=True):
        self.train = train
        
        self.target_sample_rate = cfg.sample_rate
        self.duration = cfg.duration
        self.device = cfg.device

        if isinstance(data_path, list):
            # Combine all files in the list
            combined_csv = pd.concat([pd.read_csv(f) for f in data_path])
        else:
            self.df = pd.read_csv(data_path)

    def __len__(self):
        return len(self.df)

    def pre_process(self, audio_path, total_duration):
        # Set duration
        duration = self.duration if total_duration >= 3 else total_duration  # Duration is 3 seconds or total_duration if less than 3
        
        # Set offset based on conditions
        if total_duration < self.duration or self.train == False: # 3초보다 짧으면 그냥 사용
            offset = 0.0
        else:
            # 3초보다 길면 랜덤한 구간에서 3초를 가져와서 사용
            max_offset = total_duration - duration  # Calculate the maximum possible offset
            offset = random.uniform(0, max_offset)  # Choose a random offset within the possible range
        
        # Load audio signal file
        wav = AudioSignal(audio_path, offset=offset, duration=duration)
        length = wav.signal_length

        # Encode audio signal as one long file
        wav.to_mono()
        wav.resample(self.target_sample_rate)

        if wav.duration < self.duration: # 3초보다 짧으면 패딩으로 채우기
          pad_len = int(self.duration * self.target_sample_rate) - wav.signal_length
          wav.zero_pad(0, pad_len)
        assert wav.duration <= self.duration # 3초보다 길면? 안되는데 그러면

        return wav.audio_data
        

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        
        audio_path = data['audio_path']
        total_duration = data['duration']
        description = "<RANDOM>" # This is dummy text for pre-training

        
        wav = self.pre_process(audio_path, total_duration)
        
        if data['added_audio_path']:

        return wav.squeeze(1), description, length


class TestDataset(Dataset):
    def __init__(self, cfg):
        if cfg.prompts is None:
            test_df = pd.read_csv(cfg.eval_data_path)
            self.prompts = ["<RANDOM>"] * 12
        else:
            self.prompts = cfg.prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]