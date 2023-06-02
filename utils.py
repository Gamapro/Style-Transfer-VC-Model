import torchaudio
import torch
import torch.optim
from torch.utils.data import Dataset
from torch import nn

class AudioDataset(Dataset):

    def __init__(self, metadata, device, sample_rate=16000, audio_length=1):
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.device = device

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_file = self.metadata.iloc[idx]['path']
        waveform, sample_rate = torchaudio.load(audio_file)
        waveform = waveform[0, :self.sample_rate * self.audio_length]
        waveform = waveform / torch.max(torch.abs(waveform))
        waveform = waveform.to(self.device)
        return waveform
    
class Wave_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res
    
class WaveNet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(WaveNet, self).__init__()
        self.wave_block = Wave_Block(in_channels, out_channels, dilation, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.wave_block(x)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        return x
    