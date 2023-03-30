class AudioDataset(Dataset):

    def __init__(self, data_dir, sample_rate, frame_length, frame_step, num_mels, fmin, fmax):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mels = num_mels
        self.fmin = fmin
        self.fmax = fmax
        self.train_dataset = torchaudio.datasets.VOXCELEB(data_dir, download=True)
        self.test_dataset = torchaudio.datasets.VOXCELEB(data_dir, download=True, url="test")

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, _, _, _ = self.train_dataset[idx]
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)(waveform)
        waveform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.frame_length, hop_length=self.frame_step, n_mels=self.num_mels, f_min=self.fmin, f_max=self.fmax)(waveform)
        waveform = torchaudio.transforms.AmplitudeToDB()(waveform)
        return waveform
    
    def get_test_data(self):
        waveform, sample_rate, _, _, _ = self.test_dataset[0]
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)(waveform)
        waveform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.frame_length, hop_length=self.frame_step, n_mels=self.num_mels, f_min=self.fmin, f_max=self.fmax)(waveform)
        waveform = torchaudio.transforms.AmplitudeToDB()(waveform)
        return waveform
    
    def get_test_data_loader(self, batch_size):
        return DataLoader(self.get_test_data(), batch_size=batch_size)
    

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