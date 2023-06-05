import torchaudio
import torch
import torch.optim
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchaudio import transforms

class AudioDataset(Dataset):

    def __init__(self, metadata, device, sample_rate=16000, audio_length=1):
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.device = device

    def __len__(self):
        return len(self.metadata)

    def to_waveform(self, audio_file):
        waveform, _ = torchaudio.load(audio_file, normalize=True)
        waveform = waveform[0, :self.sample_rate * self.audio_length]
        waveform = waveform / torch.max(torch.abs(waveform))
        waveform = waveform.to(self.device)
        return waveform

    def __getitem__(self, idx):
        audio_file = self.metadata.iloc[idx]['path']
        speech_waveform = self.to_waveform(audio_file)
        random_idx = torch.randint(0, len(self.metadata), (1,)).item()
        style_audio_file = self.metadata.iloc[random_idx]['path']
        style_waveform = self.to_waveform(style_audio_file)
        return speech_waveform, style_waveform
    
class AudioMELSpectogramDataset(Dataset):

    def __init__(self, metadata, device, sample_rate=32000, audio_length=1, n_fft=800):
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.device = device
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)
        # self.transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)

    def __len__(self):
        return len(self.metadata)

    def to_waveform(self, audio_file):
        waveform, _ = torchaudio.load(audio_file, normalize=True)
        waveform = waveform[0, :self.sample_rate * self.audio_length]
        waveform = waveform / torch.max(torch.abs(waveform))
        # waveform = waveform.to(self.device)
        return waveform

    def __getitem__(self, idx):
        audio_file = self.metadata.iloc[idx]['path']
        speech_waveform = self.to_waveform(audio_file)
        speech_spectrogram = self.transform(speech_waveform)
        random_idx = torch.randint(0, len(self.metadata), (1,)).item()
        style_audio_file = self.metadata.iloc[random_idx]['path']
        style_waveform = self.to_waveform(style_audio_file)
        style_spectrogram = self.transform(style_waveform)
        speech_spectrogram = speech_spectrogram.type(torch.float64)
        style_spectrogram = style_spectrogram.type(torch.float64)
        return speech_spectrogram, style_spectrogram # , speech_waveform, style_waveform

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
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std

def CalcContentLoss(gen_feat,orig_feat):
    #calculating the content loss of each layer by calculating the MSE between the content and generated features and adding it to content loss
    content_l=torch.mean((gen_feat-orig_feat)**2)
    return content_l

def CalcStyleLoss(gen,style):
    #Calculating the gram matrix for the style and the generated image
    channel,height,width=gen.shape
    G = torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
    # print('G', G.max(), G.min())
    A = torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())
    # print('A', A.max(), A.min())     
    #Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
    style_l=torch.mean((G-A)**2)
    return style_l

def plot_mel_spectrogram(mel_spectrogram, name='mel_spectrogram'):
    plt.figure(figsize=(10, 4))
    mel_spectrogram = np.log(mel_spectrogram)
    plt.imshow(mel_spectrogram, origin='lower', aspect='auto') # , cmap='inferno')
    plt.xticks([]), plt.yticks([])
    #plt.xlabel('Frame'), plt.ylabel('Mel Bin')
    # plt.title('Mel Spectrogram'), plt.colorbar(label='Magnitude (dB)')
    plt.tight_layout()
    plt.savefig(f'{name}.png')
    plt.show()

def mel_to_wav(mel_spectrogram):
    sample_rate = 16000  # Frecuencia de muestreo del audio original
    n_fft = 256  # Número de puntos de la STFT
    hop_length = 128 # Tamaño del salto entre ventanas
    n_mels = 128  # Número de bandas de frecuencia en el espectrograma de Mel
    # Crear una instancia de la transformación inversa de escala Mel
    inverse_mel_scale = transforms.InverseMelScale(n_stft=(n_fft // 2) + 1, n_mels=n_mels).to(device)
    # Convertir el espectrograma de Mel a un espectrograma lineal
    mel_spectrogram = mel_spectrogram.float()
    linear_spectrogram = inverse_mel_scale(mel_spectrogram)
    # Restaurar el espectrograma lineal a su escala original (opcional)
    linear_spectrogram = linear_spectrogram * linear_spectrogram.max()
    griffin_lim_transform = transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length).to(device)
    waveform = griffin_lim_transform(linear_spectrogram)
    return waveform.cpu().numpy()