# Exploração de Dados

> ```python
> import numpy as np
> import librosa
> import librosa.display
> import matplotlib.pyplot as plt
> import os
>
> DATASET_PATH = "../data/genres_original/"
> SAMPLE_GENRE = 'blues'
> SAMPLE_FILE = 'blues.00000.wav'
> FILE_PATH = os.path.join(DATASET_PATH, SAMPLE_GENRE, SAMPLE_FILE)
>
> # carrega arquivo de aúdio
> signal, sample_rate = librosa.load(FILE_PATH, sr=22050)
>
> # mostra waveform do arquivo de aúdio
> plt.figure(figsize=(15, 5))
> librosa.display.waveshow(signal, sr=sample_rate, alpha=0.5)
> plt.title("Forma de Onda de {}".format(SAMPLE_FILE))
> plt.xlabel("Tempo (s)")
> plt.ylabel("Amplitude")
> plt.show()
> ```

Esse código analisa e visualiza um arquivo de áudio de um dataset de gêneros musicais. Ele utiliza as bibliotecas `numpy`, `librosa` e `matplotlib` para as seguintes tarefas:

**Definição do dataset**: Especifica o caminho do dataset de arquivos de áudio, seleciona o gênero "blues" e escolhe um arquivo de exemplo (`blues.00000.wav`).

**Carregamento do áudio**: Usa `librosa.load` para carregar o arquivo de áudio. O sinal é armazenado como uma matriz NumPy e a taxa de amostragem é fixada em 22050 Hz.

**Visualização do waveform**: Plota a forma de onda do arquivo de áudio carregado usando `librosa.display.waveshow` para mostrar como o sinal varia ao longo do tempo. O gráfico é configurado com título, rótulos de eixos e transparência ajustada para melhor visualização.

# Análise de Espectro

> ```python
> fft = np.fft.fft(signal)
> magnitude = np.abs(fft)
> frequency = np.linspace(0, sample_rate, len(magnitude))
>
> # divide as partes positivas e negativas do espectro
> left_frequency = frequency[:len(frequency)//2]
> left_magnitude = magnitude[:len(magnitude)//2]
>
> # plota o espectro
> plt.figure(figsize=(15, 5))
> plt.plot(left_frequency, left_magnitude)
> plt.title("Espectro de potência de  {}".format(SAMPLE_FILE))
> plt.xlabel("Frequência (Hz)")
> plt.ylabel("Magnitude")
> plt.show()
> ```

Esse código realiza a análise de espectro de um sinal de áudio, transformando-o para o domínio da frequência e visualizando a distribuição de potência ao longo das frequências. Ele utiliza as bibliotecas `numpy` e `matplotlib` para as seguintes tarefas:

**Transformada de Fourier**: Aplica a transformada de Fourier (é realizada com `np.fft.fft`) ao sinal de áudio para converter o sinal do domínio do tempo para o domínio da frequência.

**Cálculo de magnitude e frequência**: Calcula a magnitude do espectro usando `np.abs` e define os valores correspondentes de frequência com `np.linspace` para cobrir o intervalo de 0 até a taxa de amostragem (`sample_rate`).

**Seleção do espectro positivo**: Como o espectro é simétrico, apenas a metade positiva das frequências é selecionada para análise e visualização.

**Plotagem do espectro**: Utiliza `plt.plot` para exibir o espectro de potência, com a frequência no eixo x e a magnitude no eixo y. O gráfico é configurado com um título, rótulos de eixos e dimensões ajustadas para clareza.

# Análise de Espectrograma

> ```python
> # Realizar a STFT para obter o espectrograma
> n_fft = 2048  # Número de amostras por FFT
> hop_length = 512  # Número de amostras entre janelas
>
> stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
> spectrogram = np.abs(stft)
>
> # Exibir o espectrograma
> plt.figure(figsize=(15, 5))
> librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='linear')
> plt.colorbar(format="%+2.f")
> plt.title("Espectrograma de {}".format(SAMPLE_FILE))
> plt.xlabel("Tempo (s)")
> plt.ylabel("Frequência (Hz)")
> plt.show()
> ```

Este código realiza a Análise de Fourier de Curto Prazo (STFT) para gerar um espectrograma do sinal de áudio, permitindo a visualização da distribuição de frequências ao longo do tempo. Ele utiliza `librosa` e `matplotlib` para as seguintes tarefas:

**Configuração dos parâmetros da STFT**: Define o número de amostras por FFT (`n_fft`) e o espaçamento entre janelas consecutivas (`hop_length`). Esses parâmetros controlam a resolução em frequência e tempo.

**Cálculo da STFT**: Aplica a STFT ao sinal usando `librosa.stft` e calcula a magnitude do espectro com `np.abs`.

**Visualização do espectrograma**: Utiliza `librosa.display.specshow` para exibir o espectrograma em um formato gráfico, com o tempo no eixo x, a frequência no eixo y e a magnitude representada por um mapa de cores.

# Espectrograma em Escala Logarítmica

> ```python
> # Converter amplitude para decibéis
> log_spectrogram = librosa.amplitude_to_db(spectrogram)
>
> # Exibir o espectrograma em escala logarítmica
> plt.figure(figsize=(15, 5))
> librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log')
> plt.colorbar(format="%+2.f dB")
> plt.title("Espectrograma em Escala Logarítmica de {}".format(SAMPLE_FILE))
> plt.xlabel("Tempo (s)")
> plt.ylabel("Frequência (Hz) (Escala Log)")
> plt.show()
> ```

Este código converte o espectrograma em escala linear para escala logarítmica e exibe os dados em um gráfico, permitindo uma interpretação mais intuitiva da distribuição de frequências. Ele utiliza `librosa` e `matplotlib` para as seguintes tarefas:

**Conversão para decibéis**: Aplica a função `librosa.amplitude_to_db` ao espectrograma para converter os valores de amplitude em uma escala logarítmica representada em decibéis. Isso destaca melhor variações de magnitude em diferentes frequências.

**Visualização em escala logarítmica**: Utiliza `librosa.display.specshow` para exibir o espectrograma com uma escala logarítmica no eixo das frequências. O tempo é mostrado no eixo x, enquanto a frequência em escala logarítmica aparece no eixo y.

**Adiciona elementos ao gráfico**: Inclui uma barra de cores formatada em decibéis, bem como título e rótulos nos eixos para facilitar a interpretação.

**Adiciona elementos ao gráfico**: Inclui uma barra de cores para indicar os valores de magnitude, bem como título e rótulos para os eixos do gráfico.

# Extração de MFCCs

> ```python
> # Extração de MFCCs
> MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
>
> # Exibir os MFCCs
> plt.figure(figsize=(15, 5))
> librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length, x_axis='time')
> plt.colorbar()
> plt.title("MFCCs de {}".format(SAMPLE_FILE))
> plt.xlabel("Tempo (s)")
> plt.ylabel("Coeficientes MFCC")
> plt.show()
> ```

Este código realiza a extração e visualização dos Mel-Frequency Cepstral Coefficients (MFCCs), que são amplamente utilizados na análise e reconhecimento de áudio, especialmente em aplicações de processamento de fala e música. Ele utiliza `librosa` e `matplotlib` para as seguintes tarefas:

**Extração dos MFCCs**: A função `librosa.feature.mfcc` é usada para calcular 13 coeficientes MFCC a partir do sinal de áudio. Os parâmetros incluem:
- `n_mfcc`: Define o número de coeficientes a serem calculados (13 neste caso).
- `n_fft`: Especifica o tamanho da janela FFT.
- `hop_length`: Define o espaçamento entre janelas.

**Visualização dos MFCCs**: Utiliza `librosa.display.specshow` para exibir os coeficientes MFCC como um mapa de calor, com o tempo no eixo x e os coeficientes no eixo y.

**Adiciona elementos ao gráfico**: Inclui uma barra de cores para indicar a magnitude dos coeficientes, bem como título e rótulos nos eixos para facilitar a interpretação dos dados.

