# Importação de Bibliotecas e Definição de Parâmetros

> ```python
> # Importar bibliotecas necessárias
> import json
> import os
> import math
> import librosa
> import librosa.display
> import numpy as np
> import matplotlib.pyplot as plt
>
> # Definir caminhos e parâmetros
> DATASET_PATH = "../data/genres_original/"  # Caminho para o dataset de gêneros originais
> JSON_PATH = "../data/data_10.json"        # Caminho para salvar o arquivo JSON
> SAMPLE_RATE = 22050                        # Taxa de amostragem
> TRACK_DURATION = 30                         # Duração de cada faixa em segundos
> SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION  # Número de amostras por faixa
>
> # Número de segmentos que cada faixa de áudio será dividida
> NUM_SEGMENTS = 10
>
> # Parâmetros para extração das MFCCs
> NUM_MFCC = 13
> N_FFT = 2048
> HOP_LENGTH = 512
> ```

Este código configura o ambiente de análise de áudio e define parâmetros essenciais para processar um dataset de áudio. As bibliotecas `librosa` e `matplotlib` são usadas para processamento e visualização de áudio, enquanto `numpy` auxilia na manipulação de arrays.

**Importação de bibliotecas**:
- `json`, `os`, `math`: Bibliotecas para manipulação de arquivos e código auxiliar.
- `librosa`, `librosa.display`: Para processamento e exibição de recursos de áudio.
- `numpy`: Para operações matemáticas em arrays.
- `matplotlib.pyplot`: Para visualização de gráficos.

**Definição de caminhos e parâmetros**:
- `DATASET_PATH`: Caminho para o diretório que contém os arquivos de áudio.
- `JSON_PATH`: Caminho para salvar um arquivo JSON que contém informações processadas.
- `SAMPLE_RATE`: Taxa de amostragem, que define a frequência de amostras por segundo (22.050 Hz, padrão para áudio digital).
- `TRACK_DURATION`: Duração de cada faixa de áudio em segundos.
- `SAMPLES_PER_TRACK`: Total de amostras em uma faixa áudio (calculado como `SAMPLE_RATE * TRACK_DURATION`).

**Parâmetros de segmentação e extração de MFCCs**:
- `NUM_SEGMENTS`: O número de segmentos em que cada faixa será dividida para processamento.
- `NUM_MFCC`: O número de coeficientes MFCC a serem extraçados (13, um valor comum).
- `N_FFT`: O tamanho da janela FFT (2048 amostras).
- `HOP_LENGTH`: O número de amostras entre janelas de FFT (512 amostras).

Esses parâmetros e bibliotecas configuram a base para a extração e análise dos MFCCs, que são utilizados para caracterizar as propriedades espectrais do áudio.

# Extrai as MFCCs do dataset de música e as salva em um arquivo JSON junto com os rótulos de gêneros.

```python
def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10): # Extrai as MFCCs do dataset de música e as salva em um arquivo JSON junto com os rótulos de gêneros.

    """
    dataset_path (str): Caminho para o dataset de gênros musicais.
    json_path (str): Caminho para salvar o arquivo JSON.
    num_mfcc (int): Número de coeficientes MFCC a serem extraídos.
    n_fft (int): Número de amostras por FFT.
    hop_length (int): Número de amostras entre sucessivas transformadas FFT.
    num_segments (int): Número de segmentos em que cada faixa de áudio será dividida.
    """

    # Dicionário para armazenar os dados
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # Lista para registrar arquivos problemáticos
    problematic_files = []

    # Loop através de todas as pastas de gênero
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # Ignorar a pasta principal
        if dirpath != dataset_path:
            # Extrair o rótulo do gênero a partir do nome da pasta
            semantic_label = os.path.basename(dirpath)
            data["mapping"].append(semantic_label)
            print(f"\nProcessing Genre: {semantic_label}")

            # Processar todos os arquivos de áudio na pasta de gênero
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                try:
                    # Carregar o arquivo de áudio
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    # Verificar se a duração do sinal corresponde ao esperado
                    if len(signal) < SAMPLES_PER_TRACK:
                        print(f"File {file_path} is shorter than expected. Skipping.")
                        problematic_files.append(file_path)
                        continue

                    # Processar todos os segmentos de áudio
                    for d in range(num_segments):
                        # Calcular o início e o fim das amostras para o segmento atual
                        start_sample = samples_per_segment * d
                        end_sample = start_sample + samples_per_segment

                        # Extrair as MFCCs
                        mfcc = librosa.feature.mfcc(y=signal[start_sample:end_sample],
                                                    sr=sr,
                                                    n_mfcc=num_mfcc,
                                                    n_fft=n_fft,
                                                    hop_length=hop_length)
                        mfcc = mfcc.T  # Transpor para que cada linha represente um vetor de MFCC

                        # Verificar se o número de vetores de MFCC está correto
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)  # Ajustar o índice do rótulo
                        else:
                            print(f"Segment {d+1} in file {file_path} has an unexpected number of MFCC vectors. Skipping.")
                            problematic_files.append(file_path)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    problematic_files.append(file_path)

    # Salvar os dados extraídos no arquivo JSON
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    print("\nMFCCs have been successfully saved to JSON file!")

    # Salvar os arquivos problemáticos em um log
    if problematic_files:
        with open("../data/problematic_files.log", "w") as log_file:
            for file in problematic_files:
                log_file.write(f"{file}\n")
        print(f"\nFound {len(problematic_files)} problematic files. Details are saved in 'problematic_files.log'.")
    else:
        print("\nNo problematic files found.")

# Executar a função para extrair e salvar as MFCCs
save_mfcc(DATASET_PATH, JSON_PATH, num_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, num_segments=NUM_SEGMENTS)
```

Este código implementa uma função para extrair coeficientes de Mel-frequency cepstral (MFCCs) de arquivos de áudio em um dataset de gênros musicais e salvar os resultados em um arquivo JSON. Ele realiza as seguintes tarefas:

**Configuração e Parâmetros:**
- Define o caminho para o dataset de áudio e o caminho de salvação do arquivo JSON.
- Configura parâmetros para a extração dos MFCCs, incluindo o número de coeficientes, número de amostras por FFT, e o espaço entre janelas de FFT.

**Processamento do Dataset:**
- Itera sobre todas as pastas de gêneros e carrega os arquivos de áudio usando a biblioteca Librosa.
- Divide cada faixa em segmentos iguais e extrai os MFCCs de cada segmento.
- Verifica se o número de vetores de MFCC por segmento está correto e armazena os dados em um dicionário.

**Tratamento de Arquivos Problemáticos:**
- Identifica e registra arquivos que apresentam erros ou não atendem aos requisitos de comprimento.

**Saída de Dados:**
- Salva os MFCCs e os rótulos em um arquivo JSON.
- Registra arquivos problemáticos em um log separado, se existirem.

## Verificação do Conteúdo do Arquivo JSON

```python
import json

with open(JSON_PATH, "r") as fp:
    data = json.load(fp)

print(f"\nNúmero de gêneros mapeados: {len(data['mapping'])}")
print(f"Número total de segmentos de áudio: {len(data['mfcc'])}")
print(f"Número total de rótulos: {len(data['labels'])}")
```

Carrega um arquivo JSON especificado pelo caminho `JSON_PATH`, e em seguida imprime três informações:
1. O número de gêneros mapeados.
2. O número total de segmentos de áudio, representados por 'mfcc'.
3. O número total de rótulos contidos no arquivo.
