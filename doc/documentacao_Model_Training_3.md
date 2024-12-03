```python
# Define o caminho para o arquivo JSON contendo as MFCCs e rótulos
DATA_PATH = "../data/data_10.json"

def load_data(data_path):   # Carrega os dados de treinamento a partir de um arquivo JSON.
    """
    data_path (str): Caminho para o arquivo JSON contendo os dados.
    return X: Inputs (MFCCs).
    return y: Targets (Rótulos de gêneros).
    mapping: Lista de gêneros musicais.
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # Converter listas para arrays numpy
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    mapping = data["mapping"]

    print("Dados carregados com sucesso!")
    print(f"Shape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")
    print(f"Gêneros: {mapping}")

    return X, y, mapping

# Carregar os dados
X, y, genres = load_data(DATA_PATH)
```

1. **Carregar o arquivo JSON:** O arquivo é aberto e lido para extrair os dados armazenados.
2. **Conversão para arrays Numpy:** As listas de MFCCs e rótulos são convertidas em arrays numpy para facilitar o uso em algoritmos de aprendizado de máquina.
3. **Exibir informações:** O código imprime o tamanho dos arrays `X` (MFCCs) e `y` (rótulos), bem como a lista de gêneros.

**Uso:**
- `X`: Contém os coeficientes MFCC extraídos das faixas de áudio.
- `y`: Contém os rótulos numéricos correspondentes aos gêneros musicais.
- `genres`: Lista com os nomes dos gêneros musicais.

```python
# Verificar a distribuição das classes
import pandas as pd

# Criar um DataFrame para facilitar a visualização
df = pd.DataFrame(y, columns=["label"])

# Contagem de cada classe
label_counts = df["label"].value_counts().sort_index()

# Plotar a distribuição das classes
plt.figure(figsize=(10,6))
sns.barplot(x=genres, y=label_counts.values, palette="viridis")
plt.title("Distribuição das Classes")
plt.xlabel("Gêneros Musicais")
plt.ylabel("Número de Segmentos")
plt.xticks(rotation=45)
plt.show()
```

1. **Cria um DataFrame**: Utiliza os rótulos (`y`) para criar uma estrutura de dados tabular com a coluna `label` representando os gêneros musicais.

2. **Conta os rótulos**: A função `value_counts` é usada para calcular o número de ocorrências de cada gênero musical (classe), e o resultado é ordenado por índice.

3. **Visualiza os dados**: Utiliza `seaborn` para criar um gráfico de barras, exibindo o número de segmentos associados a cada gênero musical.

### Resultado:
- Um gráfico que mostra a distribuição de segmentos para cada classe (índice da classe) alinhada aos gêneros musicais listados na variável `genres`.
- O eixo X representa os gêneros musicais e o eixo Y, a quantidade de segmentos de áudio para cada gênero.

### Utilidade:
Este gráfico é útil para verificar a balanceabilidade do conjunto de dados. Desequilíbrios significativos podem exigir técnicas como reamostragem ou ajustes nos hiperparâmetros do modelo para melhorar o desempenho durante o treinamento.

> ### Dividir os dados em treino e teste
```python
# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Shape de X_train: {X_train.shape}")
print(f"Shape de X_test: {X_test.shape}")
print(f"Shape de y_train: {y_train.shape}")
print(f"Shape de y_test: {y_test.shape}")
```

- **Parâmetros da Função:**
  - `X`: Dados de entrada (MFCCs extraídos dos segmentos de áudio).
  - `y`: Rótulos correspondentes aos gêneros musicais.
  - `test_size=0.2`: Proporção de 20% dos dados reservada para o conjunto de teste.
  - `random_state=42`: Garante a reprodutibilidade da divisão dos dados.
  - `stratify=y`: Mantém a distribuição proporcional das classes nos conjuntos de treino e teste.

### Resultados Esperados
O código imprime as dimensões dos conjuntos gerados:

- **`X_train`**: Dados de entrada para treino.
- **`X_test`**: Dados de entrada para teste.
- **`y_train`**: Rótulos para treino.
- **`y_test`**: Rótulos para teste.

Essa separação garante que o modelo seja avaliado de forma justa, utilizando exemplos que ele não viu durante o treinamento.

```python
# Definir a arquitetura do modelo
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
    
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(len(genres), activation='softmax')  # Camada de saída com softmax para classificação multi-classe
])

# Compilar o modelo
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Resumo do modelo
model.summary()
```

---

1. **Arquitetura do Modelo**:
    - **Entrada (Flatten)**: Transforma os dados de entrada 2D (MFCCs) em um vetor 1D para ser usado em camadas densas.
    - **Camadas Ocultas**:
        - Primeira camada densa com 512 neurônios, seguida por uma camada de Dropout para evitar overfitting.
        - Segunda camada densa com 256 neurônios e Dropout.
        - Terceira camada densa com 128 neurônios e Dropout.
    - **Camada de Saída**:
        - Uma camada totalmente conectada com o número de saídas igual ao número de gêneros musicais (classes).
        - Ativação `softmax` é usada para calcular a probabilidade de cada classe.

2. **Compilação do Modelo**:
    - **Otimizador**: Adam com uma taxa de aprendizado de 0.0001.
    - **Função de Perda**: `sparse_categorical_crossentropy`, adequada para classificação multi-classe com rótulos inteiros.
    - **Métrica**: `accuracy` para avaliar o desempenho do modelo.

3. **Resumo do Modelo**:
    - Apresenta uma descrição detalhada das camadas e parâmetros do modelo.

```python
# Treinar o modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Avaliar o modelo nos dados de teste
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nAcurácia no conjunto de teste: {test_acc * 100:.2f}%")

# Plotar a acurácia do treinamento e validação
plt.figure(figsize=(14, 5))

# Acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Acurácia Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
plt.title('Acurácia do Modelo')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

# Perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perda Treino')
plt.plot(history.history['val_loss'], label='Perda Validação')
plt.title('Perda do Modelo')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.show()

# Salvar o modelo treinado
model.save("../models/music_genre_classifier.h5")
print("Modelo salvo em '../models/music_genre_classifier.h5'")

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Gerar o relatório de classificação
print("Relatório de Classificação:\n")
print(classification_report(y_test, y_pred, target_names=genres))

# Gerar a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotar a matriz de confusão
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=genres, yticklabels=genres)
plt.title("Matriz de Confusão")
plt.ylabel('Classe Verdadeira')
plt.xlabel('Classe Predita')
plt.show()
```

1. **Treinamento do modelo**:
   - Utiliza os dados de treinamento (`X_train`, `y_train`) e de validação (`X_test`, `y_test`).
   - O treinamento é realizado por 50 épocas com um tamanho de lote de 32.
   - Exibe a acurácia e a perda durante o processo.

2. **Avaliação do modelo**:
   - Calcula a acurácia no conjunto de teste.
   - Exibe os gráficos de acurácia e perda para treino e validação ao longo das épocas.

3. **Salvamento do modelo**:
   - O modelo treinado é salvo no formato H5 em `../models/music_genre_classifier.h5`.

4. **Geração de previsões**:
   - Realiza previsões no conjunto de teste (`X_test`).
   - Converte as probabilidades previstas para as classes com maior probabilidade.

5. **Relatório de classificação**:
   - Gera um relatório detalhado com métricas como precisão, recall e F1-score para cada gênero.

6. **Matriz de confusão**:
   - Calcula a matriz de confusão entre os valores verdadeiros e preditos.
   - Plota a matriz para visualização das discrepâncias de classificação.

A implementação é eficiente para avaliar o desempenho do modelo e identificar pontos fracos na classificação de diferentes gêneros musicais.

```python
from sklearn.preprocessing import StandardScaler

# Normalização dos dados
scaler = StandardScaler()

# Refatorar os dados para aplicar a normalização
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Ajustar e transformar os dados
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

# Reshape de volta para a forma original
X_train_scaled = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test_scaled = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Atualizar as variáveis de treino e teste
X_train, X_test = X_train_scaled, X_test_scaled

print("Dados normalizados com sucesso!")

# Redefinir a arquitetura do modelo com Batch Normalization
model_bn = keras.Sequential([
    keras.layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
    
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(len(genres), activation='softmax')
])

# Compilar o modelo
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model_bn.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Resumo do modelo
model_bn.summary()

# Callbacks necessários
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Caminho para salvar o melhor modelo com extensão .keras
checkpoint_path = "../models/best_model.keras"

# Definir os callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                   monitor='val_accuracy',
                                   save_best_only=True,
                                   mode='max',
                                   verbose=1)

# Treinando o modelo com callbacks
history_bn = model_bn.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,  # Aumentando o número de épocas para permitir que o modelo tenha mais chances de melhorar
    batch_size=32,
    callbacks=[early_stop, model_checkpoint],
    verbose=1
)
```

1. **Normalização dos Dados**:
   - Utiliza `StandardScaler` para normalizar os dados de entrada. Este processo é necessário para garantir que os valores estejam centrados em torno de zero e tenham uma distribuição com desvio padrão unitário.
   - Aplica a normalização aos conjuntos de treino e teste e os reformata para suas formas originais.

2. **Arquitetura do Modelo com Batch Normalization**:
   - Redefine o modelo neural adicionando camadas de `BatchNormalization` para acelerar o treinamento e melhorar a estabilidade.
   - Inclui três camadas densas com `Dropout` para prevenir overfitting.
   - A última camada utiliza `softmax` para classificação multi-classe.

3. **Callbacks**:
   - Configura `EarlyStopping` para interromper o treinamento quando o desempenho de validação não melhorar por um determinado número de épocas (paciencia de 10 épocas).
   - Configura `ModelCheckpoint` para salvar o melhor modelo com base na acurácia de validação.

4. **Treinamento do Modelo**:
   - Treina o modelo usando os dados normalizados, aumentando o número de épocas (100) para permitir melhoria do desempenho.
   - Inclui os callbacks configurados para melhorar a eficiência do treinamento e salvar o melhor modelo.

```python
# Definir o caminho para salvar o modelo aprimorado
checkpoint_path_improved = "../models/best_model_improved.keras"

# Callbacks aprimorados para Early Stopping e Checkpoint
early_stop_improved = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model_checkpoint_improved = ModelCheckpoint(filepath=checkpoint_path_improved,
                                           monitor='val_accuracy',
                                           save_best_only=True,
                                           mode='max',
                                           verbose=1)

# Treinar modelo aprimorado com callbacks atualizados
history_improved = model_bn.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,  # Pode ser ajustado conforme necessário
    batch_size=32,
    callbacks=[early_stop_improved, model_checkpoint_improved],
    verbose=1
)

# Callback para reduzir a taxa de aprendizado
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                 patience=5,
                                 factor=0.5,
                                 min_lr=1e-6,
                                 verbose=1)

# Treinar o modelo com callbacks aprimorados e redução de taxa de aprendizado
history_lr = model_bn.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop_improved, model_checkpoint_improved, lr_reduction],
    verbose=1
)

# Avaliar e gerar métricas do modelo aprimorado
from sklearn.metrics import classification_report, confusion_matrix

# Carregar o melhor modelo salvo
best_model = keras.models.load_model("../models/best_model_improved.keras")
print("Melhor modelo carregado com sucesso!")

# Avaliar o modelo nos dados de teste
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=2)
print(f"\nAcurácia no conjunto de teste após melhorias: {test_acc * 100:.2f}%")

# Fazer previsões no conjunto de teste
y_pred = best_model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Gerar o relatório de classificação
print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred, target_names=genres))

# Gerar a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotar a matriz de confusão
plt.figure(figsize=(12,10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=genres, yticklabels=genres)
plt.title("Matriz de Confusão")
plt.ylabel('Classe Verdadeira')
plt.xlabel('Classe Predita')
plt.show()
```

### Normalização e Redução de Taxa de Aprendizado
1. **Objetivo**: Ajustar os hiperparâmetros do modelo e melhorar a performance geral utilizando callbacks.
2. **EarlyStopping**: Interrompe o treinamento caso a validação não melhore após 15 épocas.
3. **ModelCheckpoint**: Salva o melhor modelo com base na métrica de acurácia da validação.
4. **ReduceLROnPlateau**: Reduz a taxa de aprendizado pela metade se a acurácia da validação não melhorar após 5 épocas consecutivas.

### Avaliação do Modelo
1. **Carga do Modelo**: Carrega o melhor modelo salvo durante o treinamento.
2. **Teste de Acurácia**: Avalia a performance do modelo no conjunto de teste.
3. **Relatório de Classificação**: Exibe métricas detalhadas, incluindo precisão, revocação e F1-score.
4. **Matriz de Confusão**: Visualiza a performance do modelo em diferentes classes.

### Visualizações
1. **Treinamento e Validação**: Monitoramento das curvas de perda e acurácia ao longo das épocas.
2. **Matriz de Confusão**: Oferece uma visualização clara das previsões corretas e incorretas do modelo.

```python
# Plotar a acurácia do treinamento e validação
plt.figure(figsize=(14, 5))

# Acurácia
plt.subplot(1, 2, 1)
plt.plot(history_lr.history['accuracy'], label='Acurácia Treino')
plt.plot(history_lr.history['val_accuracy'], label='Acurácia Validação')
plt.title('Acurácia do Modelo com Redução de Taxa de Aprendizado')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

# Perda
plt.subplot(1, 2, 2)
plt.plot(history_lr.history['loss'], label='Perda Treino')
plt.plot(history_lr.history['val_loss'], label='Perda Validação')
plt.title('Perda do Modelo com Redução de Taxa de Aprendizado')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.show()

# Salvar o modelo final melhorado
best_model.save("../models/music_genre_classifier_final.keras")
print("Modelo final melhorado salvo em '../models/music_genre_classifier_final.keras'")

# Modelo usando uma Rede Neural Convolucional (CNN) com Ajustes no Pooling

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Expandir as dimensões dos dados para incluir o canal (necessário para CNN)
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

# Definir a arquitetura do modelo CNN com pool_size=(2,1) para evitar redução excessiva da largura
model_cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2], 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,1)),
    
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,1)),
    
    Conv2D(128, (2,2), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,1)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    
    Dense(len(genres), activation='softmax')
])

# Compilar o modelo CNN
model_cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Resumo do modelo CNN
model_cnn.summary()

# Treinando o modelo CNN com Callbacks Ajustados

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Definindo o caminho para salvar o melhor modelo CNN com extensão .keras
checkpoint_path_cnn = "../models/best_model_cnn.keras"

# Definindo os callbacks
early_stop_cnn = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint_cnn = ModelCheckpoint(filepath=checkpoint_path_cnn,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max',
                                      verbose=1)
lr_reduction_cnn = ReduceLROnPlateau(monitor='val_loss',
                                     patience=5,
                                     factor=0.5,
                                     min_lr=1e-6,
                                     verbose=1)

# Treinando o modelo CNN com callbacks ajustados
history_cnn = model_cnn.fit(
    X_train_cnn, y_train,
    validation_data=(X_test_cnn, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop_cnn, model_checkpoint_cnn, lr_reduction_cnn],
    verbose=1
)

# Bloco de Código 3: Avaliação Detalhada do Modelo CNN

from sklearn.metrics import classification_report, confusion_matrix

# Carregar o melhor modelo CNN salvo
best_model_cnn = keras.models.load_model("../models/best_model_cnn.keras")
print("Melhor modelo CNN carregado com sucesso!")

# Avaliar o modelo nos dados de teste
test_loss_cnn, test_acc_cnn = best_model_cnn.evaluate(X_test_cnn, y_test, verbose=2)
print(f"\nAcurácia no conjunto de teste para o modelo CNN: {test_acc_cnn * 100:.2f}%")

# Fazer previsões no conjunto de teste
y_pred_cnn = best_model_cnn.predict(X_test_cnn)
y_pred_cnn = np.argmax(y_pred_cnn, axis=1)

# Gerar o relatório de classificação
print("\nRelatório de Classificação para o Modelo CNN:\n")
print(classification_report(y_test, y_pred_cnn, target_names=genres))

# Gerar a matriz de confusão
conf_matrix_cnn = confusion_matrix(y_test, y_pred_cnn)

# Plotar a matriz de confusão
plt.figure(figsize=(12,10))
sns.heatmap(conf_matrix_cnn, annot=True, fmt='d', cmap='Blues',
            xticklabels=genres, yticklabels=genres)
plt.title("Matriz de Confusão - Modelo CNN")
plt.ylabel('Classe Verdadeira')
plt.xlabel('Classe Predita')
plt.show()

# Plota acurácia do treinamento e validação
plt.figure(figsize=(14, 5))

# Acurácia
plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='Acurácia Treino')
plt.plot(history_cnn.history['val_accuracy'], label='Acurácia Validação')
plt.title('Acurácia do Modelo CNN')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

# Perda
plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['loss'], label='Perda Treino')
plt.plot(history_cnn.history['val_loss'], label='Perda Validação')
plt.title('Perda do Modelo CNN')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.show()
```

## Visualização e Salvamento do Modelo com Redução de Taxa de Aprendizado

### Objetivo:
- Plota as curvas de treinamento e validação para acurácia e perda.
- Salva o modelo treinado com o menor erro de validação.

### Trechos Importantes:
- `plt.subplot`: Cria gráficos comparando as métricas ao longo das épocas.
- `best_model.save`: Salva o modelo final com a menor perda de validação.

## Treinamento e Avaliação de um Modelo CNN

### Arquitetura da CNN:
- Três camadas convolucionais (`Conv2D`) com normalização de lotes (`BatchNormalization`).
- Camadas de pooling modificadas para `pool_size=(2,1)`, mantendo informações horizontais.
- Camadas densas e `Dropout` para evitar overfitting.
- Função `softmax` na saída para classificação multiclasses.

### Treinamento com Callbacks:
- **`EarlyStopping`**: Interrompe o treinamento ao detectar estagnação na validação.
- **`ModelCheckpoint`**: Salva o melhor modelo com base na acurácia de validação.
- **`ReduceLROnPlateau`**: Reduz a taxa de aprendizado se a perda não melhorar.

### Avaliação e Relatório:
- Avaliação final no conjunto de teste.
- Geração do relatório de classificação com métricas detalhadas (precisão, recall, etc.).
- Matriz de confusão visualiza erros por classe.

## Visualização Gráfica

- **Curvas de Treinamento**: Permite identificar overfitting ou underfitting.
- **Matriz de Confusão**: Ajuda a interpretar quais classes têm maior taxa de erro.

Este fluxo é ideal para refinar modelos de classificação de áudio, permitindo ajustes iterativos e avaliação detalhada.
