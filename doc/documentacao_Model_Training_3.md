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

**Explicação:**
Este código define uma função chamada `load_data` que carrega dados de treinamento a partir de um arquivo JSON contendo coeficientes MFCC, rótulos de gêneros musicais e a lista de gêneros. Aqui estão as etapas principais:

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

Este código realiza uma análise da distribuição das classes no conjunto de dados carregado:

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

### Explicação do Código
Este código utiliza a função `train_test_split` da biblioteca `scikit-learn` para dividir os dados carregados em dois conjuntos: treino e teste. Essa divisão permite treinar um modelo com uma parte dos dados e avaliá-lo em outra parte para verificar seu desempenho.

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

### Resumo do código

O código acima define e compila um modelo de rede neural profunda para classificação multi-classe usando TensorFlow/Keras. Aqui estão os passos detalhados:

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

### Documentação

Este código realiza o treinamento de um modelo de classificação de gêneros musicais com as seguintes etapas:

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







