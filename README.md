# Trabalho-Pratico-Final-de-IA
Trabalho Prático Final de Inteligência Artificial: Resolvendo um Problema Real com IA



Problema de escolhido:

Classificar músicas em gêneros musiciais usando MLP.

Tipo de problema: Multi Class Clasification




dataset utilizado: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification



//////////////////////////////////////////


sobre o data set:


Contexto:

Música. 

Os especialistas vêm tentando há muito tempo entender gostos musicais das pessoas e o que diferencia uma música da outra. Além dissom, procuravam também formas de visualizar o som, e de entender o que torna um hit diferente de outro.


Conteúdo:

gêneros originais - Uma coleção de 10 gêneros com 100 arquivos de áudio cada, todos com duração de 30 segundos (o famoso conjunto de dados GTZAN, o MNIST de sons)
imagens originais - Uma representação visual para cada arquivo de áudio. Uma forma de classificar dados é por meio de redes neurais. Como os NNs (como o CNN, o que usaremos hoje) geralmente aceitam algum tipo de representação de imagem, os arquivos de áudio foram convertidos em espectrogramas Mel para tornar isso possível.
2 Arquivos CSV – Contendo recursos dos arquivos de áudio. Um arquivo tem para cada música (30 segundos de duração) uma média e uma variância calculadas sobre vários recursos que podem ser extraídos de um arquivo de áudio. O outro arquivo tem a mesma estrutura, mas as músicas foram divididas antes em arquivos de áudio de 3 segundos (aumentando assim em 10 vezes a quantidade de dados que alimentamos em nossos modelos de classificação). Com dados, mais é sempre melhor.



/////////////////////////////////////////////////////////////



# Objetivo a ser alcançado com nosso projeto


Com esse projeto, propomos uma aplicação da IA em um campo ainda pouco explorado por assim dizer, o campo da música. Dada sua vasta complexidade e magnitude de dados não diretamente correlacionados entre si, a penetração da IA neste campo tem sido muitas vezes exemplos de desafios modernos para computação, dado o desbalan'ço entre os tipos de problemas que computadores foram projetos para resolver, e a natureza dos problemas musicais, que muitas vezes fogem da lógica de Turing. Ao início do semestre realizamos uma atividade semelhante criando um modelo para classificação de músicas como sendo populares ou não, utilzando neste processo uma base de dados imensa do spotify com diversas variáveis. Neste primeiro projeto, obtivemos uma acurácia baixíssima do modelo, pouco acima de 50%, mas essa experiência acabou tornando-se um pilar para nossa compreensão dos desafios modernos enfrentados pela IA e a computação geral. Além disso, nos permitiu exercer e aprofundar nosso conhecimentos e entendimentos sobre o tema, nos permitindo então realizar neste segundo projeto um plano de atacar novamente um problema músical, mas dessa vez, usando uma estratégia que favoreça a lógica do modelo e princípios computacionais. Nessa linha de pensamento, nos propomos a desenvovler um modelo de classificação de gênero musical baseado em aprendizado supervisionado, que faria uso de espectogramas e MFCCs para discernimento de padrões no dataset.


**Objetivo sucinto:** Desenvolver em cima daquilo que desenvolvemos para o primeiro projeto da disicplina um modelo que pudesse obter uma acurácia superior de 80%.


**Técnica mais adequada encontrada:** Aprendizado Supervisionado: para classificação.


**Problema que nos propomos a resovler:** Modelos de IA em contextos de aplicação ligados a música tendem a ter dificuldade no tratamento e processamento de dados dada não só a magnitude dos dados, como também a natureza multi-direcional destes dados.




/////////////////////////////////////////////////////////////

# Como nosso modelo de classificação de gêneros musicais funciona!

### **1. Extração de características importantes do áudio**

A música que ouvimos é um arquivo MP3 em formato de aúdio. Todavia, precisamos transformar esse som em dados que o computador possa entender e processar. Fazemos isso através da **extração de características**:

- **MFCCs (Coeficientes Cepstrais de Frequência Mel):** São valores que representam as características mais importantes do áudio. Eles capturam informações sobre as frequências que contituem aquela música e como elas variam ao longo do tempo. Esses MFCCs podem ser interpretados como uma "impressão digital" sonora da música.

Ao fim do notebook 3, adotamos uma estratégia no desenvolvimento do nosso modelo para aumentar significativamente nossa acurácia, nesse sentido, desenvolvemos um modelo melhorado com **CNNs**. Para aplicar o modelo CNN decidimos representar a música de forma visual através de **espectrogramas** ou **mel-espectrogramas**, que são basicamente formas de transformar as características do áudio em imagens que possam ser visualizadas e com isso entender como as frequências variam ao longo do tempo. Com isso, aproveitamos da vantagem da rede neural convulocional em ser ótima para reconhecer padrões visuais e convertemos o aúdio para um formato que pudesse aproveitar desta vantagem.

### **2. Preparação dos Dados para o Modelo**


Depois de extrair esses MFCCs, organizamos os dados de forma que o modelo possa ser alimentado por eles:

- **Segmentação:** Dividimos a música em pequenos pedaços (segmentos) para capturar variações ao longo da música.

- **Normalização:** Ajustamos os valores dos dados e espectogramas para que fiquem em uma mesma escala, facilitando o aprendizado do modelo.

- **Redimensionamento das Imagens:** Para que o modelo conseguisse processar as imagens corretamente, os espectrogramas foram redimensionados para um tamanho padrão, garantindo que todas imagens usadas para alimentar o modelo tivessem a mesma dimensão.




### **3. Treinamento do Modelo de Rede Neural Convolucional (CNN)**



Utilizamos os dados preparados para treinar o modelo.

- **Aprendizado de Padrões:** O modelo analisa os MFCCs e espectogramas de várias músicas e aprende a associar certos padrões sonoros evidenciados pelos MFCC a gêneros musicais específicos. Por exemplo, pode aprender que músicas de rock têm padrões diferentes da músicas de jazz.

- **Arquitetura da CNN:** Nosso modelo melhorado, o CNN usa **camadas convolucionais** para identificar características locais, **camadas de pooling** para reduzir a dimensionalidade das imagens e **camadas densas** para interpretar as informações extraídas pelas camadas convolucionais e a partir delas fazer a previsão do gênero. Por último, a **camada de saída**, gera as predições finais.



### **4. Classificação de Novas Músicas**

Para classificar uma nova música, o modelo passa pelo fluxo abaixo:

1. **Extração de MFCCs:** Aplica-se o mesmo processo de extração de MFCCs na nova música.

2. **Pré-processamento:** Aplicamos o pré-processamento, consistindo de normalizar e redimensionar os dados , para garantir que a imagem seja compatível com o que a CNN foi treinada para entender.

3. **Entrada no Modelo:** Fornecemos os MFCCs e espectogramas aos modelos.

4. **Previsão do Gênero:** O modelo analisa os padrões nos MFCCs e espectogramas e calcula a probabilidade de a música pertencer a cada um dos gêneros no qual foi treinado.



### **Por que Funciona?!**

- **Padrões Sonoros e Visuais:** Cada gênero musical tem características sonoras únicas que por sua vez são manifestadas visualmente nos espectrogramas. A CNN em particular é ótima em detectar esses padrões e associá-los a um gênero específico, mas os outros modelos que aplicamos também fazem isso através dos MFCCs.
  
- **Aprendizado:** O modelo vai ajustando seus parâmetros para melhorar a precisão das previsões com base nos dados que foi alimentado. Assim sendo, quanto maior a gama de dados que fornecemos para alimentar o modelo, melhor.




### **Resumo**



Em resumo, o processo é:

1. **Transformar** o áudio em dados computáveis para representar características das músicas.

2. **Treinar** o modelo para reconhecer padrões nos dados que correspondem a diferentes gêneros musicais.

3. **Usar** o modelo treinado para **prever** o gênero de novas músicas com base nos padrões identificados.


Poe analogia, podemos abstrair este processo para como se fosse ensinar alguém a diferenciar estilos musicais mostrando exemplos e destacando características típicas de cada gênero. 







