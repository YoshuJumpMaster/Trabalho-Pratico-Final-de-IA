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


**Técnica mais adequada encontrada:** Aprendizado Supervisionado: para classificação ou regressão.


**Problema que nos propomos a resovler:** Modelos de IA em contextos de aplicação ligados a música tendem a ter dificuldade no tratamento e processamento de dados dada não só a magnitude dos dados, como também a natureza multi-direcional destes dados.




/////////////////////////////////////////////////////////////


# Como nosso modelo de classificação de gêneros musical!

### **1. Extração de Características Importantes do Áudio**

A música que ouvimos é um arquivo MP3 em formato de aúdio. Todavia, precisamos transformar esse som em dados que o computador possa entender e processar. Fazemos isso através da **extração de características**:

- **MFCCs (Coeficientes Cepstrais de Frequência Mel):** São valores que representam as características mais importantes do áudio. Eles capturam informações sobre as frequências que contituem aquela música e como elas variam ao longo do tempo. Esses MFCCs podem ser interpretados como uma "impressão digital" sonora da música.

### **2. Preparação dos Dados para o Modelo**

Depois de extrair esses MFCCs, organizamos os dados de forma que o modelo possa ser alimentado por eles:

- **Segmentação:** Dividimos a música em pequenos pedaços (segmentos) para capturar variações ao longo da música.

- **Normalização:** Ajustamos os valores dos dados para que fiquem em uma mesma escala, facilitando o aprendizado do modelo.

### **3. Treinamento do Modelo de Rede Neural**

Utilizamos os dados preparados para treinar o modelo.

- **Aprendizado de Padrões:** O modelo analisa os MFCCs de várias músicas e aprende a associar certos padrões sonoros evidenciados pelos MFCC a gêneros musicais específicos. Por exemplo, pode aprender que músicas de rock têm padrões diferentes da músicas de jazz.

### **4. Classificação de Novas Músicas**

Para classificar uma nova música, o modelo passa pelo fluxo abaixo:

1. **Extração de MFCCs:** Aplica-se o mesmo processo de extração de MFCCs na nova música.

2. **Entrada no Modelo:** Fornecemos esses MFCCs ao modelo.

3. **Previsão do Gênero:** O modelo analisa os padrões nos MFCCs e calcula a probabilidade de a música pertencer a cada um dos gêneros no qual foi treinado.

### **Por que Funciona?!**

- **Padrões Sonoros:** Cada gênero musical possui características sonoras únicas e especiais. Dessa forma, nosso modelo é capaz de aprender a identificar essas características nos dados que fornecemos a ele.

- **Aprendizado:** O modelo vai ajustando seus parâmetros para melhorar a precisão das previsões com base nos dados que foi alimentado. Assim sendo, quanto maior a gama de dados que fornecemos para alimentar o modelo, melhor.

### **Resumo**

Em resumo, o processo é:

1. **Transformar** o áudio em dados computáveis para representar características das músicas.

2. **Treinar** o modelo para reconhecer padrões nos dados que correspondem a diferentes gêneros musicais.

3. **Usar** o modelo treinado para **prever** o gênero de novas músicas com base nos padrões identificados.


Poe analogia, podemos abstrair este processo para como se fosse ensinar alguém a diferenciar estilos musicais mostrando exemplos e destacando características típicas de cada gênero. 






