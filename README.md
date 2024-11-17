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