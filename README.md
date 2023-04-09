# Classificacao_python

### Este código utiliza os módulos do scikit-learn para aplicar a técnica de classificação em um conjunto de dados de flores iris, utilizando as árvores de decisão e máquinas de vetor suporte (SVM).

<br>


## Pré-processamento
### O conjunto de dados utilizado é carregado pelo módulo load_iris e é composto por 150 amostras de flores iris, sendo que cada amostra possui quatro características medidas em centímetros (comprimento da sépala, largura da sépala, comprimento da pétala e largura da pétala). Os dados são divididos aleatoriamente em conjuntos de treino e teste usando o método train_test_split.

<br>

## Mineração
### Duas técnicas de classificação são aplicadas aos dados: árvores de decisão e SVM. A árvore de decisão é instanciada com uma profundidade máxima de 2 e treinada no conjunto de treino usando o método fit. Em seguida, o método predict é usado para gerar as previsões no conjunto de teste. A acurácia é calculada usando o método accuracy_score do scikit-learn. A visualização da árvore é plotada utilizando o método plot_tree do scikit-learn.

<br>


### A SVM é instanciada com seus parâmetros padrão e também é treinada no conjunto de treino usando o método fit. Em seguida, o método predict é usado para gerar as previsões no conjunto de teste e a acurácia é calculada usando o método accuracy_score.

<br>


## Pós-processamento
### As acurácias das duas técnicas são impressas na tela utilizando a função print, bem como a estrutura da árvore gerada pela técnica de árvore de decisão usando o método export_text.