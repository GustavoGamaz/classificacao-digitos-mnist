# Classificação de Dígitos com Rede Neural (MNIST)

## Objetivo
Desenvolver um modelo de rede neural para classificar imagens de dígitos manuscritos utilizando o dataset MNIST.

## Problema
Aplicar conceitos introdutórios de machine learning e redes neurais em um problema clássico de classificação de imagens.

## Ferramentas utilizadas
- Python
- TensorFlow
- Keras
- Matplotlib

## Etapas do projeto
1. Carregamento do dataset MNIST
2. Normalização das imagens
3. Redimensionamento dos dados
4. Construção da rede neural
5. Treinamento do modelo
6. Avaliação da acurácia
7. Visualização das previsões

## Resultados
O modelo foi treinado para classificar imagens de dígitos manuscritos e apresentou boa capacidade de predição no conjunto de teste.

## Dataset utilizado

O projeto utiliza o dataset MNIST, um conjunto de dados clássico de imagens de dígitos manuscritos amplamente utilizado para treinamento e avaliação de modelos de machine learning.

O dataset contém:

- 60.000 imagens de treinamento
- 10.000 imagens de teste
- imagens em escala de cinza de 28x28 pixels representando dígitos de 0 a 9

Os dados são carregados automaticamente utilizando:

```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

## Arquivos do projeto
- Código Python do modelo
- Imagem com exemplos de previsões
- Relatório da atividade

## Resultado do modelo

Exemplo de previsões da rede neural para dígitos do dataset MNIST.

![Resultado do modelo](resultado_modelo.png)

## Resultados

O modelo alcançou aproximadamente **97% de acurácia** no conjunto de teste do dataset MNIST, demonstrando boa capacidade de classificação de dígitos manuscritos utilizando uma rede neural simples.

## Autor
Gustavo Vitor Santos da Gama
