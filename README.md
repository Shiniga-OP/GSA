# GSA
GSA (Gerador de Sequências Autônomo). Arquitetura de Rede Neural baseada em Transformer, criada para ser compacta e leve em um ambiente limitado. Sem uso de bibliotecas externas.

## sobre os arquivos
a implementação em JS é a implementação mais perto de uma IA real, o GSA é um modelo inacabado, e por ser muito complexo, acaba não aprendendo mais do que padrões simples de linguagem (ou nem isso), seu aprendizado fica estagnado depois de algumas épocas e lotes de treino.

a o arquivo C é uma implementação de parte da minha biblioteca de Redes Neurais, uma tentativa de ganhar mais desempenho utilizando baixo nível, mas incompleta.

## conteúdo 

você pode encontrar no arquivo JS:

Tokenizador SubPalavra.
Camada de normalização.
Camada de atenção.
Camada feedforward.
Bloco Transformer.
Treinador do tokenizador e salvamento com bin das merges.
Modelo GSA padrão.
Treinador do modelo GSA.
Cálculo de parâmetros.

## sobre o futuro
esse é meu modelo de IA mais complexo e maior, utilizando tudo que eu aprendi sobre IA nos últimos anos menos Aprendizado Por Reforço, então talvez algum dia eu o atualize para funcionar corretamente.
