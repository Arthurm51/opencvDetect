# ARTHUR VIEIRA MACHADO
# MATEUS WRZESMSKA MACHADO



Antes de começarmos o tutorial, temos que entender algumas coisas como a estrutura do projeto e os seus comandos essenciais.

ESTRUTURA DE PROJETO 
Para começarmos devemos criar o diretório do projeto. Dentro do Diretório devem ser criadas os seguintes diretórios:
- Treinamento (O local onde os resultados do treinamento do algoritmo serão salvos, o arquivo XML será usado para detecção de objetos); 
- Positivas (A imagem do objeto a ser detectado deve preferencialmente ter uma resolução padrão, não muito grande em relação aos pixels); 
- Negativas (imagens negativas, imagens que não contém o objeto que se deseja detectar);

OPENCV_ANNOTATION
A ferramenta opencv_annotations é muito útil para ajudá-lo a capturar todas as coordenadas retangulares de todas as amostras positivas que você gostaria de usar no treinamento em cascata. A coordenada do retângulo significa (x, y, w, h). O programa irá gerar um arquivo de texto que contém vários dados como as coordenadas retangulares, caminho do arquivo até a imagem e número de imagens.
OPENCV_CREATESAMPLES
Quando já foi terminado o passo de demarcar as imagens positivas então é hora do opencv_createsamples ele é usado para preparar um conjunto de dados de treinamento de amostras positivas e de teste.
OPENCV_TRAINCASCADE
E para finalizar, o opencv_traincascade é o comando dado para que o seu programa comece rodar e sendo assim treinar sua IA usando as imagens amostras positivas e as imagens negativas.

Então agora que já entendemos a teoria vamos para a prática.


1º Passo: Organizar nossa estrutura de projeto.
Uma lista das localizações das imagens no catálogo de imagens negativas deve ser gerada para que esta lista possa ser usada para criar amostras e treinar algoritmos para gerar arquivos com características que devem ser detectadas. O seguinte algoritmo Python pode ser usado para gerar a lista:

import urllib 
import 
numpy as np 
import cv2 
import os 
for file_type in ['negatives']: 
for img in os.listdir(file_type): 
line = file_type+'/'+img+'\n' 
with open('negatives.txt','a') as f: 
f.write(line)
Na posse dos arquivos de imagens negativas ('negativas.txt’), precisamos criar uma lista com as marcações dos objetos encontrados nas imagens positivas.


2º Passo: Utilizar o OPENCV_ANNOTATION para fazer as marcações nas imagens positivas.
Devemos usar o mouse para marcar manualmente o objeto a ser reconhecido, e a seguir pressionar a tecla "c" para aceitar a marca, e após marcar todos os objetos, pressionar a tecla "n" para ir para a próxima imagem. Após a marcação, será gerado um arquivo .txt a partir da imagem e suas respectivas coordenadas.


3º Passo: Utilizar o OPENCV_CREATESAMPLES para criar as amostrar para o treinamento usando determinados parâmetros.
Agora você possui arquivos com imagens negativas e marcações de imagens positivas, então é necessário tirar as amostras para treinamento. Essas amostras podem ser criadas com OPENCV executando o seguinte comando no prompt:
opencv_createsamples -info saida.txt -bg negativas.txt -vec vec.vec -w 24 -h 24
O parâmetro ‘saida.txt’ é as marcações das imagens positivas, o parâmetro ‘negativas.txt’ é a lista gerada pelo python de imagens negativas e o vetor a ser criado foi denominado como ‘vec.vec. Esse arquivo será criado, para ser utilizado no treinamento.


4º Passo: Utilizar o OPENCV_TRAINCASCADE para realizar o treinamento da IA.
Para realizar o treinamento deve ser utilizado o seguinte comando:
opencv_traincascade -data treinamento -vec vec.vev -bg negativas.txt -numPos 450 -numNeg 435 -w 24 -h 24 - precalcValBufSize 1024 -precalcIdxBufSize 1024 -numStages 30 -acceptanceRatioBreakValue 1.0e-5
Durante o treinamento, vários arquivos .XML serão gerados nesta pasta, um para cada estágio e um para arquivos de parâmetros. Após o treinamento, um arquivo denominado cascade.xml será gerado e usado para detecção.
Em seguida, execute o código e veja como funciona:

import numpy as np 
import cv2 
car_cascade = cv2.CascadeClassifier("treinamento/cascade.xml") 
img = cv2.imread("analise04.jpg") 
height, width, c = img.shape 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
objetos = car_cascade.detectMultiScale(gray, 1.2, 5) 
print(objetos) 
for (x,y,w,h) in objetos: 
cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) 
cv2.imshow('Analise', img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

