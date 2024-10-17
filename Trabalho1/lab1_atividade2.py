import cv2
import numpy as np
import sys

sys.path.append('./funcoes')

import visaoComputacional as visco

# abri arquivos de videos
cap1 = cv2.VideoCapture('./Trabalho1/Chromakey.mp4')
cap2 = cv2.VideoCapture('./Trabalho1/Clouds.mp4')

n_linhas = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_colunas  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('resultado_atividade2.avi', fourcc, 24, (n_colunas, n_linhas))

# cores de referencia e delta
cor_referencia = np.array([150, 255, 44])
delta = 95

# definido mascaras e matriz de distancia euclidiana
M1 = np.zeros((n_linhas, n_colunas), np.uint8)
M2 = np.zeros((n_linhas, n_colunas), np.uint8)
dist = np.zeros((n_linhas, n_colunas), np.uint8)

# processa cada quadro dos vídeos
ret = True
while ret:

    # lê frame dos vídeos abrindo a porta
    ret, I1 = cap1.read()

    # lê frame dos vídeos das nuvens 
    _, I2 = cap2.read()

    if I1 is None:
        break

    # atribuindo cores BGR da imagem I1
    B = I1[:,:,0]
    G = I1[:,:,1]
    R = I1[:,:,2]
    
    dist = np.sqrt((B-cor_referencia[0])**2 + (G-cor_referencia[1])**2 + (R-cor_referencia[2])**2 )
    # Efeito de chromakey
    
    # Obtendo mascara m1
    for y in range(n_linhas):
        for x in range(n_colunas):
            if dist[y, x] <= delta:
                M1[y, x] = 255
            else:
                M1[y, x] = 0

    # Obtendo mascara m2
    M2 = 255 - M1

    # Realizando operacao de AND entre imagens e mascaras
    I3 = cv2.bitwise_and(I1, I1, mask= M2)
    I4 = cv2.bitwise_and(I2, I2, mask = M1)

    # Obtendo I_final através das imagens I3 e I4
    I_final = cv2.add(I3, I4)

    #cv2.imshow('teste', I_final)
    #cv2.waitKey(30)
    video.write(I_final)


cv2.destroyAllWindows()
video.release()