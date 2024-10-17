import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# ----------------------------------------------------------
#Video com a bolinha
#cap = cv2.VideoCapture('./Trabalho1/sequencia1.mp4')

#Video com o globo
cap = cv2.VideoCapture('./Trabalho1/globo_seq.mp4')

n_linhas = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_colunas  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


# ----------------------------------------------------------
# Gera figura
fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=0.9)

fig_width = 20
fig_height = 6
fig.set_size_inches((fig_width/2.54, fig_height/2.54))
fig.tight_layout()

# ----------------------------------------------------------
ret = True

# Seção de variáveis
# IMPLEMENTE O SEU CÓDIGO AQUI

# definindo cor de referencia  e delta para o video com a bolinha
#cor_referencia = np.array([0, 255, 255])
#delta = 170

# definindo cor de referencia  e delta para o video com o globo
cor_referencia = np.array([255, 0, 0])
delta = 180

# Definindo matriz m para imagem binaria
M = np.zeros((n_linhas, n_colunas), np.uint8)

# declarando variaveis para utilizar na extracao da coluna centra e  
# contagem de objeto 
ref_linha = np.zeros(260)
contador = 0
time_elapsed = []
frame = 0
flag = False

while ret:

    # lê frame do vídeo
    ret, I1 = cap.read()
    
    start = cv2.getTickCount()
    frame += 1

    if I1 is None:
        break

    # Algoritmo de detecção de objetos
    # IMPLEMENTE O SEU CÓDIGO AQUI

    b = np.float32(I1[:, :, 0])  # Canal B
    g = np.float32(I1[:, :, 1])  # Canal G
    r = np.float32(I1[:, :, 2])  # Canal R

    # calcula a distancia euclidiana
    dist = np.sqrt(((b - cor_referencia[0])**2) + ((g - cor_referencia[1])**2) + ((r - cor_referencia[2])**2))
    
    for y in range(n_linhas):
        for x in range(n_colunas):
            if dist[y, x] <= delta:
                M[y, x] = 255
            else:
                M[y, x] = 0
    
    # verifica se existe algum pixel no valor de 255  na coluna central 
    if np.any(M[0:449, 399] == 255):
        if not flag:
            contador += 1
            flag = True
    else:
        flag = False
        
    # cria linha de referencia da coluna central
    ref_linha = M[0:449, 399]
    
    end = cv2.getTickCount()

    # Cálculo do tempo decorrido
    time_elapsed = (end - start) / cv2.getTickFrequency()


    # atualiza plot
    ax1.clear()
    ax1.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
    ax1.text(250, -20, f'Frame: {frame}', fontsize=12, color='black')

    ax2.clear()
    ax2.imshow(M, cmap='gray')
    ax2.plot([n_colunas/2, n_colunas/2], [0, n_linhas-1], ':', color = 'pink')
    ax2.text(250, -20, f'Contador: {contador}', fontsize=12, color='black')

    ax3.clear()
    ax3.plot(np.arange(0, ref_linha.size), ref_linha)
    ax3.set_ylim([0, 260])
    ax3.set_xlim([0, 450])
    ax3.grid(True)
    ax3.set_title('Coluna central da\n imagem binária')

    plt.pause(0.05)

# mostra na tela a media do tempo final
print(np.mean(time_elapsed))
print('Processo finalizado!')
cv2.waitKey(0)