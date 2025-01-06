import numpy as np 
import imageio 
import matplotlib.pyplot as plt 
from scipy.fftpack import dct,idct 
from scipy.ndimage import median_filter 

io = imageio.imread('./Lena_512x512.png') 
io = np.round(0.299*io[:,:,0] + 0.587*io[:,:,1] + 0.114*io[:,:,2]) 
im = io.copy() 

lg = imageio.imread('./logo.png') 
lg = np.round(0.299*lg[:,:,0] + 0.587*lg[:,:,1] + 0.114*lg[:,:,2])//255 


"""""
PSNR = Peak signal to noise ratio 
este o măsură utilizată pentru a evalua calitatea 
unei imagini compresate sau procesate în comparație cu 
imaginea originală. Este frecvent utilizată în compresia 
de imagini pentru a determina cât de mult s-a degradat 
imaginea după procesare.
"""
def PSNR(i1, i2): 
    mse = np.mean((i1 - i2) ** 2)  # eroarea medie patratica intre im orgi si im procesata
    if(mse == 0):    
        return np.inf   # valorile mari inidca o calitate buna
    psnr = 20 * np.log10(255 / np.sqrt(mse))  # 255 este val max posibila a unui pixel pt imagini pe 8 biti
    return psnr  



""""
DCT  = Discrete Cosine Transform
este o transformare matematică utilizată pentru 
a converti pixelii în componente frecvențiale. 
Este folosită frecvent în compresia imaginilor.

Cum funcționează:

Intrare: O matrice de pixeli dintr-o imagine.
Ieșire: O matrice de coeficienți care reprezintă frecvențele. 
Coeficienții cu valori mari corespund frecvențelor mai joase (structuri globale), 
iar cei cu valori mici corespund detaliilor fine.
"""

def insertie_in_DCT(im,lg): 
    p1=3  # primul coef
    p2=3  # primul coef
    p3=3  # al doilea coef
    p4=4  # al doilea coef
    k=20 
    m,n=io.shape   # !!! ar trebui sa fie im.shape (dimensiunile gazdei)   
    for i in range(0,32): 
        for j in range(0,32): 
            b=im[16*i:16*(i+1),16*j:16*(j+1)] 
            b=dct(dct(b.T, norm = 'ortho').T, norm = 'ortho') 
            if lg[i,j]==0: # daca pixelul curent din logo e 0
                if b[p1,p2]<b[p3,p4]: #daca primul coef < al coilea coef din dct, se interschimba cele doua valori
                    t=b[p1,p2] 
                    b[p1,p2]=b[p3,p4] 
                    b[p3,p4]=t 
                if b[p1,p2]-b[p3,p4]<k: # daca de la coef1 la coef2 e o dist < k, se mareste distanta cu valoarea k 
                    b[p1,p2]=b[p1,p2]+k/2 
                    b[p3,p4]=b[p3,p4]-k/2 
            else: # daca pixelul curent din logo e 1
                if b[p1,p2]>b[p3,p4]: # daca primul coef>al doiela coef, se interschimba
                    t=b[p1,p2] 
                    b[p1,p2]=b[p3,p4] 
                    b[p3,p4]=t 
                if b[p3,p4]-b[p1,p2]<k:  # daca dist de la al doilea coef pana la primul < k, distanta se mareste cu k
                    b[p1,p2]=b[p1,p2]-k/2; 
                    b[p3,p4]=b[p3,p4]+k/2; 
            # dupa toate modificarile privind pixelii, se inverseaza dct si se da copy paste in gazda
            # din toate aceste reguli ar trebui sa ne dam seama ce biti au alcatuit imaginea logo ascunsa in gazda
            b=idct(idct(b.T, norm = 'ortho').T, norm = 'ortho') 
            im[16*i:16*(i+1),16*j:16*(j+1)]=b 
    return im.astype(np.uint8)


def decodare_in_DCT(im): 
    p1=3 
    p2=3 
    p3=3 
    p4=4 
    m,n=io.shape  
    lr=np.zeros([32,32]) 
    for i in range(0,32): 
        for j in range(0,32): 
            b=im[16*i:16*(i+1),16*j:16*(j+1)] 
            b=dct(dct(b.T, norm = 'ortho').T, norm = 'ortho') 
            if b[p1,p2]>b[p3,p4]: 
                lr[i,j]=0
            else: 
                lr[i,j]=1 
    return lr.astype(np.uint8) 


im=insertie_in_DCT(im,lg) 
im2=im.copy() 
m,n=im.shape 

lr=decodare_in_DCT(im2) 
print(PSNR(io,im2)) # psnr=calitatea imaginii (distorsiunea) dupa insertie logo (im2) in raport cu formatul original(io)
ber=np.sum(np.abs(lg-lr))/32/32 
print('ber=', ber) # ber=rata de detectie a logo-ului

plt.subplot(1,2,1).imshow(io,cmap='gray') 
plt.subplot(1,2,2).imshow(im2,cmap='gray') 
plt.show() 
plt.subplot(1,2,1).imshow(lg*255, cmap='gray') 
plt.subplot(1,2,2).imshow(lr*255,cmap='gray') 
plt.show()



