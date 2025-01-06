import numpy as np 
import imageio 
import matplotlib.pyplot as plt 
from scipy.fftpack import dct,idct 
from scipy.ndimage import median_filter 
import random

random.seed(42)

io = imageio.imread('./host.png') 
io = np.round(0.299*io[:,:,0] + 0.587*io[:,:,1] + 0.114*io[:,:,2]) 
im = io.copy() 

lg = imageio.imread('./logo.png') 
lg = np.round(0.299*lg[:,:,0] + 0.587*lg[:,:,1] + 0.114*lg[:,:,2])//255 


def PSNR(i1, i2): 
    mse = np.mean((i1 - i2) ** 2)  # eroarea medie patratica intre im orgi si im procesata
    if(mse == 0):    
        return np.inf   # valorile mari inidca o calitate buna
    psnr = 20 * np.log10(255 / np.sqrt(mse))  # 255 este val max posibila a unui pixel pt imagini pe 8 biti
    return psnr  

def BER(lg, lr, dim1=32, dim2=32):
    # ber = rata de detectie a logo-ului
    # lg = imaginea logo (marcajul)
    # lr = imaginea logo detectata
    return np.sum(np.abs(lg-lr))/dim1/dim2


def insertie_in_DCT(im,lg, p1=3,p2=3,p3=3,p4=4,k=20): 
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

def decodare_in_DCT(im, p1=3,p2=3,p3=3,p4=4): 
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


def get_results(description:str, 
                io,
                im2, 
                lg, 
                p1=3, p2=3, p3=3, p4=4,
                filename_host='host ',
                filename_logo='logo '):
    print(description)
    
    def plot_images(description, io, im2, lg, lr, filename_host, filename_logo):
        plt.figure(figsize=(12, 16))
        plt.subplot(1,2,1)
        plt.imshow(io,cmap='gray') 
        plt.title(description + ' - gazda originala')
        plt.subplot(1,2,2)
        plt.imshow(im2,cmap='gray') 
        plt.title(description + ' - gazda cu logo')
        
        plt.savefig(filename_host)
        # plt.show() 
        # plt.close()  
        
        plt.figure(figsize=(12, 16))
        plt.subplot(1,2,1)
        plt.imshow(lg*255, cmap='gray') 
        plt.title(description + ' - logo original')
        plt.subplot(1,2,2)
        plt.imshow(lr*255,cmap='gray') 
        plt.title(description + ' - logo detectat')
        
        plt.savefig(filename_logo)
        # plt.show()
        # plt.close()  
        
        
    lr=decodare_in_DCT(im2, p1,p2,p3,p4) 

    print(description + ': ber=', BER(lg, lr)) # ber=rata de detectie a logo-ului
    print(description +': psnr=',PSNR(io, im2)) # psnr=calitatea imaginii 
    plot_images(description, io, im2, lg, lr, filename_host, filename_logo)


im=insertie_in_DCT(im,lg) 
im2=im.copy()
get_results(description='Rezultate initiale (3,3), (3,4)',  
            filename_host='./plots/Figure_1_host.png',
            filename_logo='./plots/Figure_1_logo.png',
            io=io, im2=im2, lg=lg)

def attack_a(p1,p2,p3,p4, save_filename_host, save_filename_logo):
    # Schimbare luminozitate 
    global io, im, lg
    im2=im.copy()
    
    im2=im-20 
    im2=im2.astype(np.uint8)
    get_results(description=f'Rezultate a) ({p1},{p2}), ({p3},{p4})',  
            filename_host=save_filename_host,
            filename_logo=save_filename_logo,
            io=io, im2=im2, lg=lg)


def attack_b(p1,p2,p3,p4, save_filename_host, save_filename_logo):
    #  Adaugare zgomot gaussian
    global io, im, lg

    var = 70 
    sigma = var**0.5 
    m,n=im.shape 
    np.random.seed(42)
    gauss = np.random.normal(0,sigma,(m,n))  # seed 42
    im2=im+gauss    
    im2=im2.astype(np.uint8)
    
    get_results(description=f'Rezultate b) ({p1},{p2}), ({p3},{p4})',  
            filename_host=save_filename_host,
            filename_logo=save_filename_logo,
            io=io, im2=im2, lg=lg)


def attack_c(p1,p2,p3,p4,save_filename_host, save_filename_logo):
    # Adaugare zgomot sare-si-piper 
    global io, im, lg
    im2=im.copy()

    m,n=im.shape 
    np.random.seed(42)
    z=np.random.randint(1,101,(m,n)) 
    
    im2[z<2]=255 
    im2[z>98]=0 
    
    get_results(description=f'Rezultate c) ({p1},{p2}), ({p3},{p4})',  
            filename_host=save_filename_host,
            filename_logo=save_filename_logo,
            io=io, im2=im2, lg=lg)


def attack_d(p1,p2,p3,p4,save_filename_host, save_filename_logo):
    # Filtrare cu filtrul median 
    global io, im, lg
    
    im2=median_filter(im,5) 

    get_results(description=f'Rezultate d) ({p1},{p2}), ({p3},{p4})',  
            filename_host=save_filename_host,
            filename_logo=save_filename_logo,
            io=io, im2=im2, lg=lg)

def attack_e(p1,p2,p3,p4,save_filename_host, save_filename_logo):
    # Rotire 90 de grade
    global io, im, lg
    
    m,n=im.shape 
    im2 = im.copy()
    
    for i in range(0,m): 
        for j in range(0,n): 
            im2[n-1-j,i]=im[i,j]
            
    get_results(description=f'Rezultate e) ({p1},{p2}), ({p3},{p4})',  
            filename_host=save_filename_host,
            filename_logo=save_filename_logo,
            io=io, im2=im2, lg=lg)


def attack_f(p1,p2,p3,p4,save_filename_host, save_filename_logo):
    # Compresie jpeg 
    global io, im, lg
    
    imageio.imwrite('marcatDCT.jpg',im, quality=80) 
    im2 = imageio.imread('marcatDCT.jpg') 
    get_results(description=f'Rezultate f) ({p1},{p2}), ({p3},{p4})',  
            filename_host=save_filename_host,
            filename_logo=save_filename_logo,
            io=io, im2=im2, lg=lg)
    

attack_a(3,3,3,4,'./plots/Figure_a_3334_host.png','./plots/Figure_a_3334_logo.png')
attack_a(4,5,5,5,'./plots/Figure_a_4555_host.png','./plots/Figure_a_4555_logo.png')

attack_b(3,3,3,4,'./plots/Figure_b_3334_host.png','./plots/Figure_b_3334_logo.png')
attack_b(4,5,5,5,'./plots/Figure_b_4555_host.png','./plots/Figure_b_4555_logo.png')

attack_c(3,3,3,4,'./plots/Figure_c_3334_host.png','./plots/Figure_c_3334_logo.png')
attack_c(4,5,5,5,'./plots/Figure_c_4555_host.png','./plots/Figure_c_4555_logo.png')


attack_d(3,3,3,4,'./plots/Figure_d_3334_host.png','./plots/Figure_d_3334_logo.png')
attack_d(4,5,5,5,'./plots/Figure_d_4555_host.png','./plots/Figure_d_4555_logo.png')


attack_e(3,3,3,4,'./plots/Figure_e_3334_host.png','./plots/Figure_e_3334_logo.png')
attack_e(4,5,5,5,'./plots/Figure_e_4555_host.png','./plots/Figure_e_4555_logo.png')

attack_f(3,3,3,4,'./plots/Figure_f_3334_host.png','./plots/Figure_f_3334_logo.png')
attack_f(4,5,5,5,'./plots/Figure_f_4555_host.png','./plots/Figure_f_4555_logo.png')
