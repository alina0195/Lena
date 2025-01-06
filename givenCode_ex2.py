
import numpy as np 
import imageio 
import matplotlib.pyplot as plt 
 
io = imageio.imread('./host.png') 
io = np.round(0.299*io[:,:,0] + 0.587*io[:,:,1] + 0.114*io[:,:,2]) 
 
def histograma(io): 
    h=np.zeros(256) 
    for i in range(0,256): 
        h[i]=np.sum(io==i) 
    plt.stem(h,use_line_collection=True) 
    plt.savefig("histograma_gazda.png")
    plt.show()
    return h 


def PSNR(i1, i2):  
    mse = np.mean((i1 - i2) ** 2)  
    if(mse == 0):    
        return np.inf 
    psnr = 20 * np.log10(255 / np.sqrt(mse))  
    return psnr 

def insertie_in_hist(io,mesaj,l,r): 
    N=len(mesaj) 
    m,n=io.shape 
    k=0 
    for i in range(0,m): 
        for j in range(0,n): 
            if k<N: 
                p=io[i,j] 
                if p==l: 
                    io[i,j]=p-mesaj[k] 
                    k=k+1 
                elif p==r: 
                    io[i,j]=p+mesaj[k] 
                    k=k+1 
                elif p<l: 
                    io[i,j]=p-1 
                elif p>r: 
                    io[i,j]=p+1 
    if k<N: 
        print('Spatiu insuficient') 
    return io 

def decodare_in_hist(io,l,r,N): 
    m,n=io.shape 
    mesaj_r=np.zeros(N) 
    k=0 
    for i in range(0,m): 
        for j in range(0,n): 
            if k<N: 
                p=io[i,j] 
                if p==l or p==r: 
                    mesaj_r[k]=0                     
                    k=k+1 
                elif p<l: 
                    if p==l-1: 
                        mesaj_r[k]=1 
                        k=k+1 
                    io[i,j]=p+1                     
                elif p>r: 
                    if p==r+1: 
                        mesaj_r[k]=1 
                        k=k+1 
                    io[i,j]=p-1     
    return io,mesaj_r 


def obtine_rezultate(l=155, r=156, N=4000):
    # np.random.seed(42)
    mesaj=np.random.randint(0,2,N)
    # mesaj=np.random.randint(0,1,N)  
    im=io.copy() 
    im=insertie_in_hist(im,mesaj,l,r) 
    ps=PSNR(im,io) 
    ir=im.copy() 
    ir,mesaj_r=decodare_in_hist(ir,l,r,N) 
    ps2=PSNR(ir,io) 
    errors = np.sum(np.abs(mesaj - mesaj_r))  
    print(f'Parametrii: l={l}, r={r}, N={N}')
    print('PSNR imagine marcata = ',ps) 
    print('PSNR imagine recupearta = ',ps2) 
    print('Erori mesaj recuperat: ', errors)
    print()
    # print('Mesaj initial:', mesaj)
    # print('Mesaj recuperat:', mesaj_r)
    
    return N, ps, ps2, errors
    

# Analyze for different N values
results = []
# l, r = 155, 156 
# l, r = 55, 56 
# l, r = 65, 66 
# l, r = 58, 59 
# l, r = 53, 54 
l, r = 68, 69 

N_values = [1000, 2000, 3000, 4000]

# Generate the histogram of the host image
h = histograma(io)

for N in N_values:
    N, ps, ps2, errors = obtine_rezultate(l,r,N)
    results.append((N, ps, ps2, errors))

print(results)