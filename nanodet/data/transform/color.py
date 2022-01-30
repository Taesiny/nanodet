import numpy as np
import cv2
import random
import os
import torch
import torchvision.transforms.functional as T
#def double_thresh(img,th_low,th_high):
#    ind0=img<(th_low/255)
#    ind1=img>(th_high/255)
#    ind2=(img>=(th_low/255)) & (img<=(th_high/255))
#    img[ind0]=0
#    img[ind1]=1
#    img[ind2]=0.5
#    return img

# Homomorphic filter class
class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.
    
    High-frequency filters implemented:
        butterworth
        gaussian
    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H
        
        .
    """

    def __init__(self, a = 0.5, b = 1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered

    def filter(self, I, filter_params, filter='gaussian', H = None):
        """
        Method to apply homormophic filter on an image
        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency 
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) is not 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain 
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter=='butterworth':
            H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='gaussian':
            H = self.__gaussian_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='external':
            print('external')
            if len(H.shape) is not 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')
        
        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I = I_fft, H = H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return np.uint8(I)
# End of class HomomorphicFilter

def sharpning(img,kernel=1):
    k= np.zeros((3,3))
    if kernel==1:
        k= np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    elif kernel==2:
        k= np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
    img=cv2.filter2D(img, -1, k)    
    return img

def dft2d(img):
    img=img.astype(np.float32)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    print(fshift)
    magnitude_spectrum = 20*np.log(np.abs(fshift)+1e-8).astype(np.uint8)
    return magnitude_spectrum
def binarize(img,th):
    img = cv2.threshold(img, th/255, 1, cv2.THRESH_BINARY)[1]
    return img

def con_str(img):
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return img

def hist_eq(img):
    img=cv2.equalizeHist(img)
    return img

def morph(img):
    kernel1=np.ones((7, 7),  np.uint8)
    kernel2=np.ones((9, 9),  np.uint8)
    img = cv2.erode(img, kernel1)
    img = cv2.dilate(img, kernel2)
    img = cv2.erode(img, kernel2)
    return img

def blur(img, blur_type= 'normal'):
    if blur_type =='normal':
        img = cv2.blur(img, (5,5))
    elif blur_type =='gaussian':
        img = cv2.GaussianBlur(img, (5, 5), 0)
    elif blur_type =='median':
        img = cv2.medianBlur(img, 5)
    elif blur_type =='bilateral':
        img = cv2.bilateralFilter(img, 9, 75, 75)
    else:
        raise NotImplementedError
    return img

def laplacian(img):
    img = cv2.Laplacian(img,cv2.CV_32F)
    return img
    
def random_brightness(img, delta):
    img += random.uniform(-delta, delta)
    return img

#def homomorphic_filter(img, d0=1, intensity=1, rh=2, c=4, h=2.0, l=0.5):
#    rows, cols = img.shape
#    gray_fft = np.fft.fft2(img)
#    gray_fftshift = np.fft.fftshift(gray_fft)
#    dst_fftshift = np.zeros_like(gray_fftshift)
#    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows//2, rows//2))
#    D = np.sqrt(M ** 2 + N ** 2)
#    Z = (rh - intensity) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + intensity
#    dst_fftshift = Z * gray_fftshift
#    dst_fftshift = (h - l) * dst_fftshift + l
#    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
#    dst_ifft = np.fft.ifft2(dst_ifftshift)
#    dst = np.real(dst_ifft)
#    dst = np.uint8(np.clip(dst, 0, 255))
#    return dst

def log_cor(img):
    img = np.log(1+10*img)/np.log(11)
    return img

def clahe(img, limit=3, grid=(7,7), gray=False):
    if (len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray = True
    
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))

    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if gray: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return np.uint8(img) 

def gamma_cor(img, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(img, table)



def random_contrast(img, alpha_low, alpha_up):
    img *= random.uniform(alpha_low, alpha_up)
    return img


def random_saturation(img, alpha_low, alpha_up):
    hsv_img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    hsv_img[..., 1] *= random.uniform(alpha_low, alpha_up)
    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return img


def normalize(meta, mean, std):
    img = meta['img'].astype(np.float32)
    mean = np.array(mean, dtype=np.float64).reshape(1, -1)
    stdinv = 1 / np.array(std, dtype=np.float64).reshape(1, -1)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    meta['img'] = img
    return meta


def _normalize(img, mean, std):
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 1) / 255
    std = np.array(std, dtype=np.float32).reshape(1, 1, 1) / 255
    img = (img - mean) / std
    return img

def blur(x):
    out= T.gaussian_blur(x,kernel_size=(5, 5))
    return out

def sharpening(x,factor=2):
    out= T.adjust_sharpness(x,sharpness_factor=factor)
    return out

def histo_equ(x):
    out= T.equalize(x.to(dtype=torch.uint8))
    return out.to(dtype=torch.float32)

def adj_brightness(x,factor=0.1):
    out= T.adjust_brightness(x,brightness_factor=factor)
    return out
  
def adj_contrast(x,factor=2):
    out= T.adjust_contrast(x,contrast_factor=factor)
    return out

def auto_contrast(x):
    out= T.autocontrast(x)
    return out

def adj_gamma(x,factor=0.1):
    out= T.adjust_gamma(x,gamma=factor)
    return out

def log_cor_T(x):
    c = 255 / torch.log(1 + torch.max(x))
    out = c * (torch.log(x + 1))
    return out

def color_aug_and_norm(meta, kwargs):
#    if kwargs['histogramm_equalization']:
#        meta['img'] = hist_eq(meta['img'])
#    if kwargs['homomorphic_filter']:
#        homo_filter = HomomorphicFilter(a = 1, b = 1) # a = 0.75, b = 1.25
#        meta['img'] = homo_filter.filter(I=meta['img'],filter_params=[100,2])
#    if kwargs['clahe']:
#        meta['img'] = clahe(meta['img'])
#    if 'gamma_correction' in kwargs:
#        meta['img'] = gamma_cor(meta['img'],kwargs['gamma_correction'])
#    if 'sharpning' in kwargs:
#        meta['img'] = sharpning(meta['img'],kwargs['sharpning'])
#    if kwargs['dft']:
#        meta['img'] = dft2d(meta['img'])
    img = meta['img'].astype(np.float32) / 255
#    if kwargs['laplacian']:
#        img=laplacian(img)
#    if 'binarization_th' in kwargs:
#        img1 = binarize(img,kwargs['binarization_th'][0])
#        img2 = binarize(img,kwargs['binarization_th'][1])
#        if kwargs['morph']:
#            img1=morph(img1)
#            img2=morph(img2)
#        img=(img1+img2)/2
#    if 'blur' in kwargs:
#        img = blur(img,kwargs['blur'])
#    if kwargs['contrast_streching']:
#        img = con_str(img)
#    if kwargs['log_correction']:
#        img = log_cor(img)


        
    if 'brightness' in kwargs and random.randint(0, 1):
        img = random_brightness(img, kwargs['brightness'])

    if 'contrast' in kwargs and random.randint(0, 1):
        img = random_contrast(img, *kwargs['contrast'])

    if 'saturation' in kwargs and random.randint(0, 1):
        img = random_saturation(img, *kwargs['saturation'])
    # cv2.imshow('trans', img)
    # cv2.waitKey(0)
    
    if 'autoaug_value' in kwargs:
        img = torch.from_numpy(img)
        img = img[None, :]
        out1= blur(img)
        out2= sharpening(img,factor=0)
        out3= sharpening(img,factor=2)
        out4= histo_equ(img)
        out5= adj_brightness(img,factor=0.1)
        out6= adj_brightness(img,factor=2)
        out7= adj_contrast(img,factor=0.1)
        out8= adj_contrast(img,factor=2)
        out9= auto_contrast(img)
        out10= adj_gamma(img,factor=0.1)
        out11= adj_gamma(img,factor=2)
#        out12= log_cor_T(img)
        a1= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['a1']))
        a2= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['a2']))
        a3= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['a3']))
        a4= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['a4']))
        a5= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['a5']))
        a6= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['a6']))
        a7= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['a7']))
        a8= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['a8']))
        a9= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['a9']))
        a10= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['a10']))
        a11= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['a11']))
#        a12= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['a11']))

        b0= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['b0']))
        b1= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['b1']))
        b2= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['b2']))
        b3= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['b3']))
        b4= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['b4']))
        b5= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['b5']))
        b6= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['b6']))
        b7= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['b7']))
        b8= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['b8']))
        b9= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['b9']))
        b10= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['b10']))
        b11= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['b11']))
#        b12= torch.sigmoid(torch.tensor(kwargs['autoaug_value']['b11']))       
        
        sum_b=b0+b1+b2+b3+b4+b5+b6+b7+b8+b9+b10+b11
        
        
        img = b0/sum_b*img+b1/sum_b*(a1*img+(1-a1)*out1)+b2/sum_b*(a2*img+(1-a2)*out2)+b3/sum_b*(a3*img+(1-a3)*out3)+b4/sum_b*(a4*img+(1-a4)*out4)+b5/sum_b*(a5*img+(1-a5)*out5)+b6/sum_b*(a6*img+(1-a6)*out6)+b7/sum_b*(a7*img+(1-a7)*out7)+b8/sum_b*(a8*img+(1-a8)*out8)+b9/sum_b*(a9*img+(1-a9)*out9)+b10/sum_b*(a10*img+(1-a10)*out10)+b11/sum_b*(a11*img+(1-a11)*out11)
        img=img[0].numpy()
    img = _normalize(img, *kwargs['normalize'])
    meta['img'] = img
    return meta


#test_image = cv2.imread('000000_10.png', cv2.IMREAD_GRAYSCALE)
#print(test_image.shape)
#cv2.imshow('test_image1',test_image)
#test_image=test_image*0
#
#
#from PIL import Image
#
#with Image.open('ADE_train_00000001.png') as io:
#    seg = np.array(io)

# Obtain the segmentation mask, bult from the RGB channels of the _seg file
#R = seg[:,:,0];
#G = seg[:,:,1];
#B = seg[:,:,2];
#ObjectClassMasks = (R/10).astype(np.int32)*256+(G.astype(np.int32))
#ObjectClassMasks[400,340]
#test_image=hist_eq(test_image)
#homo_filter = HomomorphicFilter(a = 1, b = 1) # a = 0.75, b = 1.25
#test_image = homo_filter.filter(I=test_image,filter_params=[100,2])
#cv2.imshow('test_image1',test_image)
#test_image=gamma_cor(test_image,1.5)
#print(test_image)
#test_image=sharpning(test_image,1)
#test_image=test_image.astype(np.float32) / 255


#test_image = random_brightness(test_image,0.4)
#test_image1= binarize(test_image,130)
#test_image1= morph(test_image1)
#test_image2= binarize(test_image,100)
#test_image2= morph(test_image2)
#cv2.imshow('test_image1',test_image1)
#cv2.imshow('test_image2',test_image2)
#test_image=(test_image1+test_image2)/2
#test_image = laplacian(test_image)
#print(test_image.shape)
#test_image = blur(test_image,'normal')
# print(test_image)
# print('max',test_image.max(),'min',test_image.min())
#cv2.imshow('test_image',test_image)
#cv2.imwrite(os.path.join(r'D:\Workspace\nanodet\nanodet\data\transform' , 'histo_equ1.png'), test_image*255)


#cv2.imshow('test_image',test_image)
#cv2.imwrite(os.path.join(r'D:\Workspace\nanodet\nanodet\data\transform' , 'log_cor.png'), test_image*255)

#test_image1=dft2d(test_image)
#cv2.imshow('test_image1',test_image1)
#cv2.imshow('test_image2',img_filtered)
#cv2.imshow('test_image2',test_image1-test_image)



# PCA
# matrix_test = None
# for image in os.listdir('C:/Users/zm3171/Desktop/test/211102/data'):
#     if image.endswith('.png'):
#         imgraw = cv2.imread(os.path.join('C:/Users/zm3171/Desktop/test/211102/data', image), 0)
#         imgvector = imgraw.reshape(72*1024)
#         try:
#             matrix_test = np.vstack((matrix_test, imgvector))
#         except:
#             matrix_test = imgvector


# mean, eigenvectors = cv2.PCACompute(matrix_test, mean=None)

# V = eigenvectors[:1000]
# temp=np.zeros((len(matrix_test),len(V)))
# for i in range(len(matrix_test)):
#     temp[i] = np.dot(V, matrix_test[i] - mean[0])
# after_trans=np.matmul(temp,V)+mean

# img1=matrix_test[0].reshape(72,1024) /255.0
# img2=after_trans[0].reshape(72,1024)/255.0
# cv2.imshow('1',img1)
# cv2.imshow('2',img2)

#ZCA
# def zca_whiten(X):
#     """
#     Applies ZCA whitening to the data (X)
#     http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/

#     X: numpy 2d array
#         input data, rows are data points, columns are features

#     Returns: ZCA whitened 2d array
#     """
#     assert(X.ndim == 2)
#     EPS = 10e-5

#     #   covariance matrix
#     cov = np.dot(X.T, X)
#     #   d = (lambda1, lambda2, ..., lambdaN)
#     d, E = np.linalg.eigh(cov)
#     #   D = diag(d) ^ (-1/2)
#     D = np.diag(1. / np.sqrt(d + EPS))
#     #   W_zca = E * D * E.T
#     W = np.dot(np.dot(E, D), E.T)

#     X_white = np.dot(X, W)

#     return X_white

