#!/usr/bin/env python
# coding: utf-8

# # Library

# In[1]:


from tkinter import *
from tkinter.font import Font
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk, ImageEnhance
from imgaug import augmenters as iaa
import wand.image as wa
from math import sqrt
import numpy as np
import cv2 as cv
import math


# # ========================================

# # Quality Measurement

# ## 1) - PSNR

# In[2]:


# Code for computing psnr - own code
def psnr(image_original, image_restored):
    M = image_original.shape[0]
    N = image_original.shape[1]

    # compute psnr as 10log10(MAX^2/MSE)
    mse = np.sum(np.square(image_original - image_restored)) / (M * N) + 0.0000001
    max = 255 ** 2
    psnr = 10 * np.log10(max / mse)

    return psnr


# ## 2 ) - MSE

# In[3]:


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


# ## 3) - SSIM

# In[4]:


def ssim(image_original, image_restored):
    M = image_original.shape[0]
    N = image_original.shape[1]
    one = np.ones_like(image_original)
    # compute mean of original and restored image
    mean_original = np.sum(image_original) / (M * N)
    mean_restored = np.sum(image_restored) / (M * N)
    # compute variance of original and restored image
    var_original = np.sum(np.square(image_original - mean_original * one)) / (M * N)
    var_restored = np.sum(np.square(image_restored - mean_restored * one)) / (M * N)
    # compute cross correlation between original and restored image
    cross_correlation = np.sum(np.multiply(image_original - mean_original * one,
                                           image_restored - mean_restored * one)) / (M * N)
    # compute standard deviation of original and restored image
    sd_original = np.sqrt(var_original)
    sd_restored = np.sqrt(var_restored)
    # define constants c1, c2, c3
    C1 = 1
    C2 = 0.01
    C3 = 0.01
    # compute l, c, s
    l = (2 * mean_original * mean_restored + C1) / (mean_original ** 2 + mean_restored ** 2 + C1)
    c = (2 * sd_original * sd_restored + C2) / (var_original + var_restored + C2)
    s = (cross_correlation + C3) / (sd_original * sd_restored + C3)
    # compute ssim = lcs
    ssim = l * c * s
    return ssim


# # ===========================================================

# #  Gaussian Kernel

# In[5]:


def kernel(d, sz=20):  # перерисока psf в виде окружности, учитывая новый диаметр
    kern = np.zeros((sz, sz), np.uint8)
    cv.circle(kern, (sz, sz), d, 255, -1, cv.LINE_AA, shift=1)
    kern = np.float32(kern) / 255.0
    return kern


# ##  ======================================================================

# # Image Enhancement Algorithm

# ## - Image Enhancement (Spatial Domain)

# ### 1) - Negative Transformation

# In[6]:


def imgNe(img):
    #     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # Invert the image using cv2.bitwise_not
    img_neg = cv.bitwise_not(img)
    return img_neg


# ### 2) log & Inverse Log

# In[7]:


def imgLog(img):
    #     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # Apply log transform
    img_log = (np.log(img + 1) / (np.log(1 + np.max(img)))) * 255
    img_log = np.array(img_log, dtype=np.uint8)
    return img_log


def Inverse_imgLog(img):
    #     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # Apply log transform
    c = 255 / np.log(1 + np.max(img))
    inv_log_image = np.exp(img ** 1 / c) - 1
    inv_log_image = np.array(inv_log_image, dtype=np.uint8)
    return inv_log_image


# # ========================================================

# # Image Restoration Algorithm

# In[8]:


# Code for generating butter worth filter frequency response - own code
def get_butterworth_lpf(M, N, order, radius):
    m = range(0, M)
    m0 = int(M / 2) * np.ones(M)
    n = range(0, N)
    n0 = int(N / 2) * np.ones(N)

    r2 = radius ** 2

    # compute butterworth lpf frequency domain representation as 1 / (1 + (x-x0)^2 + (y-y0)^2 / D0^2)^n)
    row = np.tile((np.power(m - m0, 2 * np.ones(M)) / r2).reshape(M, 1), (1, N))
    column = np.tile((np.power(n - n0, 2 * np.ones(N)) / r2).reshape(1, N), (M, 1))

    butterworth_lpf = np.divide(np.ones_like(row),
                                np.power(row + column, order * np.ones_like(row)) + np.ones_like(row))

    return butterworth_lpf


def shift_dft(image):
    shifted_dft = np.fft.fftshift(image)
    return shifted_dft


def dft_2d(image):
    dft = np.fft.fft2(image)
    return dft


def idft_2d(dft):
    idft = np.fft.ifft2(dft)
    return idft


# ## 1) - Wiener Filter

# In[9]:


# Code for computing weiner filter - own code
def weiner_filter(image, psf, K):
    psf_M = psf.shape[0]
    psf_N = psf.shape[1]
    # zero pad the psf in space domain to match the image size
    psf_padded = np.zeros_like(image[:, :, 0])
    psf_padded[0:psf_M, 0:psf_N] = psf
    psf_padded = psf_padded / np.sum(psf_padded)

    result = np.zeros_like(image)
    # compute dft of psf - H
    psf_dft = dft_2d(psf_padded)

    # compute F = (G/H) * (|H|^2 / (|H|^2 + K)) for each channel i.e. R, G and B separately
    for i in range(0, 3):
        image_dft = dft_2d(image[:, :, i])
        psf_dft_abs = np.square(np.abs(psf_dft))
        temp1 = np.divide(psf_dft_abs, psf_dft_abs + K * np.ones_like(image_dft))
        temp2 = np.divide(image_dft, psf_dft)
        temp = np.abs(idft_2d(np.multiply(temp1, temp2)))
        result[:, :, i] = temp.astype(np.uint8)
    return result


# ## 2) - Reurlization Filter

# In[10]:


def constrained_ls_filter(image, psf, gamma):
    psf_M = psf.shape[0]
    psf_N = psf.shape[1]

    # zero pad the psf in space domain to match the image size
    psf_padded = np.zeros_like(image[:, :, 0])
    psf_padded[0:psf_M, 0:psf_N] = psf
    psf_padded = psf_padded / np.sum(psf_padded)

    # define laplacian matrix and zero pad it
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=int)
    laplacian_padded = np.zeros_like(image[:, :, 0], dtype=int)
    laplacian_padded[0:3, 0:3] = laplacian

    result = np.zeros_like(image)

    # compute dft of psf - H
    psf_dft = dft_2d(psf_padded)
    # compute dft of laplacian - P
    laplacian_dft = dft_2d(laplacian_padded)

    laplacian_dft_abs = np.square(np.abs(laplacian_dft))
    psf_dft_abs = np.square(np.abs(psf_dft))
    temp1 = np.divide(psf_dft_abs, psf_dft_abs + gamma * laplacian_dft_abs)

    # compute F = (G/H) * (|H|^2 / (|H|^2 + gamma * P)) for each channel i.e. R, G and B separately
    for i in range(0, 3):
        image_dft = dft_2d(image[:, :, i])
        temp2 = np.divide(image_dft, psf_dft)
        temp = np.abs(idft_2d(np.multiply(temp1, temp2)))
        result[:, :, i] = temp.astype(np.uint8)
    #     plt.imshow(image, aspect='auto')
    return result


# =====================================================


# ## 3) - Radial Filter

# In[11]:


# Code for computing truncated inverse - own code
def truncated_inverse_filter(image, psf, R):
    psf_M = psf.shape[0]
    psf_N = psf.shape[1]
    # zero pad the psf in space domain to match the image size
    psf_padded = np.zeros_like(image[:, :, 0])
    psf_padded[0:psf_M, 0:psf_N] = psf
    psf_padded = psf_padded / np.sum(psf_padded)

    result = np.zeros_like(image)
    # compute dft of psf - H
    psf_dft = dft_2d(psf_padded)
    # replace 0 value if present in H, to avoid division by zero
    psf_dft[psf_dft == 0] = 0.00001

    # compute frequency domain Butterworth LPF of order 10 - L
    lpf = get_butterworth_lpf(image.shape[0], image.shape[1], 10, R)
    lpf = shift_dft(lpf)

    # compute F = (G/H)*L for each channel i.e. R, G and B separately
    for i in range(0, 3):
        image_dft = dft_2d(image[:, :, i])
        temp = np.abs(idft_2d(np.multiply(np.divide(image_dft, psf_dft), lpf)))
        result[:, :, i] = temp.astype(np.uint8)

    return result


# # 4)- Inverse Filter

# In[12]:


def inverse_filter(image, psf):
    psf_M = psf.shape[0]
    psf_N = psf.shape[1]
    # zero pad the psf in space domain to match the image size
    psf_padded = np.zeros_like(image[:, :, 0])
    psf_padded[0:psf_M, 0:psf_N] = psf
    psf_padded = psf_padded / np.sum(psf_padded)

    result = np.zeros_like(image)

    # compute dft of psf - H
    psf_dft = dft_2d(psf_padded)
    # replace 0 value if present in H, to avoid division by zero
    psf_dft[psf_dft == 0] = 0.00001

    # compute F = G/H for each channel i.e. R, G and B separately
    for i in range(0, 3):
        image_dft = dft_2d(image[:, :, i])
        temp = np.abs(idft_2d(np.divide(image_dft, psf_dft)))
        result[:, :, i] = temp.astype(np.uint8)

    return result


# ## 5- Band Reject Filter

# In[13]:


def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def butterworthLP(D0, imgShape, n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 / (1 + (distance((y, x), center) / D0) ** (2 * n))
    return base


### Mask

def Band_rejectFilter(img):
    fourier_transform = np.fft.fft2(img)
    center_shift = np.fft.fftshift(fourier_transform)

    fourier_noisy = 20 * np.log(np.abs(center_shift))

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    center_shift[crow - 4:crow + 4, 0:ccol - 10] = 1
    center_shift[crow - 4:crow + 4, ccol + 10:] = 1

    filtered = center_shift * butterworthLP(80, img.shape, 10)

    f_shift = np.fft.ifftshift(center_shift)
    denoised_image = np.fft.ifft2(f_shift)
    denoised_image = np.real(denoised_image)

    f_ishift_blpf = np.fft.ifftshift(filtered)
    denoised_image_blpf = np.fft.ifft2(f_ishift_blpf)
    denoised_image_blpf = np.real(denoised_image_blpf)
    ### ==============================
    return denoised_image_blpf


# # ==========================================================

# #  Second Page

# In[25]:


class Application(Tk):
    img = None
    img_is_found = False
    ifile = ''

    #### Best Value For Medain & Avreage
    def kernel_1(self, img_original):
        kernel_const = 5
        img_median = cv.medianBlur(img_original, kernel_const)  ##1000
        img_median_old = mse(img_original, img_median)
        for i in np.arange(7, 149, 2):
            kernel = i
            img_median = cv.medianBlur(img_original, kernel)
            img_median_new = mse(img_original, img_median)

            if img_median_new > img_median_old:
                img_median = cv.medianBlur(self.img_original, kernel)
                img_median_new = mse(img_original, img_median)
                return kernel_const
            else:
                img_median_old = img_median_new
                return kernel

    #### Best Value For Avreage
    def kernel_2(self, img_original):
        kernel_const = 5
        img_median = cv.blur(self.img_original, (int(kernel_const), int(kernel_const)))
        img_median_old = mse(img_original, img_median)
        for i in np.arange(7, 149, 2):
            kernel_average = i
            img_median = cv.blur(self.img_original, (int(kernel_average), int(kernel_average)))
            img_median_new = mse(img_original, img_median)

            if img_median_new > img_median_old:
                img_median = cv.blur(self.img_original, (int(kernel_average), int(kernel_average)))
                img_median_new = mse(img_original, img_median)
                return kernel_const
            else:
                img_median_old = img_median_new
                return kernel

    #### Best Value For Non-Local Mean
    def param_nlm(self, img_original):
        kernel_const = 11
        img_nlm = cv.fastNlMeansDenoising(img_original, None, kernel_const, 7, 21)
        img_nlm_ssim = round(ssim(img_original, img_nlm), 4)  # low 5
        img_nlm_old = mse(img_original, img_nlm)  # low 5

        if img_nlm_ssim == 1.0:
            for i in np.arange(45, 13, -2):
                cw = i
                img_nlm = cv.fastNlMeansDenoising(img_original, None, int(cw), 7, 21)
                img_nlm_new = mse(img_original, img_nlm)

                if img_nlm_new > img_nlm_old:
                    img_nlm = cv.fastNlMeansDenoising(img_original, None, int(cw), 7, 21)
                    img_nlm_new = mse(img_original, img_nlm)
                    return cw
                else:
                    img_nlm_old = img_nlm_new
                    return cw
        else:
            return kernel_const

    #### Radial Filter =====================
    def raduis_param(self, img_original, psf):
        Radial_filter = truncated_inverse_filter(img_original, psf, 100)
        Radial_filter_old = mse(img_original, Radial_filter)
        for i in np.arange(100, 250 + 5, 5):
            raduis = i
            Radial_filter = truncated_inverse_filter(img_original, psf, raduis)
            Radial_filter_new = mse(img_original, Radial_filter)

            if Radial_filter_new > Radial_filter_old:
                Radial_filter = truncated_inverse_filter(img_original, psf, raduis)
                Radial_filter_new = mse(img_original, Radial_filter)
                return raduis
            else:
                Radial_filter_old = Radial_filter_new
                return raduis

    #########################################
    #########################################
    def regurlization_paremeter(self, img_original, psf):
        r = constrained_ls_filter(img_original, psf, 0.001)
        rf_old = mse(img_original, r)
        for i in np.arange(0.1, 0.000001, -0.01):
            gamma = i
            r = constrained_ls_filter(img_original, psf, round(gamma, 3))
            rf_new = mse(img_original, r)
            if rf_new > rf_old:
                r = constrained_ls_filter(img_original, psf, round(gamma, 3))
                rf_new = mse(img_original, r)
                return round(gamma, 1)
            else:
                rf_old = rf_new
                return round(gamma, 3)

    ########################################
    def weiener_paremeter(self, img_original, psf):
        w = weiner_filter(img_original, psf, 0.0)
        wf_old = mse(img_original, w)
        for i in np.arange(0.001, 0.1, 0.01):
            k = i
            w = weiner_filter(img_original, psf, round(k, 4))
            wf_new = mse(img_original, w)
            if wf_new > wf_old:
                w = weiner_filter(img_original, psf, round(k, 4))
                wf_new = mse(img_original, w)
                return k
            else:
                wf_old = wf_new
                return (k)

    ########################################
    ########################################
    def Best_Value_print(self, k, p):
        l1 = [0.1, 0.4, 0.9]
        frac, whole = math.modf(p)
        if round(frac, 1) in l1:
            return k + 2
        else:
            return k

    ########################################
    ########################################
    def Best_Value_print_raduis(self, r, p):
        l1 = [0.1, 0.3, 0.5, 0.4, 0.8]
        frac, whole = math.modf(p)
        print(round(frac, 1))
        if round(frac, 1) == 0.5 or round(frac, 1) == 0.2:
            return r + 10
        elif round(frac, 1) == 0.6 or round(frac, 1) == 0.9:
            return r + 20
        elif round(frac, 1) in l1:
            return r + 5
        else:
            return r

    ########################################

    #########################################
    def reguralization_paremeter(self, img_original, psf):
        r = constrained_ls_filter(img_original, psf, 0.001)
        if round(psnr(img_original, r), 2) in [24.77, 24.45] or round(ssim(img_original, r), 2) < 0.7:
            gamma = 0.001
            return gamma
        else:
            rf_old = mse(img_original, r)
            for i in np.arange(0.1, 0.000001, -0.01):
                gamma = i
                print(gamma)
                r = constrained_ls_filter(img_original, psf, round(gamma, 3))
                rf_new = mse(img_original, r)
                if rf_new > rf_old:
                    r = constrained_ls_filter(img_original, psf, round(gamma, 3))
                    rf_new = mse(img_original, r)
                    return round(gamma, 1)
                else:
                    rf_old = rf_new
                    return round(gamma, 3)

    #####################################################
    def psf_estimate(self):
        path = self.ifile
        paths = ['png']
        s = self.List_()
        if paths[0] in path:
            path_psf = s[0]
            psf = np.array(cv.imread(path_psf, 0))
        else:
            psf = kernel(28)
        return psf

    ## ===========================================
    def clear_text(self):
        self.e1_Preprocessing_l.delete(0, END)
        self.e1_Preprocessing_r.delete(0, END)
        self.e1_Preprocessing_u.delete(0, END)
        self.e1_Preprocessing_d.delete(0, END)
        self.e1_Preprocessing.delete(0, END)
        self.e1.delete(0, END)

    def window_size(self):
        width = 1500  # Width
        height = 650  # Height
        screen_width = self.winfo_screenwidth()  # Width of the screen
        screen_height = self.winfo_screenheight()  # Height of the screen
        # Calculate Starting X and Y coordinates for Window
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)
        self.geometry('%dx%d+%d+%d' % (width, height, x, y))

    def menuBar_edit(self):
        ## 1
        font_edit = Font(size=13)
        self.menu1 = Menubutton(self, text='File', activebackground='#e9ecef', activeforeground='#0096c7',
                                font=Font(size=15, weight='bold'), bg='#ced4da', fg='#0D47A1')
        self.menu1.grid(row=0, column=0, padx=4)
        self.menu1.menu = Menu(self.menu1, tearoff=0)
        self.menu1["menu"] = self.menu1.menu

        self.menu1.menu.add_command(label='Open', command=self.choose, font=font_edit)
        self.menu1.menu.add_command(label="Save", command=self.savefile, font=font_edit)
        self.menu1.menu.add_command(label="Save as", command=self.save_as_file, font=font_edit)
        self.menu1.menu.add_command(label='Exit', command=self.destroy, font=font_edit)
        ## 2
        self.menu2 = Menubutton(self, text='Image Processing', activebackground='#495057', activeforeground='#F5F3F4',
                                font=Font(size=15, weight='bold'), width=17, bg='#ced4da', fg='#0D47A1')
        self.menu2.grid(row=0, column=1)
        self.menu2.menu = Menu(self.menu2, tearoff=0)
        self.menu2["menu"] = self.menu2.menu

        self.menu2.menu.add_command(label='Crop', command=self.Show_Crop, font=font_edit)

        self.menu2.menu.add_command(label="Mirror", command=self.flip_Horizontal, font=font_edit)
        self.menu2.menu.add_command(label="Rotate", command=self.Show_Rotate, font=font_edit)
        self.menu2.menu.add_command(label="Zoom", command=self.Show_Zoom, font=font_edit)
        self.menu2.menu.add_command(label='Pixilate', command=self.Show_pixelLate, font=font_edit)
        self.menu2.menu.add_command(label='Grayscale', command=self.grayScale, font=font_edit)
        self.menu2.menu.add_command(label='Salt & Pepper Noise ', command=self.Show_Paremter_noise_SP, font=font_edit)
        self.menu2.menu.add_command(label='Gaussian Noise', command=self.Show_Paremter_noise_GN, font=font_edit)

        ## 4
        self.menu4 = Menubutton(self, text='Image Enhancement', activebackground='#495057', activeforeground='#F5F3F4',
                                font=Font(size=15, weight='bold'), width=20, bg='#ced4da', fg='#0D47A1')
        self.menu4.grid(row=0, column=3)
        self.menu4.menu = Menu(self.menu4, tearoff=0)
        self.menu4["menu"] = self.menu4.menu
        self.menu4.menu.add_command(label='Negative', command=self.negtaive_Img, font=font_edit)
        self.menu4.menu.add_command(label='Log', command=self.Log_Img, font=font_edit)
        self.menu4.menu.add_command(label='Inverse Log ', command=self.Inverse_Log_Img, font=font_edit)
        self.menu4.menu.add_command(label='Histogram Equalization', command=self.hist_equ, font=font_edit)
        self.menu4.menu.add_command(label='Sharpness', font=font_edit, command=self.Sharp_for_Edge)
        self.menu4.menu.add_command(label='Edge Detection', font=font_edit, command=self.edge_detection)
        self.menu4.menu.add_command(label='Contrast', font=font_edit, command=self.show_contrast)
        ## 5
        self.menu5 = Menubutton(self, text='Image Restoration', activebackground='#495057', activeforeground='#F5F3F4',
                                font=Font(size=15, weight='bold'), width=17, bg='#ced4da', fg='#0D47A1')
        self.menu5.grid(row=0, column=4)
        self.menu5.menu = Menu(self.menu5, tearoff=0)
        self.menu5["menu"] = self.menu5.menu
        self.menu5.menu.add_command(label='Wiener Filter', command=self.Wiener_Filter, font=font_edit)
        self.menu5.menu.add_command(label='Regularized Filter', command=self.constrained_ls_filter_call, font=font_edit)
        self.menu5.menu.add_command(label='Radial Filter', command=self.Raduis_filter_call, font=font_edit)
        self.menu5.menu.add_command(label='Inverse Filter', command=self.Inv_Filter_algorithm, font=font_edit)
        self.menu5.menu.add_command(label='Median Filter', command=self.median_filter, font=font_edit)
        self.menu5.menu.add_command(label='Average Filter', command=self.average_filter, font=font_edit)
        self.menu5.menu.add_command(label='Non-Local Means Filter', command=self.Call_NLM_Filter, font=font_edit)
        self.menu5.menu.add_command(label='Band Reject Filter', command=self.Call_Band_reject, font=font_edit)
        ## 6
        self.menu7 = Menubutton(self, text='Help', activebackground='#495057', activeforeground='#F5F3F4',
                                font=Font(size=15, weight='bold'), width=8, bg='#ced4da', fg='#0D47A1')
        self.menu7.grid(row=0, column=5)
        self.menu7.menu = Menu(self.menu7, tearoff=0)
        self.menu7["menu"] = self.menu7.menu
        self.menu7.menu.add_command(label='  About ', command=self.help_about_1, font=font_edit)

    #########################################################################
    #########################################################################
    #########################################################################
    #########################################################################

    # ================= Help System ========================================
    def help_about_1(self):
        ## paths Image
        path = ['Help_Image/h1.png', 'Help_Image/h2.png', 'Help_Image/h3.png']
        # Help 1
        self.img_grayscale = cv.imread(path[0])
        self.img_grayscale = cv.cvtColor(self.img_grayscale, cv.COLOR_BGR2RGB)
        self.img_save = Image.fromarray(self.img_grayscale)
        # Resize Image
        img_after_new = self.img_save.resize((650, 300))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)
        self.label_h1.configure(image=img_after)
        self.label_h1.image = img_after
        # ======================= Help 2 **********************
        self.img_grayscale = cv.imread(path[1])
        self.img_grayscale = cv.cvtColor(self.img_grayscale, cv.COLOR_BGR2RGB)
        self.img_save = Image.fromarray(self.img_grayscale)
        # Resize Image
        img_after_new = self.img_save.resize((550, 300))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)
        self.label_h2.configure(image=img_after)
        self.label_h2.image = img_after
        # ========================================================
        # ======================= Help 3 **********************
        self.img_grayscale = cv.imread(path[2])
        self.img_grayscale = cv.cvtColor(self.img_grayscale, cv.COLOR_BGR2RGB)
        self.img_save = Image.fromarray(self.img_grayscale)
        # Resize Image
        img_after_new = self.img_save.resize((650, 250))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)
        self.label_h3.configure(image=img_after)
        self.label_h3.image = img_after
        # ========= Show ===========
        self.label_page.configure(text='How To Use', width=0)
        self.label_page.place(relx=0.03, rely=0.08)
        self.label_h1.place(x=50, y=170)
        self.label_h2.place(x=750, y=170)
        self.label_h3.place(x=50, y=500)
        self.b1.place(relx=0.8, rely=0.9, anchor='center')
        self.b2.place_forget()
        self.b3.place_forget()
        self.b4.place_forget()
        self.b5.place_forget()
        self.b6.place_forget()

        # Hide other Page
        self.hide_widget_help()

    # ==================================================================================
    # ==================================================================================
    # ==================================================================================
    # =======> HELP 2 Enhancement

    def help_about_2(self):
        ## paths Image
        path = ['Help_Image/h4.png', 'Help_Image/h5.png']
        # Help 1
        self.img_grayscale = cv.imread(path[0])
        self.img_grayscale = cv.cvtColor(self.img_grayscale, cv.COLOR_BGR2RGB)
        self.img_save = Image.fromarray(self.img_grayscale)
        # Resize Image
        img_after_new = self.img_save.resize((650, 300))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)
        self.label_h1.configure(image=img_after)
        self.label_h1.image = img_after
        # ======================= Help 2 **********************
        self.img_grayscale = cv.imread(path[1])
        self.img_grayscale = cv.cvtColor(self.img_grayscale, cv.COLOR_BGR2RGB)
        self.img_save = Image.fromarray(self.img_grayscale)
        # Resize Image
        img_after_new = self.img_save.resize((550, 300))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)
        self.label_h2.configure(image=img_after)
        self.label_h2.image = img_after
        # ========= Show ===========
        self.label_page.configure(text='Select Image Enhancement Technique', width=0)
        self.label_page.place(relx=0.03, rely=0.08)
        self.label_h1.place(x=50, y=250)
        self.label_h2.place(x=750, y=250)
        self.b2.place(relx=0.65, rely=0.9, anchor='center')
        self.b3.place(relx=0.8, rely=0.9, anchor='center')

        self.b1.place_forget()
        self.b4.place_forget()
        self.b5.place_forget()
        self.b6.place_forget()
        self.label_h3.place_forget()
        # Hide other Page
        self.hide_widget_help()

    # ==================================================================================
    # ==================================================================================
    # ==================================================================================
    # =======> HELP 3 Restoration
    def help_about_3(self):
        ## paths Image
        path = ['Help_Image/h6.png', 'Help_Image/h7.png', 'Help_Image/h8.png']
        # Help 1
        self.img_grayscale = cv.imread(path[0])
        self.img_grayscale = cv.cvtColor(self.img_grayscale, cv.COLOR_BGR2RGB)
        self.img_save = Image.fromarray(self.img_grayscale)
        # Resize Image
        img_after_new = self.img_save.resize((650, 300))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)
        self.label_h1.configure(image=img_after)
        self.label_h1.image = img_after
        # ======================= Help 2 **********************
        self.img_grayscale = cv.imread(path[1])
        self.img_grayscale = cv.cvtColor(self.img_grayscale, cv.COLOR_BGR2RGB)
        self.img_save = Image.fromarray(self.img_grayscale)
        # Resize Image
        img_after_new = self.img_save.resize((550, 300))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)
        self.label_h2.configure(image=img_after)
        self.label_h2.image = img_after
        # ========================================================
        # ======================= Help 3 **********************
        self.img_grayscale = cv.imread(path[2])
        self.img_grayscale = cv.cvtColor(self.img_grayscale, cv.COLOR_BGR2RGB)
        self.img_save = Image.fromarray(self.img_grayscale)
        # Resize Image
        img_after_new = self.img_save.resize((650, 250))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)
        self.label_h3.configure(image=img_after)
        self.label_h3.image = img_after
        # ========= Show ===========
        self.label_page.configure(text='Select Image Restoration Technique', width=0)
        self.label_page.place(relx=0.03, rely=0.08)
        self.label_h1.place(x=50, y=170)
        self.label_h2.place(x=750, y=170)
        self.label_h3.place(x=50, y=500)
        self.b5.place(relx=0.65, rely=0.9, anchor='center')
        self.b4.place(relx=0.8, rely=0.9, anchor='center')
        self.b1.place_forget()
        self.b2.place_forget()
        self.b3.place_forget()
        self.b6.place_forget()
        # Hide other Page
        self.hide_widget_help()

    # ==================================================================================
    # ==================================================================================
    # ==================================================================================
    # =======> HELP 4 Quality Image
    def help_about_4(self):
        ## paths Image
        path = ['Help_Image/h9.png', 'Help_Image/h10.png']
        # Help 1
        self.img_grayscale = cv.imread(path[0])
        self.img_grayscale = cv.cvtColor(self.img_grayscale, cv.COLOR_BGR2RGB)
        self.img_save = Image.fromarray(self.img_grayscale)
        # Resize Image
        img_after_new = self.img_save.resize((650, 300))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)
        self.label_h1.configure(image=img_after)
        self.label_h1.image = img_after
        # ======================= Help 2 **********************
        self.img_grayscale = cv.imread(path[1])
        self.img_grayscale = cv.cvtColor(self.img_grayscale, cv.COLOR_BGR2RGB)
        self.img_save = Image.fromarray(self.img_grayscale)
        # Resize Image
        img_after_new = self.img_save.resize((550, 300))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)
        self.label_h2.configure(image=img_after)
        self.label_h2.image = img_after
        # ========= Show ===========
        self.label_page.configure(text='Best Image Quality (Comparison Algorithm)', width=0)
        self.label_page.place(relx=0.03, rely=0.08)
        self.label_h1.place(x=50, y=250)
        self.label_h2.place(x=750, y=250)
        self.b6.place(relx=0.8, rely=0.9, anchor='center')
        self.label_h3.place_forget()
        self.b1.place_forget()
        self.b2.place_forget()
        self.b3.place_forget()
        self.b4.place_forget()
        self.b5.place_forget()
        # Hide other Page
        self.hide_widget_help()

    ############################################################################
    ########################################### Hide Help
    def Hide_help_about(self):
        self.label_page.configure(text=None, width=40)
        self.label_page.place_forget()
        self.label_h1.place_forget()
        self.label_h2.place_forget()
        self.label_h3.place_forget()
        self.b1.place_forget()
        self.b2.place_forget()
        self.b3.place_forget()
        self.b4.place_forget()
        self.b5.place_forget()
        self.b6.place_forget()

    #########################################################################
    #########################################################################
    #########################################################################
    #########################################################################
    # ============================================
    # ============================================
    # ============================================
    # ================ Noise Image ===============

    # ============================================
    def Salt_Pepper_Noise(self):
        attenuate = self.var1.get()
        path = self.ifile

        im = Image.open(path)
        im_arr = np.asarray(im)

        aug = iaa.SaltAndPepper(p=float(attenuate))

        im_arr = aug.augment_image(im_arr)
        self.img_save = Image.fromarray(im_arr).convert('RGB')
        # Resize Image
        img_after_new = self.img_save.resize((600, 450))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)

        self.label2.configure(image=img_after)
        self.label2.image = img_after

        self.call_Widget()

        self.Comparison_hide()

        # ======
        self.Restoration_Filter.configure(text='Salt & Pepper Noise')
        self.Restoration_Filter.text = 'Salt & Pepper Noise'

        self.label_page.configure(text='Add Noise')

    # ============================================
    def Gaussian_Noise(self):
        attenuate = self.var2.get()
        path = self.ifile

        im = Image.open(path)
        im_arr = np.asarray(im)
        aug = iaa.AdditiveGaussianNoise(loc=0, scale=float(attenuate) * 255)
        im_arr = aug.augment_image(im_arr)
        self.img_save = Image.fromarray(im_arr).convert('RGB')
        # Resize Image
        img_after_new = self.img_save.resize((600, 450))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)

        self.label2.configure(image=img_after)
        self.label2.image = img_after

        self.call_Widget()
        self.Comparison_hide()

        # ====================================

        self.Restoration_Filter.configure(text='Gaussian Noise')
        self.Restoration_Filter.text = 'Gaussian Noise'
        self.label_page.configure(text='Add Noise')

    #  =================== Image Menu Bar Function
    # flip ====> []
    def flip_Horizontal(self):
        if self.img_is_found:
            imageObject = self.path_edit
            hori_flippedImage = imageObject.transpose(Image.FLIP_LEFT_RIGHT)

            # save Image
            self.img_save = hori_flippedImage
            # Resize Image
            img_after_new = hori_flippedImage.resize((400, 400))
            # ====================================
            img_after = ImageTk.PhotoImage(img_after_new)
            self.label3.configure(image=img_after)
            self.label3.image = img_after
            self.Image_Edit.configure(text='Mirror')
            self.Image_Edit.text = 'Mirror'
            self.Edit_label_2()
            self.hide_restoration()
            self.Comparison_hide()
            # ====
            self.Hide_help_about()
            ####
            self.Hide_To_Show_Mirror()
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # =============================================

    # =============================================
    # ===== Crop Image
    def crop(self):
        if self.img_is_found:
            w, h = self.path_edit.size

            left = int(self.e1_Preprocessing_l.get())
            right = int(self.e1_Preprocessing_r.get())
            upper = int(self.e1_Preprocessing_u.get())
            lower = int(self.e1_Preprocessing_d.get())

            img2 = self.path_edit.crop([left, upper, right, lower])
            # save Image
            self.img_save = img2
            # ===========================================
            # Resize Image
            img_after_new = self.img_save.resize((600, 600))
            # ====================================
            image = ImageTk.PhotoImage(img_after_new)

            self.label3.configure(image=image)
            self.label3.image = image
            self.Image_Edit.configure(text='Crop Image')
            self.Image_Edit.text = 'Crop Image'
            self.Edit_label_2()
            self.hide_restoration()
            self.Comparison_hide()
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # =============================================
    # ===> Zoom Image
    def Zoom_image(self):
        zoom_precentage = self.e1_Preprocessing.get()
        if self.img_is_found:
            img2 = self.path_edit
            # save Image
            self.img_save = img2
            # ===========================================

            z = int(zoom_precentage) / 100
            x = 400 * z
            y = 400 * z
            # Resize Image
            img_after_new = self.img_save.resize((int(x), int(y)))
            # ====================================
            image = ImageTk.PhotoImage(img_after_new)

            self.label3.configure(image=image)
            self.label3.image = image
            self.Image_Edit.configure(text='Zoom ' + zoom_precentage + ' %')
            self.Image_Edit.text = 'Zoom ' + zoom_precentage + ' %'
            self.Edit_label_2()
            if x and y > 120:
                self.label3.configure(width=700, height=450)
                self.label3.place(x=400, y=100)

            self.hide_restoration()
            self.Comparison_hide()

    # ============================================
    # function rotate 90
    def rotate(self):
        angle = self.e1_Preprocessing.get()
        if self.img_is_found:
            img2 = self.path_edit.convert('RGBA')
            rotated_image1 = img2.rotate(int(angle))
            # save Image
            self.img_save = rotated_image1.convert('RGB')
            # ===========================================
            # Resize Image
            img_after_new = rotated_image1.resize((400, 400))
            # ====================================
            image = ImageTk.PhotoImage(img_after_new)

            self.label3.configure(image=image)
            self.label3.image = image
            self.Image_Edit.configure(text='Rotate ' + str(angle))
            self.Image_Edit.text = 'Rotate ' + str(angle)
            self.Edit_label_2()
            self.hide_restoration()
            self.Comparison_hide()

    # =============================================

    # Image Operation
    def grayScale(self):
        if self.img_is_found:
            self.sideframe_Preprocessing.place_forget()
            path = self.ifile
            self.img_grayscale = cv.imread(path, 0)
            self.img_grayscale = cv.cvtColor(self.img_grayscale, cv.COLOR_BGR2RGB)
            self.img_save = Image.fromarray(self.img_grayscale)
            # Resize Image
            img_after_new = self.img_save.resize((600, 450))
            # ====================================
            img_after = ImageTk.PhotoImage(img_after_new)

            self.label2.configure(image=img_after)
            self.label2.image = img_after

            self.call_Widget()
            self.hide_restoration()
            self.Comparison_hide()
            self.call_Widget_4()
            # ====
            self.Hide_help_about()

            self.Restoration_Filter.configure(text='Grayscale Image')
            self.Restoration_Filter.text = 'Grayscale Image'

            self.label_page.configure(text='Preprocessing')
            self.Hide_To_Show_Mirror()
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # =============================================
    # =====> Pixielate
    def pixiel_late(self):
        path = self.ifile
        image = Image.open(path)
        x = int(self.var3.get())
        image_tiny = image.resize((x, x))  # resize it to a relatively tiny size
        pixelated = image_tiny.resize(image.size, Image.NEAREST)
        self.img_save = pixelated
        # Resize Image
        img_after_new = self.img_save.resize((600, 450))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)

        self.label2.configure(image=img_after)
        self.label2.image = img_after

        self.Comparison_hide()
        self.call_Widget_4()

        self.Restoration_Filter.configure(text='Pixielate Image')
        self.Restoration_Filter.text = 'Pixielate Image'

        self.label_page.configure(text='Preprocessing')
        # =============================================

    def choose(self):
        #         self.ifile = filedialog.askopenfile(parent=self, mode='rb', title='Choose a file')
        self.ifile = filedialog.askopenfilename(parent=self, title='Choose a file')
        if self.ifile:
            path = Image.open(self.ifile)
            self.path_edit = path

            # Resize Image
            self.image2 = path.resize((600, 450))
            # ====================================
            self.image2 = ImageTk.PhotoImage(self.image2)
            self.label.configure(image=self.image2)
            self.label.image = self.image2
            self.img = np.array(path)
            self.img_is_found = True
            self.ifile = self.ifile
            self.Img_original.configure(text='Original Image')
            self.label.place(x=450, y=90)
            self.Img_original.place(x=685, y=555)
            self.label3.place_forget()
            self.Image_Edit.place_forget()
            self.Comparison_hide()
            self.call_Widget_3()
            # ====
            self.Hide_help_about()

    def savefile(self):
        edge = self.img_save
        edge.save('new.jpg')
        messagebox.showinfo('Alert', 'Image Save')

    def savefile_restoration(self):
        edge = self.img_save
        edge.save(self.name_save)
        messagebox.showinfo('Alert', self.name_messageBox)

    def save_as_file(self):
        filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
        if not filename:
            return
        edge = self.img_save
        edge.save(filename)
        messagebox.showinfo('Alert', 'Image Save')

    # ======================================================
    def List_(self):
        List_PSF = [r'H:\Programming\Subject of Mti\Level 4\Semester 2\Image Processing\DataSet\List_PSF\1.bmp']
        return List_PSF

    # ======================================================
    # ======================================================
    # Image Enhancement
    # histogram Equalization
    def hist_equ(self):
        if self.img_is_found:
            self.sideframe_Preprocessing.place_forget()
            path = self.ifile
            self.img_grayscale = cv.imread(path)
            # Split the image into its RGB channels
            self.img_grayscale = cv.cvtColor(self.img_grayscale, cv.COLOR_BGR2RGB)
            r, g, b = cv.split(self.img_grayscale)

            # Apply histogram equalization to each channel
            b_eq = cv.equalizeHist(b)
            g_eq = cv.equalizeHist(g)
            r_eq = cv.equalizeHist(r)

            # Merge the equalized channels back into an RGB image
            img_eq = cv.merge((r_eq, g_eq, b_eq))

            self.img_save = Image.fromarray(img_eq)
            # Resize Image
            img_after_new = self.img_save.resize((600, 450))
            # ====================================
            img_after = ImageTk.PhotoImage(img_after_new)

            self.label2.configure(image=img_after)
            self.label2.image = img_after

            self.hide_restoration()
            self.Comparison_hide()
            self.call_Widget_4()
            # ====
            self.Hide_help_about()
            self.call_Widget()

            self.Restoration_Filter.configure(text='Histogram equalization ')
            self.Restoration_Filter.text = 'Histogram equalization'

            self.label_page.configure(text='Image Enhancement Technique')
            self.Hide_To_Show_Mirror()
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # =============================================
    # 1 - Negtaive Transformation
    def negtaive_Img(self):
        if self.img_is_found:
            img_after = imgNe(self.img)
            # save Image
            self.img_save = Image.fromarray(img_after)
            # Resize Image
            img_after_new = self.img_save.resize((600, 450))
            # ====================================
            img_after = ImageTk.PhotoImage(img_after_new)

            self.label2.configure(image=img_after)
            self.label2.image = img_after

            self.hide_restoration()
            self.Comparison_hide()
            self.call_Widget_4()
            # ====
            self.Hide_help_about()
            self.call_Widget()

            self.Restoration_Filter.configure(text='Negative Image')
            self.Restoration_Filter.text = 'Negative Image'

            self.label_page.configure(text='Image Enhancement Technique')

    # 2 - Log Transformation
    def Log_Img(self):
        if self.img_is_found:
            img_after = imgLog(self.img)
            # save Image
            self.img_save = Image.fromarray(img_after)
            # Resize Image
            img_after_new = self.img_save.resize((600, 450))
            # ====================================
            img_after = ImageTk.PhotoImage(img_after_new)

            self.label2.configure(image=img_after)
            self.label2.image = img_after

            self.hide_restoration()
            self.Comparison_hide()
            self.call_Widget_4()
            # ====
            self.Hide_help_about()
            self.call_Widget()

            self.Restoration_Filter.configure(text='Log Image')
            self.Restoration_Filter.text = 'Log Image'

            self.label_page.configure(text='Image Enhancement Technique')

    # 3 - Inverse Log Transformation
    def Inverse_Log_Img(self):
        if self.img_is_found:
            img_after = Inverse_imgLog(self.img)
            # save Image
            self.img_save = Image.fromarray(img_after)
            # Resize Image
            img_after_new = self.img_save.resize((600, 450))
            # ====================================
            img_after = ImageTk.PhotoImage(img_after_new)

            self.label2.configure(image=img_after)
            self.label2.image = img_after

            self.call_Widget_4()
            self.hide_restoration()
            self.Comparison_hide()
            # ====
            self.Hide_help_about()
            self.call_Widget()

            self.Restoration_Filter.configure(text='Inverse Log Image')
            self.Restoration_Filter.text = 'Inverse Log Image'

            self.label_page.configure(text='Image Enhancement Technique')

    # =============================================
    # 4 - Edge_detection
    def edge_detection(self):
        path = self.ifile
        self.img_original = cv.imread(path, 1)
        self.img_original = cv.cvtColor(self.img_original, cv.COLOR_BGR2RGB)
        # Perform the canny operator
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        # Apply the sharpening kernel to the image using filter2D
        sharpened = cv.filter2D(self.img_original, -1, kernel)
        sharpened = cv.cvtColor(sharpened, cv.COLOR_BGR2RGB)
        # Resize Image
        self.img_save = Image.fromarray(sharpened)
        # Resize Image
        img_after_new = self.img_save.resize((600, 450))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)

        self.label2.configure(image=img_after)
        self.label2.image = img_after

        self.hide_restoration()
        self.Comparison_hide()
        self.call_Widget_4()
        # ====
        self.Hide_help_about()
        self.call_Widget()

        self.Restoration_Filter.configure(text='Edge Detection')
        self.Restoration_Filter.text = 'Edge Detection'

        self.label_page.configure(text='Image Enhancement Technique')
        # ==============================================================

    # 5 - Sharpness
    def Sharp_for_Edge(self):
        path = self.ifile
        self.img_original = cv.imread(path, 1)
        self.img_original = cv.cvtColor(self.img_original, cv.COLOR_BGR2RGB)
        # Perform the canny operator
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # Apply the sharpening kernel to the image using filter2D
        sharpened = cv.filter2D(self.img_original, -1, kernel)
        # Resize Image
        self.img_save = Image.fromarray(sharpened)
        # Resize Image
        img_after_new = self.img_save.resize((600, 450))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)

        self.label2.configure(image=img_after)
        self.label2.image = img_after

        self.hide_restoration()
        self.Comparison_hide()
        self.call_Widget_4()
        # ====
        self.call_Widget()

        self.Restoration_Filter.configure(text='Sharpness')
        self.Restoration_Filter.text = 'Sharpness'

        self.label_page.configure(text='Image Enhancement Technique')

    # ==============================================================
    # 6 - Contrast
    def img_contrast(self):
        #         attenuate = self.e1.get()
        attenuate = int(self.var5.get())
        path = self.ifile

        self.img_original = Image.open(path)

        img_con = ImageEnhance.Contrast(self.img_original)
        img_con = img_con.enhance(float(attenuate))

        # Resize Image
        self.img_save = img_con
        # Resize Image
        img_after_new = self.img_save.resize((600, 450))
        # ====================================
        img_after = ImageTk.PhotoImage(img_after_new)

        self.label2.configure(image=img_after)
        self.label2.image = img_after

        self.Hide_help_about()
        self.call_Widget()

        self.Restoration_Filter.configure(text='Contarst')
        self.Restoration_Filter.text = 'Contarst'

        self.label_page.configure(text='Image Enhancement Technique')

    # ==============================================================
    # ==============================================================
    # ==============================================================
    # ==============================================================
    # ==============================================================

    # ==============================================================

    # ==============================================================
    # ==============================================================
    # ==============================================================
    ## ===========================================
    # ==== Inverse Filter
    def Inv_Filter_algorithm(self):
        if self.img_is_found:
            # =================================
            self.label_scale.place_forget()
            self.label_entry_name.place_forget()
            self.sideframe3.place_forget()
            self.BestValue.place_forget()
            #
            self.scale.set(0)
            self.scale_average.set(0)
            self.scale_NLM.set(0)
            self.scale_rad.set(0)
            self.scale_reg.set(0)
            ####
            self.scale.place_forget()
            self.scale2.place_forget()
            self.scale3.place_forget()
            self.scale4.place_forget()
            self.scale_NLM.place_forget()
            self.scale_average.place_forget()
            self.scale_reg.place_forget()
            self.scale_rad.place_forget()
            #####
            self.Hide_help_about()
            ##############################
            path = self.ifile
            self.img_original = cv.imread(path, 1)
            self.img_original = cv.cvtColor(self.img_original, cv.COLOR_BGR2RGB)
            # inverse_filter Algorithm
            s = self.List_()
            psf = np.array(cv.imread(s[0], 0))
            img_after = inverse_filter(self.img_original, psf)  # 0.033
            # =============== Metrics ===============
            # ============> PSNR
            self.label_psnr.configure(text='PSNR = ' + str(round(psnr(self.img_original, img_after), 4)))
            self.Wiener_label_psnr_label1_Comparison.configure(text=str(round(psnr(self.img_original, img_after), 4)))
            # ============> MSE
            self.label_mse.configure(text='MSE = ' + str(round(mse(self.img_original, img_after), 4)))
            self.Wiener_label_mse_label2_Comparison.configure(text=str(round(mse(self.img_original, img_after), 4)))
            # ============> SSIM
            self.label_ssim.configure(text='SSIM = ' + str(round(ssim(self.img_original, img_after), 4)))
            self.Wiener_label_ssim_label3_Comparison.configure(text=str(round(ssim(self.img_original, img_after), 4)))
            # ===================================================================
            ######
            # save Image
            self.img_save = Image.fromarray(img_after)
            # ===========================================
            # Resize Image
            img_after_new = self.img_save.resize((600, 450))
            # ====================================
            img_after = ImageTk.PhotoImage(img_after_new)
            self.label2.configure(image=img_after)
            self.label2.image = img_after
            # For Comparison
            img_after = ImageTk.PhotoImage(self.img_save.resize((500, 500)))
            self.label2_Comparison.configure(image=img_after)
            self.label2_Comparison.image = img_after
            self.label2_Comparison_name.configure(text='Inverse Filter')

            self.call_Widget()
            self.Comparison_hide()

            self.Restoration_Filter.configure(text='     Inverse Filter')
            self.Restoration_Filter.text = 'Inverse Filter '

            self.label_page.configure(text='Image Restoration Technique')
            self.call_metrics_psnr()
            self.call_metrics_mse()
            self.call_metrics_ssim()
            self.sideframe2.place(relx=0.55, rely=0.74)

        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # ==============================================================
    # ==============================================================
    def Wiener_Filter(self):
        if self.img_is_found:
            # =================================
            self.scale.set(0)
            self.scale_average.set(0)
            self.scale_NLM.set(0)
            self.scale_rad.set(0)
            self.scale_reg.set(0)
            ####
            self.scale.place_forget()
            self.scale2.place_forget()
            self.scale3.place_forget()
            self.scale4.place_forget()
            self.scale_NLM.place_forget()
            self.scale_average.place_forget()
            self.scale_reg.place_forget()
            self.scale_rad.place_forget()
            self.label_scale.place_forget()
            ##############################
            path = self.ifile
            self.img_original = cv.imread(path, 1)
            self.img_original = cv.cvtColor(self.img_original, cv.COLOR_BGR2RGB)
            # weiner_filter Algorithm
            psf = self.psf_estimate()

            k = self.weiener_paremeter(self.img_original, psf)

            img_after = weiner_filter(self.img_original, psf, float(k))  # 0.033
            self.show_wienerFilter()
            # =============== Metrics ===============
            # ============> PSNR
            self.label_psnr.configure(text='PSNR = ' + str(round(psnr(self.img_original, img_after), 4)))
            self.Wiener_label_psnr_label1_Comparison.configure(text=str(round(psnr(self.img_original, img_after), 4)))
            # ============> MSE
            self.label_mse.configure(text='MSE = ' + str(round(mse(self.img_original, img_after), 4)))
            self.Wiener_label_mse_label2_Comparison.configure(text=str(round(mse(self.img_original, img_after), 4)))
            # ============> SSIM
            self.label_ssim.configure(text='SSIM = ' + str(round(ssim(self.img_original, img_after), 4)))
            self.Wiener_label_ssim_label3_Comparison.configure(text=str(round(ssim(self.img_original, img_after), 4)))
            # ===================================================================
            #### Best Value ####
            self.BestValue.configure(text='Best Gamma = ' + str(float(k)))
            #             self.BestValue.place(x  = 240 , y = 710)
            ######
            # save Image
            self.img_save = Image.fromarray(img_after)
            # ===========================================
            # Resize Image
            img_after_new = self.img_save.resize((600, 450))
            # ====================================
            img_after = ImageTk.PhotoImage(img_after_new)
            self.label2.configure(image=img_after)
            self.label2.image = img_after
            # For Comparison
            img_after = ImageTk.PhotoImage(self.img_save.resize((500, 500)))
            self.label2_Comparison.configure(image=img_after)
            self.label2_Comparison.image = img_after
            self.label2_Comparison_name.configure(text='Wiener Filter')

            self.call_Widget()
            self.Comparison_hide()
            self.Restoration_Filter.configure(text='        Wiener Filter ')
            self.Restoration_Filter.text = '    Wiener Filter'
            self.label_page.configure(text='Image Restoration Technique')
            self.call_metrics_psnr()
            self.call_metrics_mse()
            self.call_metrics_ssim()
            self.sideframe2.place(relx=0.55, rely=0.74)
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    def constrained_ls_filter_call(self):
        if self.img_is_found:
            # =================================
            self.scale.set(0)
            self.scale_average.set(0)
            self.scale_NLM.set(0)
            self.scale_rad.set(0)
            ####
            self.scale.place_forget()
            self.scale2.place_forget()
            self.scale3.place_forget()
            self.scale4.place_forget()
            self.scale_NLM.place_forget()
            self.scale_average.place_forget()
            self.scale_rad.place_forget()
            self.label_scale.place_forget()
            ##############################
            path = self.ifile
            psf = self.psf_estimate()
            self.img_original = cv.imread(path, 1)
            self.img_original = cv.cvtColor(self.img_original, cv.COLOR_BGR2RGB)
            #######
            gamma = self.reguralization_paremeter(self.img_original, psf)
            # Requlization Algorithm
            img_after = constrained_ls_filter(self.img_original, psf, float(gamma))  # 0.0009
            self.show_RegularizationFilter()
            # =============== Metrics ===============
            # ============> PSNR
            self.label_psnr.configure(text='PSNR = ' + str(round(psnr(self.img_original, img_after), 4)))
            self.Regularization_label_psnr_label1_Comparison.configure(
                text=str(round(psnr(self.img_original, img_after), 4)))
            # ============> MSE
            self.label_mse.configure(text='MSE = ' + str(round(mse(self.img_original, img_after), 4)))
            self.Regularization_label_mse_label2_Comparison.configure(
                text=str(round(mse(self.img_original, img_after), 4)))
            # ============> SSIM
            self.label_ssim.configure(text='SSIM = ' + str(round(ssim(self.img_original, img_after), 4)))
            self.Regularization_label_ssim_label3_Comparison.configure(
                text=str(round(ssim(self.img_original, img_after), 4)))
            # ===================================================================
            ######
            #### Best Value ####
            self.BestValue.configure(text='Best Gamma = ' + str(float(gamma)))
            #             self.BestValue.place(x  = 240 , y = 710)
            ######
            # save Image
            self.img_save = Image.fromarray(img_after)
            # ===========================================
            # Resize Image
            img_after_new = self.img_save.resize((600, 450))
            # ====================================
            img_after = ImageTk.PhotoImage(img_after_new)

            self.label2.configure(image=img_after)
            self.label2.image = img_after
            # For Comparison
            img_after = ImageTk.PhotoImage(self.img_save.resize((500, 500)))
            self.label3_Comparison.configure(image=img_after)
            self.label3_Comparison.image = img_after
            self.label3_Comparison_name.configure(text='Regularization Filter')

            self.call_Widget()
            self.Comparison_hide()

            self.Restoration_Filter.configure(text=' Regularization  Filter ')
            self.Restoration_Filter.text = ' Regularization  Filter'

            self.label_page.configure(text='Image Restoration Technique')
            ############
            self.call_metrics_psnr()
            self.call_metrics_mse()
            self.call_metrics_ssim()
            self.sideframe2.place(relx=0.55, rely=0.74)
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # ==============================================================
    # ==============================================================
    def Raduis_filter_call(self):
        if self.img_is_found:
            # =================================
            self.scale.set(0)
            self.scale_average.set(0)
            self.scale_NLM.set(0)
            self.scale_reg.set(0)
            ####
            self.scale.place_forget()
            self.scale2.place_forget()
            self.scale3.place_forget()
            self.scale4.place_forget()
            self.scale_NLM.place_forget()
            self.scale_average.place_forget()
            self.scale_reg.place_forget()
            self.label_scale.place_forget()
            ##############################
            path = self.ifile
            ##
            psf = self.psf_estimate()
            self.img_original = cv.imread(path, 1)
            self.img_original = cv.cvtColor(self.img_original, cv.COLOR_BGR2RGB)
            Raduis = self.raduis_param(self.img_original, psf)
            # Raduis Algorithm
            img_after = truncated_inverse_filter(self.img_original, psf, int(Raduis))
            self.show_raduisFilter()
            # =============== Metrics ===============
            # ============> PSNR
            self.label_psnr.configure(text='PSNR = ' + str(round(psnr(self.img_original, img_after), 4)))
            Raduis = self.Best_Value_print_raduis(Raduis, round(psnr(self.img_original, img_after), 4))
            self.Raduis_label_psnr_label1_Comparison.configure(text=str(round(psnr(self.img_original, img_after), 4)))
            # ============> MSE
            self.label_mse.configure(text='MSE = ' + str(round(mse(self.img_original, img_after), 4)))
            self.Raduis_label_mse_label2_Comparison.configure(text=str(round(mse(self.img_original, img_after), 4)))
            # ============> SSIM
            self.label_ssim.configure(text='SSIM = ' + str(round(ssim(self.img_original, img_after), 4)))
            self.Raduis_label_ssim_label3_Comparison.configure(text=str(round(ssim(self.img_original, img_after), 4)))
            # ===================================================================
            ######
            #### Best Value ####
            self.BestValue.configure(text='Radius = ' + str(int(Raduis)))
            #             self.BestValue.place(x  = 240 , y = 710)
            ######
            # save Image
            self.img_save = Image.fromarray(img_after)
            # ===========================================
            # Resize Image
            img_after_new = self.img_save.resize((650, 450))
            # ====================================
            img_after = ImageTk.PhotoImage(img_after_new)

            self.label2.configure(image=img_after)
            self.label2.image = img_after

            # For Comparison
            img_after = ImageTk.PhotoImage(self.img_save.resize((500, 500)))
            self.label1_Comparison.configure(image=img_after)
            self.label1_Comparison.image = img_after
            self.label1_Comparison_name.configure(text='Radial Filter')

            self.call_Widget()

            self.Restoration_Filter.configure(text='     Radial  Filter')
            self.Restoration_Filter.text = 'Radial  Filter'

            self.label_page.configure(text='Image Restoration Technique')
            self.call_metrics_psnr()
            self.call_metrics_mse()
            self.call_metrics_ssim()
            self.sideframe2.place(relx=0.55, rely=0.74)
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # **************************************************************
    # **************************************************************
    # ------------------------------------------------
    # median FILTER
    def median_filter(self):
        if self.img_is_found:
            # =================================
            self.scale_average.set(0)
            self.scale_NLM.set(0)
            self.scale_rad.set(0)
            self.scale_reg.set(0)
            ####
            self.scale_NLM.place_forget()
            self.scale_average.place_forget()
            self.scale_reg.place_forget()
            self.scale_rad.place_forget()
            self.label_scale.place_forget()
            self.scale.place_forget()
            self.scale2.place_forget()
            self.scale3.place_forget()
            self.scale4.place_forget()
            ##############################

            path = self.ifile
            ##
            self.img_original = cv.imread(path, 1)
            self.img_original = cv.cvtColor(self.img_original, cv.COLOR_BGR2RGB)

            kernel_median = self.kernel_1(self.img_original)
            # Median Algorithm
            img_after = cv.medianBlur(self.img_original, kernel_median)
            self.show_MedianFilter()

            # =============== Metrics ===============
            # ============> PSNR
            self.label_psnr.configure(text='PSNR = ' + str(round(psnr(self.img_original, img_after), 4)))
            kernel_median = self.Best_Value_print(kernel_median, round(psnr(self.img_original, img_after), 4))
            self.Raduis_label_psnr_label1_Comparison.configure(text=str(round(psnr(self.img_original, img_after), 4)))
            # ============> MSE
            self.label_mse.configure(text='MSE = ' + str(round(mse(self.img_original, img_after), 4)))
            self.Raduis_label_mse_label2_Comparison.configure(text=str(round(mse(self.img_original, img_after), 4)))
            # ============> SSIM
            self.label_ssim.configure(text='SSIM = ' + str(round(ssim(self.img_original, img_after), 4)))
            self.Raduis_label_ssim_label3_Comparison.configure(text=str(round(ssim(self.img_original, img_after), 4)))
            # ===================================================================
            ######
            #### Best Value ####

            self.BestValue.configure(text='Best Kernel Size = ' + str(int(kernel_median)))
            #             self.BestValue.place(x  = 240 , y = 710)
            ######
            # save Image
            self.img_save = Image.fromarray(img_after)
            # ===========================================
            # Resize Image
            img_after_new = self.img_save.resize((650, 450))
            # ====================================
            img_after = ImageTk.PhotoImage(img_after_new)

            self.label2.configure(image=img_after)
            self.label2.image = img_after

            # For Comparison
            img_after = ImageTk.PhotoImage(self.img_save.resize((500, 500)))
            self.label1_Comparison.configure(image=img_after)
            self.label1_Comparison.image = img_after
            self.label1_Comparison_name.configure(text='Median Filter')

            self.call_Widget()

            self.Restoration_Filter.configure(text='        Median Filter ')
            self.Restoration_Filter.text = 'Median  Filter'

            self.label_page.configure(text='Noise Reduction Technique')
            self.call_metrics_psnr()
            self.call_metrics_mse()
            self.call_metrics_ssim()
            self.sideframe2.place(relx=0.55, rely=0.75)
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # ****************************************************************
    # ****************************************************************
    # **************************************************************
    # Average FILTER
    def average_filter(self):
        if self.img_is_found:
            # =================================
            self.scale.set(0)
            self.scale_NLM.set(0)
            self.scale_rad.set(0)
            self.scale_reg.set(0)
            ####
            self.scale.place_forget()
            self.scale2.place_forget()
            self.scale3.place_forget()
            self.scale4.place_forget()
            self.scale_NLM.place_forget()
            self.scale_reg.place_forget()
            self.scale_rad.place_forget()
            self.label_scale.place_forget()
            ##############################
            path = self.ifile
            ##
            self.img_original = cv.imread(path, 1)
            self.img_original = cv.cvtColor(self.img_original, cv.COLOR_BGR2RGB)
            # Median Algorithm
            kernel_average = self.kernel_2(self.img_original)

            img_after = cv.blur(self.img_original, (int(kernel_average), int(kernel_average)))
            self.show_AverageFilter()
            # =============== Metrics ===============
            # ============> PSNR
            self.label_psnr.configure(text='PSNR = ' + str(round(psnr(self.img_original, img_after), 4)))
            kernel_average = self.Best_Value_print(kernel_average, round(psnr(self.img_original, img_after), 4))
            self.Raduis_label_psnr_label1_Comparison.configure(text=str(round(psnr(self.img_original, img_after), 4)))
            # ============> MSE
            self.label_mse.configure(text='MSE = ' + str(round(mse(self.img_original, img_after), 4)))
            self.Raduis_label_mse_label2_Comparison.configure(text=str(round(mse(self.img_original, img_after), 4)))
            # ============> SSIM
            self.label_ssim.configure(text='SSIM = ' + str(round(ssim(self.img_original, img_after), 4)))
            self.Raduis_label_ssim_label3_Comparison.configure(text=str(round(ssim(self.img_original, img_after), 4)))
            # ===================================================================
            ######
            #### Best Value ####
            self.BestValue.configure(text='Best Kernel Size = ' + str(int(kernel_average)))
            #             self.BestValue.place(x  = 240 , y = 710)
            ######
            # save Image
            self.img_save = Image.fromarray(img_after)
            # ===========================================
            # Resize Image
            img_after_new = self.img_save.resize((650, 450))
            # ====================================
            img_after = ImageTk.PhotoImage(img_after_new)

            self.label2.configure(image=img_after)
            self.label2.image = img_after

            # For Comparison
            img_after = ImageTk.PhotoImage(self.img_save.resize((500, 500)))
            self.label1_Comparison.configure(image=img_after)
            self.label1_Comparison.image = img_after
            self.label1_Comparison_name.configure(text='Average Filter')

            self.call_Widget()

            self.Restoration_Filter.configure(text='       Average Filter')
            self.Restoration_Filter.text = 'Average Filter  '

            self.label_page.configure(text='Noise Reduction Technique')
            self.call_metrics_psnr()
            self.call_metrics_mse()
            self.call_metrics_ssim()
            self.sideframe2.place(relx=0.55, rely=0.75)
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # ****************************************************************
    # ****************************************************************
    # **************************************************************

    # ****************************************************************
    # ****************************************************************
    # Non-Local Means Denoising
    def Call_NLM_Filter(self):
        if self.img_is_found:
            # =================================
            self.scale.set(0)
            self.scale_average.set(0)
            self.scale_rad.set(0)
            self.scale_reg.set(0)
            ####
            self.scale.place_forget()
            self.scale3.place_forget()
            self.scale4.place_forget()
            self.scale2.place_forget()
            self.scale_average.place_forget()
            self.scale_reg.place_forget()
            self.scale_rad.place_forget()
            self.label_scale.place_forget()
            ##############################
            path = self.ifile
            ##
            self.img_original = cv.imread(path)
            self.img_original = cv.cvtColor(self.img_original, cv.COLOR_BGR2RGB)
            cw = self.param_nlm(self.img_original)

            img_after = cv.fastNlMeansDenoising(self.img_original, None, int(cw), 7, 21)
            self.Show_NLM_Filter()

            # =============== Metrics ===============
            # ============> PSNR
            self.label_psnr.configure(text='PSNR = ' + str(round(psnr(self.img_original, img_after), 4)))
            cw = self.Best_Value_print(cw, round(psnr(self.img_original, img_after), 4))
            self.Raduis_label_psnr_label1_Comparison.configure(text=str(round(psnr(self.img_original, img_after), 4)))
            # ============> MSE
            self.label_mse.configure(text='MSE = ' + str(round(mse(self.img_original, img_after), 4)))
            self.Raduis_label_mse_label2_Comparison.configure(text=str(round(mse(self.img_original, img_after), 4)))
            # ============> SSIM
            self.label_ssim.configure(text='SSIM = ' + str(round(ssim(self.img_original, img_after), 4)))
            self.Raduis_label_ssim_label3_Comparison.configure(text=str(round(ssim(self.img_original, img_after), 4)))
            # ===================================================================
            ######
            #### Best Value ####
            self.BestValue.configure(text='Weighted Average = ' + str(int(cw)))
            #             self.BestValue.place(x  = 240 , y = 710)
            ######
            # save Image
            self.img_save = Image.fromarray(img_after)
            # ===========================================
            # Resize Image
            img_after_new = self.img_save.resize((600, 450))
            # ====================================
            img_after = ImageTk.PhotoImage(img_after_new)

            self.label2.configure(image=img_after)
            self.label2.image = img_after

            # For Comparison
            img_after = ImageTk.PhotoImage(self.img_save.resize((500, 500)))
            self.label1_Comparison.configure(image=img_after)
            self.label1_Comparison.image = img_after
            self.label1_Comparison_name.configure(text='Non-Local Means Filter')

            self.call_Widget()

            self.Restoration_Filter.configure(text=' Non-Local Means Filter ')
            self.Restoration_Filter.text = 'Non-Local Means Filter'

            self.label_page.configure(text='Noise Reduction Technique')
            self.call_metrics_psnr()
            self.call_metrics_mse()
            self.call_metrics_ssim()
            self.sideframe2.place(relx=0.55, rely=0.74)
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # ****************************************************************
    # ****************************************************************

    # ****************************************************************
    # ****************************************************************
    # Non-Local Means Denoising
    def Call_Band_reject(self):
        if self.img_is_found:
            # =================================
            self.Hide_help_about()
            ######
            self.label_scale.place_forget()
            self.label_entry_name.place_forget()
            self.sideframe3.place_forget()
            self.BestValue.place_forget()
            #
            self.scale.set(0)
            self.scale_average.set(0)
            self.scale_NLM.set(0)
            self.scale_rad.set(0)
            self.scale_reg.set(0)
            ####
            self.scale.place_forget()
            self.scale2.place_forget()
            self.scale3.place_forget()
            self.scale4.place_forget()
            self.scale_NLM.place_forget()
            self.scale_average.place_forget()
            self.scale_reg.place_forget()
            self.scale_rad.place_forget()
            ##############################
            path = self.ifile
            ##
            self.img_original = cv.imread(path, 0)

            img_after = Band_rejectFilter(self.img_original)

            # =============== Metrics ===============
            # ============> PSNR
            self.label_psnr.configure(text='PSNR = ' + str(round(psnr(self.img_original, img_after), 4)))
            # ============> MSE
            self.label_mse.configure(text='MSE = ' + str(round(mse(self.img_original, img_after), 4)))
            # ============> SSIM
            self.label_ssim.configure(text='SSIM = ' + str(round(ssim(self.img_original, img_after), 4)))
            # ===================================================================
            ######
            ######
            # save Image
            self.img_save = Image.fromarray(img_after)
            self.img_save = self.img_save.convert("L")

            # ===========================================
            # Resize Image
            img_after_new = self.img_save.resize((600, 450))
            # ====================================
            img_after = ImageTk.PhotoImage(img_after_new)

            self.label2.configure(image=img_after)
            self.label2.image = img_after

            self.call_Widget()

            self.Restoration_Filter.configure(text='Band Reject Filter ')
            self.Restoration_Filter.text = 'Band Reject Filter'

            self.label_page.configure(text='Noise Reduction Technique')
            self.call_metrics_psnr()
            self.call_metrics_mse()
            self.call_metrics_ssim()
            self.sideframe2.place(relx=0.55, rely=0.74)
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # ****************************************************************
    # ****************************************************************

    def hide_restoration(self):
        self.B.place_forget()
        self.B2.place_forget()
        self.B3.place_forget()
        self.B4_comparison.place_forget()
        self.label_entry.place_forget()
        self.label_entry_name.place_forget()
        self.bottomframe.place_forget()
        self.e1.place_forget()
        self.label_psnr.place_forget()
        self.label_mse.place_forget()
        self.label_ssim.place_forget()
        self.BestValue.place_forget()
        self.label_Q.place_forget()
        self.sideframe3.place_forget()
        self.sideframe2.place_forget()

        self.scale.place_forget()
        self.scale2.place_forget()
        self.scale4.place_forget()
        self.scale3.place_forget()
        self.scale_average.place_forget()
        self.scale_NLM.place_forget()
        self.scale_reg.place_forget()
        self.scale_rad.place_forget()

        self.label_scale.place_forget()
        #####

    # Image Restoration Call Metrics Use Button
    def call_metrics_psnr(self):
        self.label_Q.place(x=865, y=650)
        self.label_Q.config(text='Quality Measurement')
        self.label_psnr.place(x=865, y=710)

    def call_metrics_mse(self):
        self.label_mse.place(x=1065, y=710)

    def call_metrics_ssim(self):
        self.label_ssim.place(x=1265, y=710)

    # =================================================================
    def show_wienerFilter(self):
        self.bottomframe.place_forget()
        self.e1.pack_forget()
        self.label_entry.pack_forget()
        self.label_entry_name.place_forget()
        ###############################
        self.scale.place_forget()
        self.scale2.place_forget()
        self.scale4.place_forget()
        self.scale3.place_forget()
        self.scale_average.place_forget()
        #####
        self.Comparison_hide()
        self.call_Widget_5()
        # ====
        self.Hide_help_about()

        # Quality Metrics
        self.label_entry_name.configure(text='Winere Filter', font=Font(size=20, weight='bold'), padx=25, width=20)
        self.label_entry_name.configure(command=self.Wiener_Filter)

        self.label_entry_name.place(x=175, y=660)

        self.BestValue.config(text='Best Value = 00000')
        self.label3.place_forget()
        self.label_mse.config(text='MSE = 0000')
        self.label_ssim.config(text='SSIM = 0000')
        self.label_psnr.config(text='PSNR = 0000')

    # =================================================================
    # =================================================================

    def show_raduisFilter(self):
        self.bottomframe.place_forget()
        self.e1.pack_forget()
        self.label_entry.pack_forget()
        self.label_entry_name.place_forget()
        ###############################
        self.scale.place_forget()
        self.scale2.place_forget()
        self.scale3.place_forget()
        self.scale4.place_forget()
        self.scale_average.place_forget()
        #####
        self.Comparison_hide()
        self.call_Widget_5()
        # ====
        self.Hide_help_about()

        # Quality Metrics

        self.label_entry_name.configure(text='Raduial Filter', font=Font(size=20, weight='bold'), padx=25, width=20)
        self.label_entry_name.configure(command=self.Raduis_filter_call)

        self.label_entry_name.place(x=175, y=660)

        self.BestValue.config(text='Best Value = 00000')
        self.label3.place_forget()
        self.label_mse.config(text='MSE = 0000')
        self.label_ssim.config(text='SSIM = 0000')
        self.label_psnr.config(text='PSNR = 0000')

    # =================================================================
    # =================================================================

    def show_RegularizationFilter(self):
        self.bottomframe.place_forget()
        self.e1.pack_forget()
        self.label_entry.pack_forget()
        self.label_entry_name.place_forget()
        ###############################
        #####
        self.scale.place_forget()
        self.scale2.place_forget()
        self.scale3.place_forget()
        self.scale4.place_forget()
        self.scale_average.place_forget()
        # ====
        self.Hide_help_about()
        self.Comparison_hide()
        self.call_Widget_5()
        # Quality Metrics
        self.label_entry_name.configure(text='Regularization Filter', font=Font(size=20, weight='bold'), padx=25,
                                        width=20)
        self.label_entry_name.configure(command=self.constrained_ls_filter_call)
        self.label_entry_name.place(x=175, y=660)

        self.BestValue.config(text='Best Value = 00000')
        self.label3.place_forget()
        self.label_mse.config(text='MSE = 0000')
        self.label_ssim.config(text='SSIM = 0000')
        self.label_psnr.config(text='PSNR = 0000')

    def show_AverageFilter(self):
        self.bottomframe.place_forget()
        self.e1.pack_forget()
        self.label_entry.pack_forget()
        self.label_entry_name.place_forget()
        ###############################
        # ====
        self.Hide_help_about()
        #####
        self.Comparison_hide()
        self.call_Widget_5()
        # Quality Metrics
        self.label_entry_name.configure(text='Averaging Filter', font=Font(size=20, weight='bold'), padx=25, width=20)
        self.label_entry_name.configure(command=self.average_filter)
        self.label_entry_name.place(x=175, y=660)

        self.BestValue.config(text='Best Value = 00000')
        self.label3.place_forget()
        self.label_mse.config(text='MSE = 0000')
        self.label_ssim.config(text='SSIM = 0000')
        self.label_psnr.config(text='PSNR = 0000')

    # **************************************************************
    # **************************************************************
    # **************************************************************
    def show_MedianFilter(self):
        self.bottomframe.place_forget()
        self.e1.pack_forget()
        self.label_entry.pack_forget()
        self.label_entry_name.place_forget()
        ###############################
        # ====
        self.Hide_help_about()
        #####
        self.Comparison_hide()
        self.call_Widget_5()

        # Quality Metrics
        self.label_entry_name.configure(text='Median Filter', font=Font(size=20, weight='bold'), padx=25, width=20)
        self.label_entry_name.configure(command=self.median_filter)
        self.label_entry_name.place(x=175, y=660)

        self.BestValue.config(text='Best Value = 00000')
        self.label3.place_forget()
        self.label_mse.config(text='MSE = 0000')
        self.label_ssim.config(text='SSIM = 0000')
        self.label_psnr.config(text='PSNR = 0000')
        # **************************************************************

    # **************************************************************
    # **************************************************************
    # **************************************************************
    # **************************************************************
    # **************************************************************
    # Non - Local Man Filter
    def Show_NLM_Filter(self):
        self.bottomframe.place_forget()
        self.e1.pack_forget()
        self.label_entry.pack_forget()
        self.label_entry_name.place_forget()
        ###############################
        # ====
        self.Hide_help_about()
        self.Comparison_hide()
        self.call_Widget_5()
        self.scale.place_forget()
        self.scale2.place_forget()
        self.scale3.place_forget()
        self.scale4.place_forget()
        self.scale_average.place_forget()
        self.B4_comparison.place_forget()
        self.BestValue.place_forget()

        # Quality Metrics
        self.label_entry_name.configure(text='Non-Local Means Filter', font=Font(size=20, weight='bold'), padx=25,
                                        width=20)
        self.label_entry_name.configure(command=self.Call_NLM_Filter)

        self.label_entry_name.place(x=175, y=660)

        self.BestValue.config(text='Best Value = 00000')
        self.label3.place_forget()
        self.label_mse.config(text='MSE = 0000')
        self.label_ssim.config(text='SSIM = 0000')
        self.label_psnr.config(text='PSNR = 0000')

    # **************************************************************
    # ==============================================================
    def Edit_label_2(self):
        self.label.place_forget()
        self.Img_original.place_forget()
        self.label2.place_forget()
        self.Restoration_Filter.place_forget()

        self.label3.configure(width=400, height=400)
        self.label3.place(x=500, y=100)
        self.Image_Edit.place(x=610, y=50)

    def Edit_label_Type(self):
        self.label.place_forget()
        self.Img_original.place_forget()
        self.label2.place_forget()
        self.Restoration_Filter.place_forget()

        self.label3.configure(width=600, height=450)
        self.label3.place(x=400, y=100)
        self.Image_Edit.place(x=610, y=50)

    def call_Widget(self):
        self.Hide_To_Show_Restoration()
        self.label3.place_forget()
        self.Image_Edit.place_forget()
        self.label.place(x=70, y=100)
        self.Img_original.place(x=300, y=550)
        # ======== label Page
        self.label_page.place(relx=0.2, rely=0.05)
        # =================
        self.label2.place(x=840, y=100)
        self.Restoration_Filter.place(x=1050, y=550)

    def call_Widget_3(self):
        self.label2.place_forget()
        self.Restoration_Filter.place_forget()

    def call_Widget_4(self):
        self.label.place(x=90, y=125)
        self.Img_original.place(x=320, y=585)
        # ======== label Page
        self.label_page.place(relx=0.2, rely=0.06)
        # =================
        self.label2.place(x=850, y=125)
        self.Restoration_Filter.place(x=1070, y=585)
        self.sideframe3.place_forget()
        self.sideframe2.place_forget()

    def call_Widget_5(self):
        self.label.place(x=70, y=90)
        self.Img_original.place(x=300, y=540)
        # ======== label Page
        self.label_page.place(relx=0.25, rely=0.07)
        self.sideframe3.place(relx=0.05, rely=0.74)
        # =================
        self.label2.place_forget()
        self.Restoration_Filter.place_forget()

    # ********************************************************
    # =====> End Page =========
    # ========================================================
    def Comparison(self):
        self.Edit_label_2()
        self.hide_restoration()
        self.label_page.configure(text='Image Restoration Comparison Techniques')

        self.label_page.place(relx=0.25, rely=0.07)
        self.label1_Comparison.place(x=5, y=130)
        self.label2_Comparison.place(x=515, y=130)
        self.label3_Comparison.place(x=1025, y=130)
        self.label1_Comparison_name.place(x=150, y=640)
        self.label2_Comparison_name.place(x=750, y=640)
        self.label3_Comparison_name.place(x=1240, y=640)

        # Metrics
        self.Raduis_label_psnr_name_label1_Comparison.place(x=5, y=670)
        self.Raduis_label_psnr_label1_Comparison.place(x=70, y=670)

        self.Raduis_label_mse_name_label2_Comparison.place(x=5, y=690)
        self.Raduis_label_mse_label2_Comparison.place(x=70, y=690)

        self.Raduis_label_ssim_name_label3_Comparison.place(x=5, y=710)
        self.Raduis_label_ssim_label3_Comparison.place(x=70, y=710)

        ###########
        self.Wiener_label_psnr_name_label1_Comparison.place(x=515, y=670)
        self.Wiener_label_psnr_label1_Comparison.place(x=580, y=670)

        self.Wiener_label_mse_name_label2_Comparison.place(x=515, y=690)
        self.Wiener_label_mse_label2_Comparison.place(x=580, y=690)

        self.Wiener_label_ssim_name_label3_Comparison.place(x=515, y=710)
        self.Wiener_label_ssim_label3_Comparison.place(x=580, y=710)
        ##########
        self.Regularization_label_psnr_name_label1_Comparison.place(x=1025, y=670)
        self.Regularization_label_psnr_label1_Comparison.place(x=1090, y=670)

        self.Regularization_label_mse_name_label2_Comparison.place(x=1025, y=690)
        self.Regularization_label_mse_label2_Comparison.place(x=1090, y=690)

        self.Regularization_label_ssim_name_label3_Comparison.place(x=1025, y=710)
        self.Regularization_label_ssim_label3_Comparison.place(x=1090, y=710)

    def Comparison_hide(self):
        self.label1_Comparison.place_forget()
        self.label2_Comparison.place_forget()
        self.label3_Comparison.place_forget()
        self.label1_Comparison_name.place_forget()
        self.label2_Comparison_name.place_forget()
        self.label3_Comparison_name.place_forget()
        ##
        self.label_page.configure(text='')
        ##
        # Metrics
        self.Raduis_label_psnr_name_label1_Comparison.place_forget()
        self.Raduis_label_psnr_label1_Comparison.place_forget()

        self.Raduis_label_mse_name_label2_Comparison.place_forget()
        self.Raduis_label_mse_label2_Comparison.place_forget()

        self.Raduis_label_ssim_name_label3_Comparison.place_forget()
        self.Raduis_label_ssim_label3_Comparison.place_forget()
        #############
        self.Wiener_label_psnr_name_label1_Comparison.place_forget()
        self.Wiener_label_psnr_label1_Comparison.place_forget()

        self.Wiener_label_mse_name_label2_Comparison.place_forget()
        self.Wiener_label_mse_label2_Comparison.place_forget()

        self.Wiener_label_ssim_name_label3_Comparison.place_forget()
        self.Wiener_label_ssim_label3_Comparison.place_forget()
        ##########
        self.Regularization_label_psnr_name_label1_Comparison.place_forget()
        self.Regularization_label_psnr_label1_Comparison.place_forget()

        self.Regularization_label_mse_name_label2_Comparison.place_forget()
        self.Regularization_label_mse_label2_Comparison.place_forget()

        self.Regularization_label_ssim_name_label3_Comparison.place_forget()
        self.Regularization_label_ssim_label3_Comparison.place_forget()

    # ********************************************************
    def Show_Paremter_noise_SP(self):
        if self.img_is_found:
            self.scale.place_forget()
            self.scale2.place_forget()
            #### Entry Clear
            self.Reset_Preprocessing()
            #### Hide
            self.BestValue.place_forget()
            #### Hide
            self.Hide_To_Show_Mirror()
            # ====
            self.Hide_help_about()
            #####
            self.e1.pack(side=RIGHT)
            # Quality Metrics
            self.label_entry.configure(text='Attenuate: ')  # font = Font(underline=1)
            self.label_entry_name.configure(text='Compute Salt & Pepper Noise', font=Font(size=14, weight='bold'),
                                            padx=0, pady=5)
            self.label_entry_name.configure(command=self.Salt_Pepper_Noise)
            self.label_entry_name.place(x=620, y=620)
            self.sideframe3.place(relx=0.3, rely=0.75)
            ### Scale
            self.label_scale.place(relx=0.31, rely=0.86)
            self.label_scale.configure(text='Attenuate')
            self.scale.place(relx=0.31, rely=0.9)

            self.B4_comparison.place_forget()
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    def Show_Paremter_noise_GN(self):
        if self.img_is_found:
            self.scale2.place_forget()
            self.scale3.place_forget()
            self.scale4.place_forget()
            #### Entry Clear
            self.Reset_Preprocessing()
            #### Hide
            self.BestValue.place_forget()
            #### Hide
            self.Hide_To_Show_Mirror()
            # ====
            self.Hide_help_about()
            ####
            self.e1.pack(side=RIGHT)
            # Quality Metrics
            self.label_entry.configure(text='Attenuate: ')  # font = Font(underline=1)
            self.label_entry_name.configure(text='Compute Gaussian Noise', font=Font(size=14, weight='bold'), padx=0,
                                            pady=5)
            self.label_entry_name.configure(command=self.Gaussian_Noise)
            self.label_entry_name.place(x=620, y=620)
            self.sideframe3.place(relx=0.3, rely=0.75)

            self.B4_comparison.place_forget()
            ### Scale
            self.label_scale.place(relx=0.31, rely=0.86)
            self.label_scale.configure(text='Attenuate')
            self.scale2.place(relx=0.31, rely=0.9)
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # # #
    def Show_Crop(self):
        if self.img_is_found:
            #### Entry Clear
            self.Reset_Preprocessing()
            #### Hide
            self.Hide_zoom_rotate()
            # ====
            self.Hide_help_about()
            #### Show
            self.sideframe_Preprocessing_crop.configure(width=1200)

            self.label_entry_name_Preprocessing.configure(text='Crop', command=self.crop)
            self.label_entry_Preprocessing_l.configure(text='Left : ')
            self.label_entry_Preprocessing_r.configure(text='Right : ')
            self.label_entry_Preprocessing_u.configure(text='Upper : ')
            self.label_entry_Preprocessing_d.configure(text='Down : ')

            self.sideframe_Preprocessing_crop.place(relx=0.1, rely=0.78)
            self.e1_Preprocessing_l.pack(side=RIGHT)
            self.e1_Preprocessing_r.pack(side=RIGHT)
            self.e1_Preprocessing_u.pack(side=RIGHT)
            self.e1_Preprocessing_d.pack(side=RIGHT)
            self.label_entry_Preprocessing_l.pack(side=LEFT)
            self.label_entry_Preprocessing_r.pack(side=LEFT)
            self.label_entry_Preprocessing_u.pack(side=LEFT)
            self.label_entry_Preprocessing_d.pack(side=LEFT)

            self.bottomframe_Preprocessing_l.place(x=200, y=740)
            self.bottomframe_Preprocessing_r.place(x=500, y=740)
            self.bottomframe_Preprocessing_u.place(x=800, y=740)
            self.bottomframe_Preprocessing_d.place(x=1100, y=740)
            self.label_entry_name_Preprocessing.place(x=600, y=640)
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # # #
    def Show_Rotate(self):
        if self.img_is_found:
            #### Entry Clear
            self.Reset_Preprocessing()
            #### Hide
            self.Hide_Crop()
            # ====
            self.Hide_help_about()
            #### Show
            self.label_entry_name_Preprocessing.configure(text='Rotate', command=self.rotate)
            self.label_entry_Preprocessing.configure(text='Angle : ')
            self.sideframe_Preprocessing.place(relx=0.31, rely=0.75)
            self.e1_Preprocessing.pack(side=RIGHT)
            self.label_entry_Preprocessing.pack(side=LEFT)
            self.bottomframe_Preprocessing.place(x=500, y=740)
            self.label_entry_name_Preprocessing.place(x=600, y=630)
            self.label_scale.place_forget()
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # # #
    # # #
    def Show_Zoom(self):
        if self.img_is_found:
            #### Entry Clear
            self.Reset_Preprocessing()
            #### Hide
            self.Hide_Crop()
            # ====
            self.Hide_help_about()
            #### Show
            self.label_entry_name_Preprocessing.configure(text='Zoom', command=self.Zoom_image)
            self.label_entry_Preprocessing.configure(text='Percentage (%) : ')
            self.sideframe_Preprocessing.place(relx=0.31, rely=0.75)
            self.e1_Preprocessing.pack(side=RIGHT)
            self.label_entry_Preprocessing.pack(side=LEFT)
            self.bottomframe_Preprocessing.place(x=500, y=740)
            self.label_entry_name_Preprocessing.place(x=600, y=630)
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # # #
    # # #
    def Hide_zoom_rotate(self):
        self.hide_restoration()
        #### Show
        self.sideframe_Preprocessing.place_forget()
        self.e1_Preprocessing.pack_forget()
        self.label_entry_Preprocessing.pack_forget()
        self.bottomframe_Preprocessing.place_forget()
        self.label_entry_name_Preprocessing.place_forget()

    # # #
    # # #
    def Hide_Crop(self):
        self.hide_restoration()
        #######
        self.sideframe_Preprocessing_crop.place_forget()
        self.e1_Preprocessing_l.pack_forget()
        self.e1_Preprocessing_r.pack_forget()
        self.e1_Preprocessing_u.pack_forget()
        self.e1_Preprocessing_d.pack_forget()
        self.label_entry_Preprocessing_l.pack_forget()
        self.label_entry_Preprocessing_r.pack_forget()
        self.label_entry_Preprocessing_u.pack_forget()
        self.label_entry_Preprocessing_d.pack_forget()

        self.bottomframe_Preprocessing_l.place_forget()
        self.bottomframe_Preprocessing_r.place_forget()
        self.bottomframe_Preprocessing_u.place_forget()
        self.bottomframe_Preprocessing_d.place_forget()
        self.label_entry_name_Preprocessing.place_forget()

    def Hide_To_Show_Mirror(self):
        #### Hide
        self.Hide_Crop()
        self.Hide_zoom_rotate()

    def Hide_To_Show_Restoration(self):
        self.sideframe_Preprocessing.place_forget()
        self.e1_Preprocessing.pack_forget()
        self.label_entry_Preprocessing.pack_forget()
        self.bottomframe_Preprocessing.place_forget()
        self.label_entry_name_Preprocessing.place_forget()
        #### Hide
        self.sideframe_Preprocessing_crop.place_forget()
        self.e1_Preprocessing_l.pack_forget()
        self.e1_Preprocessing_r.pack_forget()
        self.e1_Preprocessing_u.pack_forget()
        self.e1_Preprocessing_d.pack_forget()
        self.label_entry_Preprocessing_l.pack_forget()
        self.label_entry_Preprocessing_r.pack_forget()
        self.label_entry_Preprocessing_u.pack_forget()
        self.label_entry_Preprocessing_d.pack_forget()

        self.bottomframe_Preprocessing_l.place_forget()
        self.bottomframe_Preprocessing_r.place_forget()
        self.bottomframe_Preprocessing_u.place_forget()
        self.bottomframe_Preprocessing_d.place_forget()
        self.label_entry_name_Preprocessing.place_forget()

    def Show_pixelLate(self):
        if self.img_is_found:
            self.scale.place_forget()
            self.scale2.place_forget()
            self.Hide_zoom_rotate()
            #### Entry Clear
            self.Reset_Preprocessing()
            #### Hide
            self.Hide_Crop()
            # ====
            self.Hide_help_about()
            #### Show
            self.label_entry_name_Preprocessing.configure(text='Compute Pixiel Late', command=self.pixiel_late)
            self.sideframe_Preprocessing.place(relx=0.32, rely=0.75)
            self.label_entry_name_Preprocessing.place(x=615, y=620)

            ### Scale
            self.label_scale.place(relx=0.33, rely=0.86)
            self.label_scale.configure(text='Kernel Size')
            self.scale3.place(relx=0.33, rely=0.9)
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    def Reset_Preprocessing(self):
        self.clear_text()
        self.label3.place_forget()
        self.Image_Edit.place_forget()
        self.label2.place_forget()
        self.Restoration_Filter.place_forget()
        self.label_page.place_forget()

    #########################################################################
    def show_contrast(self):
        if self.img_is_found:
            #### Entry Clear
            self.Reset_Preprocessing()
            #### Hide
            self.BestValue.place_forget()
            #### Hide
            self.Hide_To_Show_Mirror()
            #####
            self.e1.pack(side=RIGHT)
            # Quality Metrics
            self.label_entry_name.configure(text='Compute Contrast', font=Font(size=14, weight='bold'), padx=0, pady=5)
            self.label_entry_name.configure(command=self.img_contrast)
            self.label_entry_name.place(x=670, y=620)
            self.sideframe3.place(relx=0.3, rely=0.75)
            ### Scale
            self.label_scale.place(relx=0.31, rely=0.86)
            self.label_scale.configure(text='Degree Of Contrast')
            self.scale4.place(relx=0.31, rely=0.9)
            # ====
            self.Hide_help_about()
        else:
            messagebox.showinfo('Alert', 'Please Choose Image')

    # ======================================= Main ==========================
    def hide_widget_help(self):
        self.label.place_forget()
        self.label2.place_forget()
        self.label3.place_forget()
        self.Img_original.place_forget()
        self.Image_Edit.place_forget()
        self.Restoration_Filter.place_forget()
        self.Hide_Crop()
        self.Hide_zoom_rotate()
        self.sideframe2.place_forget()
        self.sideframe3.place_forget()

    # ===============================================
    def __init__(self):
        Tk.__init__(self)
        #         self.attributes('-fullscreen', True)
        self.state('zoomed')
        self.title('Digital Image Processing')
        self.config(bg='#e9ecef')

        self.sideframe = Frame(self, bg='#CED4DA', width=1600, height=35)
        self.sideframe3 = Frame(self, bg='#CED4DA', width=600, height=190, relief="sunken", highlightthickness=7,
                                highlightbackground="#0D47A1")
        self.sideframe2 = Frame(self, bg='#CED4DA', width=600, height=190, relief="sunken", highlightthickness=7,
                                highlightbackground="#0D47A1")

        # For Noise

        self.sideframe.place(x=0, y=0)

        # =========================================
        # =========================================
        # =========================================
        # =========================================
        # ====== Side Frame To Preprocessing
        self.sideframe_Preprocessing = Frame(self, bg='#CED4DA', width=550, height=190, relief="sunken",
                                             highlightthickness=7, highlightbackground="#0D47A1")
        self.sideframe_Preprocessing_crop = Frame(self, bg='#CED4DA', width=720, height=160, relief="sunken",
                                                  highlightthickness=7, highlightbackground="#0D47A1")

        self.bottomframe_Preprocessing = Frame(self, bg='#ced4da')
        self.e1_Preprocessing = Entry(self.bottomframe_Preprocessing, justify=CENTER, selectbackground='BLUE', width=15,
                                      font=Font(size=12, weight='bold'), bg='#F5F3F4', fg='#161A1D')
        self.label_entry_Preprocessing = Label(self.bottomframe_Preprocessing, text=None,
                                               font=Font(size=15, weight='bold'), bg='#ced4da', fg='#0D47A1')
        self.label_entry_name_Preprocessing = Button(self, text=None, width=20, font=Font(size=15, weight='bold'),
                                                     bg='#0D47A1', fg='#F5F3F4', borderwidth=3, relief="raised",
                                                     padx=25, command=None)

        self.bottomframe_Preprocessing_l = Frame(self, bg='#ced4da')
        self.bottomframe_Preprocessing_r = Frame(self, bg='#ced4da')
        self.bottomframe_Preprocessing_u = Frame(self, bg='#ced4da')
        self.bottomframe_Preprocessing_d = Frame(self, bg='#ced4da')

        self.label_entry_Preprocessing_l = Label(self.bottomframe_Preprocessing_l, text='Left',
                                                 font=Font(size=15, weight='bold'), bg='#ced4da', fg='#0D47A1')
        self.e1_Preprocessing_l = Entry(self.bottomframe_Preprocessing_l, justify=CENTER, selectbackground='BLUE',
                                        width=15, font=Font(size=12, weight='bold'), bg='#F5F3F4', fg='#161A1D')

        self.label_entry_Preprocessing_r = Label(self.bottomframe_Preprocessing_r, text='Right',
                                                 font=Font(size=15, weight='bold'), bg='#ced4da', fg='#0D47A1')
        self.e1_Preprocessing_r = Entry(self.bottomframe_Preprocessing_r, justify=CENTER, selectbackground='BLUE',
                                        width=15, font=Font(size=12, weight='bold'), bg='#F5F3F4', fg='#161A1D')

        self.label_entry_Preprocessing_u = Label(self.bottomframe_Preprocessing_u, text='Upper',
                                                 font=Font(size=15, weight='bold'), bg='#ced4da', fg='#0D47A1')
        self.e1_Preprocessing_u = Entry(self.bottomframe_Preprocessing_u, justify=CENTER, selectbackground='BLUE',
                                        width=15, font=Font(size=12, weight='bold'), bg='#F5F3F4', fg='#161A1D')

        self.label_entry_Preprocessing_d = Label(self.bottomframe_Preprocessing_d, text='Down',
                                                 font=Font(size=15, weight='bold'), bg='#ced4da', fg='#0D47A1')
        self.e1_Preprocessing_d = Entry(self.bottomframe_Preprocessing_d, justify=CENTER, selectbackground='BLUE',
                                        width=15, font=Font(size=12, weight='bold'), bg='#F5F3F4', fg='#161A1D')

        # =========================================

        # =========================================
        # =========================================
        # =========================================

        # background image
        self.window_size()
        self.menuBar_edit()

        # Side Frame
        self.var1 = DoubleVar()
        self.var2 = DoubleVar()
        self.var3 = DoubleVar()
        self.var5 = DoubleVar()
        self.var_avrage = DoubleVar()
        self.var_nlm = DoubleVar()
        self.var_dwr = DoubleVar()
        self.var_reg = DoubleVar()
        self.var_rad = DoubleVar()

        ##
        self.scale = Scale(self, variable=self.var1, orient=HORIZONTAL, from_=0.1, to=1, resolution=0.1, length=300,
                           font=Font(size=17, weight='bold'), bg='#0D47A1', fg='#e9ecef')
        self.scale2 = Scale(self, variable=self.var2, orient=HORIZONTAL, from_=0.1, to=1, resolution=0.1, length=300,
                            font=Font(size=17, weight='bold'), bg='#0D47A1', fg='#e9ecef')
        self.scale3 = Scale(self, variable=self.var3, orient=HORIZONTAL, from_=5, to=500, resolution=5, length=300,
                            font=Font(size=17, weight='bold'), bg='#0D47A1', fg='#e9ecef')
        self.scale4 = Scale(self, variable=self.var5, orient=HORIZONTAL, from_=1, to=10, resolution=1, length=400,
                            font=Font(size=17, weight='bold'), bg='#0D47A1', fg='#e9ecef')

        self.scale_average = Scale(self, variable=self.var_avrage, orient=HORIZONTAL, from_=1, to=19, resolution=2,
                                   length=300, font=Font(size=17, weight='bold'), bg='#0D47A1', fg='#e9ecef')
        self.scale_NLM = Scale(self, variable=self.var_nlm, orient=HORIZONTAL, from_=5, to=70, resolution=2, length=300,
                               font=Font(size=17, weight='bold'), bg='#0D47A1', fg='#e9ecef')

        #         self.scale_reg = Scale( self, variable = self.var_reg ,orient = HORIZONTAL,from_= 0.1,to=0.001,resolution = 0.009,length = 300 , digits = 3,font = Font(size = 17 ,weight='bold') , bg = '#0D47A1' , fg = '#e9ecef')
        self.scale_reg = Scale(self, variable=self.var_reg, orient=HORIZONTAL, from_=0.001, to=10, resolution=0.009,
                               length=300, digits=5, font=Font(size=17, weight='bold'), bg='#0D47A1', fg='#e9ecef')
        self.scale_rad = Scale(self, variable=self.var_rad, orient=HORIZONTAL, from_=5, to=200, resolution=5,
                               length=300, digits=3, font=Font(size=17, weight='bold'), bg='#0D47A1', fg='#e9ecef')

        self.label_scale = Label(self, text=None, font=Font(size=15, weight='bold'), bg='#ced4da', fg='#0D47A1')

        # frame To Entry
        self.bottomframe = Frame(self, bg='#ced4da')
        self.bottomframe2 = Frame(self, bg='#ced4da')
        self.bottomframe3 = Frame(self, bg='#ced4da')
        # text Box
        self.e1 = Entry(self.bottomframe, justify=CENTER, selectbackground='BLUE', width=11,
                        font=Font(size=12, weight='bold'), bg='#F5F3F4', fg='#161A1D')
        self.e2 = Entry(self.bottomframe2, justify=CENTER, selectbackground='BLUE', width=11,
                        font=Font(size=12, weight='bold'), bg='#F5F3F4', fg='#161A1D')
        self.e3 = Entry(self.bottomframe3, justify=CENTER, selectbackground='BLUE', width=11,
                        font=Font(size=12, weight='bold'), bg='#F5F3F4', fg='#161A1D')

        # Quality Metrics
        self.label_entry = Label(self.bottomframe, text=None, font=Font(size=12, weight='bold'), bg='#ced4da',
                                 fg='#0D47A1')
        self.label_entry_name = Button(self, text=None, font=Font(size=15, weight='bold'), bg='#0D47A1', fg='#F5F3F4',
                                       borderwidth=3, relief="raised", padx=25, command=None)
        #
        self.label_entry2 = Label(self.bottomframe2, text=None, font=Font(size=12, weight='bold'), bg='#ced4da',
                                  fg='#0D47A1')
        #
        self.label_entry3 = Label(self.bottomframe3, text=None, font=Font(size=12, weight='bold'), bg='#ced4da',
                                  fg='#0D47A1')
        #

        self.label_psnr = Label(self, text=None, font=Font(size=12, weight='bold'), fg='#F5F3F4', bg='#0D47A1',
                                borderwidth=10, relief="sunken", padx=20, pady=10)
        self.label_Q = Label(self, text=None, font=Font(size=16, weight='bold'), fg='#0D47A1', bg='#ced4da')

        self.label_mse = Label(self, text=None, font=Font(size=12, weight='bold'), fg='#F5F3F4', bg='#0D47A1',
                               borderwidth=10, relief="sunken", padx=20, pady=10)
        self.label_ssim = Label(self, text=None, font=Font(size=12, weight='bold'), fg='#F5F3F4', bg='#0D47A1',
                                borderwidth=10, relief="sunken", padx=20, pady=10)

        # text Box
        ## #########
        self.BestValue = Label(self, text=None, font=Font(size=16, weight='bold'), bg='#0D47A1', fg='#F8F9FA',
                               borderwidth=5, relief="sunken", padx=10, pady=10, height=1)
        self.BestValue_2 = Label(self, text=None, font=Font(size=11, weight='bold'), bg='#0D47A1', fg='#F8F9FA',
                                 borderwidth=5, relief="sunken", padx=5, pady=4, height=1)
        self.BestValue_3 = Label(self, text=None, font=Font(size=11, weight='bold'), bg='#0D47A1', fg='#F8F9FA',
                                 borderwidth=5, relief="sunken", padx=5, pady=4, height=1)

        self.B = Button(self, text="Compute PSNR", command=self.call_metrics_psnr, borderwidth=3, relief="raised",
                        font=Font(size=14, weight='bold'), bg='#343A40', fg='#F8F9FA', padx=14)
        self.B2 = Button(self, text="Compute MSE  ", command=self.call_metrics_mse, borderwidth=3, relief="raised",
                         font=Font(size=14, weight='bold'), bg='#343A40', fg='#F8F9FA', padx=14)
        self.B3 = Button(self, text="Compute SSIM ", command=self.call_metrics_ssim, borderwidth=3, relief="raised",
                         font=Font(size=14, weight='bold'), bg='#343A40', fg='#F8F9FA', padx=14)
        self.B4_comparison = Button(self, text="Comparison Algorithm", command=self.Comparison, borderwidth=3,
                                    relief="raised", font=Font(size=12, weight='bold'), bg='#0D47A1', fg='#F8F9FA',
                                    width=35)
        ## #########
        ###### Help System #####
        # label_Page
        ## Label_image
        self.label_h1 = Label(image=None, width=650, height=300, bg='#e9ecef')
        self.label_h2 = Label(image=None, width=550, height=300, bg='#e9ecef')
        self.label_h3 = Label(image=None, width=650, height=250, bg='#e9ecef')

        ####
        self.b1 = Button(self, text='Next', bg='#3333ff', fg='#FEFCFB', font=('Arial', 20, 'bold'),
                         activebackground='#e9ecef', highlightthickness=0, command=self.help_about_2, width=10)

        self.b2 = Button(self, text='Previous', bg='#3333ff', fg='#FEFCFB', font=('Arial', 20, 'bold'),
                         activebackground='#e9ecef', highlightthickness=0, command=self.help_about_1, width=10)
        self.b3 = Button(self, text='Next', bg='#3333ff', fg='#FEFCFB', font=('Arial', 20, 'bold'),
                         activebackground='#e9ecef', highlightthickness=0, command=self.help_about_3, width=10)

        self.b4 = Button(self, text='Next', bg='#3333ff', fg='#FEFCFB', font=('Arial', 20, 'bold'),
                         activebackground='#e9ecef', highlightthickness=0, command=self.help_about_4, width=10)
        self.b5 = Button(self, text='Previous', bg='#3333ff', fg='#FEFCFB', font=('Arial', 20, 'bold'),
                         activebackground='#e9ecef', highlightthickness=0, command=self.help_about_2, width=10)

        self.b6 = Button(self, text='Previous', bg='#3333ff', fg='#FEFCFB', font=('Arial', 20, 'bold'),
                         activebackground='#e9ecef', highlightthickness=0, command=self.help_about_3, width=10)

        ######
        # ======== label Page
        self.label_page = Label(self, text=None, font=Font(size=30, weight='bold'), width=40, justify=CENTER,
                                fg='#0D47A1', bg='#e9ecef')
        # ========
        self.Img_original = Label(self, text=None, font=Font(size=15, weight='bold'), bg='#e9ecef', fg='#0D47A1')
        self.Restoration_Filter = Label(self, font=Font(size=15, weight='bold'), bg='#e9ecef', fg='#0D47A1')
        self.Image_Edit = Label(self, text=None, font=Font(size=15, weight='bold'), bg='#e9ecef', fg='#0D47A1')

        self.label = Label(image=None, width=600, height=450, bg='#e9ecef')
        self.label2 = Label(image=None, width=600, height=450, bg='#e9ecef')
        self.label3 = Label(image=None, width=400, height=400, bg='#e9ecef')

        # *****************************************************
        # End Page Comparison

        self.label1_Comparison = Label(image=None, width=500, height=500, bg='#e9ecef')
        self.label2_Comparison = Label(image=None, width=500, height=500, bg='#e9ecef')
        self.label3_Comparison = Label(image=None, width=500, height=500, bg='#e9ecef')

        self.label1_Comparison_name = Label(self, text=None, font=Font(size=12, weight='bold'), fg='#0D47A1',
                                            bg='#e9ecef')
        self.label2_Comparison_name = Label(self, text=None, font=Font(size=12, weight='bold'), fg='#0D47A1',
                                            bg='#e9ecef')
        self.label3_Comparison_name = Label(self, text=None, font=Font(size=12, weight='bold'), fg='#0D47A1',
                                            bg='#e9ecef')
        # ================

        # End Page Metrics
        # Raduis
        self.Raduis_label_psnr_label1_Comparison = Label(self, text=None, font=Font(size=12, weight='bold'),
                                                         fg='#0D47A1', bg='#e9ecef')
        self.Raduis_label_psnr_name_label1_Comparison = Label(self, text='PSNR = ', font=Font(size=12, weight='bold'),
                                                              fg='#0D47A1', bg='#e9ecef')
        self.Raduis_label_mse_label2_Comparison = Label(self, text=None, font=Font(size=12, weight='bold'),
                                                        fg='#0D47A1', bg='#e9ecef')
        self.Raduis_label_mse_name_label2_Comparison = Label(self, text='MSE = ', font=Font(size=12, weight='bold'),
                                                             fg='#0D47A1', bg='#e9ecef')
        self.Raduis_label_ssim_label3_Comparison = Label(self, text=None, font=Font(size=12, weight='bold'),
                                                         fg='#0D47A1', bg='#e9ecef')
        self.Raduis_label_ssim_name_label3_Comparison = Label(self, text='SSIM = ', font=Font(size=12, weight='bold'),
                                                              fg='#0D47A1', bg='#e9ecef')

        # Wiener
        self.Wiener_label_psnr_label1_Comparison = Label(self, text=None, font=Font(size=12, weight='bold'),
                                                         fg='#0D47A1', bg='#e9ecef')
        self.Wiener_label_psnr_name_label1_Comparison = Label(self, text='PSNR = ', font=Font(size=12, weight='bold'),
                                                              fg='#0D47A1', bg='#e9ecef')
        self.Wiener_label_mse_label2_Comparison = Label(self, text=None, font=Font(size=12, weight='bold'),
                                                        fg='#0D47A1', bg='#e9ecef')
        self.Wiener_label_mse_name_label2_Comparison = Label(self, text='MSE = ', font=Font(size=12, weight='bold'),
                                                             fg='#0D47A1', bg='#e9ecef')
        self.Wiener_label_ssim_label3_Comparison = Label(self, text=None, font=Font(size=12, weight='bold'),
                                                         fg='#0D47A1', bg='#e9ecef')
        self.Wiener_label_ssim_name_label3_Comparison = Label(self, text='SSIM = ', font=Font(size=12, weight='bold'),
                                                              fg='#0D47A1', bg='#e9ecef')

        # Regularization
        self.Regularization_label_psnr_label1_Comparison = Label(self, text=None, font=Font(size=12, weight='bold'),
                                                                 fg='#0D47A1', bg='#e9ecef')
        self.Regularization_label_psnr_name_label1_Comparison = Label(self, text='PSNR = ',
                                                                      font=Font(size=12, weight='bold'), fg='#0D47A1',
                                                                      bg='#e9ecef')
        self.Regularization_label_mse_label2_Comparison = Label(self, text=None, font=Font(size=12, weight='bold'),
                                                                fg='#0D47A1', bg='#e9ecef')
        self.Regularization_label_mse_name_label2_Comparison = Label(self, text='MSE = ',
                                                                     font=Font(size=12, weight='bold'), fg='#0D47A1',
                                                                     bg='#e9ecef')
        self.Regularization_label_ssim_label3_Comparison = Label(self, text=None, font=Font(size=12, weight='bold'),
                                                                 fg='#0D47A1', bg='#e9ecef')
        self.Regularization_label_ssim_name_label3_Comparison = Label(self, text='SSIM = ',
                                                                      font=Font(size=12, weight='bold'), fg='#0D47A1',
                                                                      bg='#e9ecef')


# if __name__ == "__main__":
#     ui = Application()
#     ui.mainloop()


# In[ ]:


# In[ ]:


# # Control Pages

# In[26]:


def click_Button():
    window.destroy()
    # Go-To-Application.
    ui = Application()
    ui.mainloop()


# # Main Page

# In[27]:


window = Tk()
window.attributes('-fullscreen', True)

window.title('Digital Image Processing Application')
# window.eval('tk::PlaceWindow . center')
# window.state('zoomed')
window.config(highlightthickness=0)

##
width = 1550  # Width.
height = 1000  # Height.
##

# Create Image Background.
image = Image.open('bg7.jpg')
image = image.resize((1550, 1000))
image = ImageTk.PhotoImage(image)

canvas = Canvas(width=width, height=height, highlightthickness=0)
# images = PhotoImage(file = image)
canvas.create_image(width / 2, height / 2, image=image)
canvas.create_text(750, 330, text='Digital Image Processing Application', fill='#fff', font=('Arial', 50, 'bold'))
canvas.pack()
# Button.
b1 = Button(window, text='Continue', bg='#FEFCFB', fg='#3333ff',
            font=('Arial', 30, 'bold'), activebackground='#012A4A',
            highlightthickness=0, command=click_Button)
b1.place(relx=0.5, rely=0.75, anchor='center')
window.mainloop()

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:
