import cv2 
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from functools import partial
import time
import numpy as np
import pickle
import os
from os import path
from PIL import Image
def nothing(x):
    pass

def blur_score(img: np.ndarray) -> np.float64:
    return cv2.Laplacian(img, cv2.CV_64F).var()

def contrast_score(img: np.ndarray) -> np.float64:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    contrast = blurred.std()
    return contrast


def getMSSISM(i1, i2):
    C1 = 6.5025
    C2 = 58.5225
    I1 = np.float32(i1) 
    I2 = np.float32(i2)
    I2_2 = I2 * I2 
    I1_2 = I1 * I1 
    I1_I2 = I1 * I2 
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2                   
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2                   
    ssim_map = cv2.divide(t3, t1)   
    mssim = cv2.mean(ssim_map)      
    return mssim

def limit_decimal_places(float_array):
    rounded_array = [round(num, 4) for num in float_array]
    return rounded_array

class ThermalAnalysisToolkit:
    def __init__(self, master):
        self.master = master
        self.master.title("ThermalAnalysisToolkit")
        self.method_options_var = tk.StringVar()
        self.method_options_var.set("cv2.THRESH_BINARY")
        self.mode_options_var = tk.StringVar()
        self.mode_options_var.set("cv2.bilateralFilter")
        self.contour_method_var = tk.StringVar()
        self.contour_method_var.set("cv2.RETR_LIST")
        self.contour_mode_var = tk.StringVar()
        self.contour_mode_var.set("cv2.CHAIN_APPROX_NONE")
        self.contrast_var =tk.BooleanVar()
        self.contrast_var.set(1)
        self.noise_var =tk.BooleanVar()
        self.noise_var.set(1)
        self.is_serch_var=tk.BooleanVar()
        self.is_serch_var.set(1)
        self.is_contours_var =tk.BooleanVar()
        self.is_contours_var.set(1)
        self.new_settings = {
            "method_options_var": self.method_options_var,
            "mode_options_var": self.mode_options_var,
            "contour_method_var": self.contour_method_var,
            "contour_mode_var": self.contour_mode_var,
            "contrast_var": self.contrast_var,
            "noise_var": self.noise_var,
            "is_serch_var": self.is_serch_var,
            "is_contours_var": self.is_contours_var,
        }
        
        self.seting = {
            "minH_var": tk.IntVar(value=20),
            "minL_var": tk.IntVar(value=20),
            "maxH_var": tk.IntVar(value=50),
            "maxL_var": tk.IntVar(value=80),
            "maxObj_var": tk.IntVar(value=5),
            "maxFPS_var": tk.IntVar(value=8),
            "diameterBilateralFilter": tk.IntVar(value=9),
            "sigmaColor": tk.IntVar(value=75),
            "sigmaSpace": tk.IntVar(value=75),
            "GaussianKernelSizeHeight": tk.IntVar(value=9),
            "GaussianKernelSizeWidth": tk.IntVar(value=9),
            "KernelStandardDeviationXY": tk.IntVar(value=2),
            "KernelSizeHeight": tk.IntVar(value=9),
            "KernelSizeWidth": tk.IntVar(value=9),
            "KernelSize": tk.IntVar(value=9),
            "hue": tk.IntVar(value=179),
            "saturation": tk.IntVar(value=255),
            "value": tk.IntVar(value=255),
            "hueL": tk.IntVar(value=0),
            "saturationL": tk.IntVar(value=0),
            "valueL": tk.IntVar(value=0),
            "threshold": tk.IntVar(value=127),
            "threshold_max": tk.IntVar(value=255),
            "maxLine": tk.IntVar(value=10),
            "minLine": tk.IntVar(value=3),
            "minLine": tk.IntVar(value=3),
            "l2": tk.IntVar(value=0),
            "a2": tk.IntVar(value=0),
            "b2": tk.IntVar(value=0),
            "clipLimit": tk.IntVar(value=4),
            "tileGridSizeH": tk.IntVar(value=8),
            "tileGridSizeW": tk.IntVar(value=8),
            "clipLimit": tk.IntVar(value=2),
            "tileGridSizeH": tk.IntVar(value=8),
            "tileGridSizeW": tk.IntVar(value=8),
        }
        
        self.frameTime= tk.getdouble()
        self.frameTime=0
        self.path = None
        self.img = None
        self.label_path = tk.Label(master, text="Шлях до зображення:")
        self.label_path.grid(row=0, column=0, padx=10, pady=10)
        self.entry_path = tk.Entry(master, width=30)
        self.entry_path.grid(row=0, column=1, padx=10, pady=10)
        self.btn_browse = tk.Button(master, text="Обрати зображення", command=self.browse_video)
        self.btn_browse.grid(row=0, column=2, padx=10, pady=10)
        self.label_method = tk.Label(master, text="Метод виділення контурів:")
        self.label_method.grid(row=3, column=0, padx=10, pady=10)
        self.method_options_combobox = ttk.Combobox(root,values= ["cv2.THRESH_BINARY", "cv2.THRESH_BINARY_INV",
                                                                  "cv2.THRESH_TRUNC", "cv2.THRESH_TOZERO",
                                                                  "cv2.THRESH_TOZERO_INV", "opening", "closing",
                                                                  "cv2.Canny_Edge_Detection", "none"],
                                                    textvariable=self.method_options_var)
        self.method_options_combobox.grid(row=3, column=1, padx=10, pady=10)
        self.label_mode = tk.Label(master, text="Метод усунення шуму:")
        self.label_mode.grid(row=1, column=0, padx=10, pady=10)
        self.mode_options_combobox = ttk.Combobox(root,values= ["cv2.bilateralFilter", "cv2.GaussianBlur",
                                                                "cv2.blur","cv2.medianBlur", "none"],
                                                  textvariable=self.mode_options_var) 
        self.mode_options_combobox.grid(row=1, column=1, padx=10, pady=10)
        self.contrast = tk.Checkbutton(root,text="змінити контраст",
                                       variable=self.contrast_var).grid(row=2, column=2, padx=10, pady=10)
        self.noise = tk.Checkbutton(root,text="зменшити шум",
                                    variable=self.noise_var).grid(row=1, column=2, padx=10, pady=10)
        self.is_serch = tk.Checkbutton(root,text="пошук об'єктів",
                                       variable=self.is_serch_var).grid(row=4, column=2, padx=10, pady=10)
        self.contours = tk.Checkbutton(root,text="виділення контурів",
                                       variable=self.is_contours_var).grid(row=3, column=2, padx=10, pady=10)
        self.label_contour_method = tk.Label(master, text="Метод findContours:")
        self.label_contour_method.grid(row=4, column=0, padx=10, pady=10)
        self.contour_method_combobox = ttk.Combobox(root,values= ["cv2.RETR_EXTERNAL", "cv2.RETR_LIST",
                                                                  "cv2.RETR_TREE"],textvariable=self.contour_method_var) 
        self.contour_method_combobox.grid(row=4, column=1, padx=10, pady=10) 
        self.label_contour_mode = tk.Label(master, text="Режим findContours:")
        self.label_contour_mode.grid(row=5, column=0, padx=10, pady=10)
        self.contour_mode_combobox = ttk.Combobox(root,values= ["cv2.CHAIN_APPROX_NONE", "cv2.CHAIN_APPROX_SIMPLE",
                                                                "cv2.CHAIN_APPROX_TC89_L1", "cv2.CHAIN_APPROX_TC89_KCOS"],
                                                  textvariable=self.contour_mode_var) 

        self.contour_mode_combobox.grid(row=5, column=1, padx=10, pady=10)

        validate_cmd = (root.register(self.validate_input), '%P')
        tk.Label(root, text="----Параметри об'єкта---------").grid(row=6, column=1, padx=2, pady=2)
        tk.Label(root, text="висота MIN:").grid(row=7, column=0,columnspan=1, padx=2, pady=2)
        tk.Entry(root, textvariable=self.seting["minH_var"], validate="key", validatecommand=validate_cmd,
                 width=4).grid(row=7, column=0,columnspan=2,  padx=10, pady=10)    
        tk.Label(root, text="висота MAX:").grid(row=7, column=1,columnspan=1, padx=2, pady=2)
        tk.Entry(root, textvariable=self.seting["maxH_var"], validate="key", validatecommand=validate_cmd,
                 width=4).grid(row=7, column=1,columnspan=2, padx=10, pady=10)

        tk.Label(root, text="ширина MIN:").grid(row=8, column=0,columnspan=1, padx=2, pady=2)
        tk.Entry(root, textvariable=self.seting["minL_var"], validate="key", validatecommand=validate_cmd,
                 width=4).grid(row=8, column=0,columnspan=2, padx=10, pady=10)
        tk.Label(root, text="ширина MAX").grid(row=8, column=1,columnspan=1, padx=2, pady=2)
        tk.Entry(root, textvariable=self.seting["maxL_var"], validate="key", validatecommand=validate_cmd,
                 width=4).grid(row=8, column=1,columnspan=2, padx=10, pady=10)
        tk.Label(root, text="Кількість ліній Min").grid(row=9, column=0,columnspan=1, padx=2, pady=2)
        tk.Entry(root, textvariable=self.seting["minLine"], validate="key", validatecommand=validate_cmd,
                 width=4).grid(row=9, column=0,columnspan=2, padx=10, pady=10)
        tk.Label(root, text="Кількість ліній max").grid(row=9, column=1,columnspan=1, padx=2, pady=2)
        tk.Entry(root, textvariable=self.seting["maxLine"], validate="key", validatecommand=validate_cmd,
                 width=4).grid(row=9, column=1,columnspan=2, padx=10, pady=10)
        tk.Label(root, text="кільність об'єктів:").grid(row=10, column=0, padx=2, pady=2)
        tk.Entry(root, textvariable=self.seting["maxObj_var"], validate="key", validatecommand=validate_cmd,
                 width=2).grid(row=10, column=0,columnspan=2, padx=10, pady=10)
        self.btn_start_processing = tk.Button(master, text="Почати обробку",
                                              command=self.start_processing)
        self.btn_start_processing.grid(row=11, column=2,columnspan=2, pady=10)
        tk.Button(root, text="Save Settings", command=self.save_settings_dialog).grid(row=9,
                                                                                      column=2,columnspan=2, padx=10, pady=5)   
        tk.Button(root, text="Load Settings", command=self.load_settings_dialog).grid(row=10, column=2, padx=10, pady=5)

    def save_settings(self, filename="settings.pkl"):
        all_settings = {key: var.get() for key, var in self.seting.items()}
        all_settings.update({key: var.get() for key, var in self.new_settings.items()})

        with open(filename, 'wb') as file:
            pickle.dump(all_settings, file)

    def load_settings(self, filename="settings.pkl"):
        try:
            with open(filename, 'rb') as file:
                settings = pickle.load(file)
                self.set_settings(settings)
        except FileNotFoundError:
            print(f"Файл {filename} не знайдено. Використовуються значення за замовчуванням.")

    def get_settings(self):
        return {key: var.get() for key, var in self.seting.items()}

    def set_settings(self, settings):
        for key, value in settings.items():
            if key in self.seting:
                self.seting[key].set(value)
            elif key in self.new_settings:
                self.new_settings[key].set(value)

    def save_settings_dialog(self):
        filename = tk.filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
        if filename:
            self.save_settings(filename)

    def load_settings_dialog(self):
        filename = tk.filedialog.askopenfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
        if filename:
            self.load_settings(filename)

    def validate_input(self, value):
        return value.isdigit() or value == ""
    
    def browse_video(self):
        self.path = filedialog.askopenfilename(filetypes=[("Відео файли", "*.png;*.jpeg;")])
        self.entry_path.delete(0, tk.END)
        self.entry_path.insert(0, self.path)
        
    def start_processing(self):
        if self.path:
            
            kernel = np.ones((5,5),np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.namedWindow("Settings",cv2.WINDOW_NORMAL)
            screen_width, screen_height = 1920, 1080
            cv2.resizeWindow("Settings", screen_width // 3, screen_height)

            if (self.mode_options_var.get() =="cv2.bilateralFilter" ):
                cv2.createTrackbar("diameterBilateralFilter","Settings",self.seting["diameterBilateralFilter"].get(),30,nothing)
                cv2.createTrackbar("sigmaColor","Settings",self.seting["sigmaColor"].get(),150,nothing)
                cv2.createTrackbar("sigmaSpace","Settings",self.seting["sigmaSpace"].get(),150,nothing)
            elif (self.mode_options_var.get() =="cv2.GaussianBlur" ):
                cv2.createTrackbar("Gaussian Kernel Size Height","Settings",self.seting["GaussianKernelSizeHeight"].get(),30,nothing)
                cv2.createTrackbar("Gaussian Kernel Size Width","Settings",self.seting["GaussianKernelSizeWidth"].get(),30,nothing)
                cv2.createTrackbar("Kernel Standard Deviation XY", "Settings",self.seting["KernelStandardDeviationXY"].get(), 10, nothing)
            elif (self.mode_options_var.get() =="cv2.blur" ):
                cv2.createTrackbar("Kernel Size Height","Settings",self.seting["KernelSizeHeight"].get(),30,nothing)
                cv2.createTrackbar("Kernel Size Width","Settings",self.seting["KernelSizeWidth"].get(),30,nothing)
            elif (self.mode_options_var.get() =="cv2.medianBlur" ):
                cv2.createTrackbar("Kernel Size","Settings",self.seting["KernelSize"].get(),30,nothing)
            elif (self.mode_options_var.get() =="none" ):
                print("немає фільтра")
            
            if self.contrast_var.get()==1:        
                cv2.createTrackbar("clipLimit","Settings",self.seting["clipLimit"].get(),200,nothing)
                cv2.createTrackbar("tileGridSizeH","Settings",self.seting["tileGridSizeH"].get(),30,nothing)
                cv2.createTrackbar("tileGridSizeW","Settings",self.seting["tileGridSizeW"].get(),30,nothing)
                cv2.createTrackbar("isL","Settings",0,1,nothing)
                cv2.createTrackbar("isA","Settings",0,1,nothing)
                cv2.createTrackbar("isB","Settings",0,1,nothing)
                
            if (self.method_options_var.get()=="opening" or self.method_options_var.get()=="closing"):
                cv2.createTrackbar("hue","Settings",self.seting["hue"].get(),180,nothing)
                cv2.createTrackbar("saturation","Settings",self.seting["saturation"].get(),255,nothing)
                cv2.createTrackbar("value","Settings",self.seting["value"].get(),255,nothing)
                cv2.createTrackbar("hueL","Settings",self.seting["hueL"].get(),180,nothing)
                cv2.createTrackbar("saturationL","Settings",self.seting["saturationL"].get(),255,nothing)
                cv2.createTrackbar("valueL","Settings",self.seting["valueL"].get(),255,nothing)
            elif self.method_options_var.get()=="none":
                print("немає фільтра")
            else:
                cv2.createTrackbar("threshold","Settings",self.seting["threshold"].get(),255,nothing)
                cv2.createTrackbar("threshold max","Settings",self.seting["threshold_max"].get(),255,nothing)

           

            while True:
                start = time.time()
                self.path = os.path.abspath(self.path)
                try:
                    image = Image.open(self.path)
                    self.img = np.array(image)
                    print("Файл успішно зчитано за допомогою Pillow.")
                except Exception as e:
                    print(f"Помилка при зчитуванні зображення за допомогою Pillow: {e}")
                    
                self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)                 
                frame =self.img
                final_video = self.img
                #print("Еталоне зображення кількісна оцінка розмитя "+str(blur_score(frame)))
                #print("Еталоне зображення кількісна оцінка контрасту "+str(contrast_score(frame)))
                if self.noise_var.get():
                    if (self.mode_options_var.get() =="cv2.bilateralFilter" ):
                        diameterBilateralFilter= cv2.getTrackbarPos("diameterBilateralFilter",
                                                                    "Settings")
                        sigmaColor =cv2.getTrackbarPos("sigmaColor","Settings")
                        sigmaSpace=cv2.getTrackbarPos("sigmaSpace","Settings")   
                        if diameterBilateralFilter==0:
                            diameterBilateralFilter=1
                        self.seting["diameterBilateralFilter"].set(diameterBilateralFilter)  
                        self.seting["sigmaColor"].set(sigmaColor)  
                        self.seting["sigmaSpace"].set(sigmaSpace)
                        frame = cv2.bilateralFilter(frame,diameterBilateralFilter
                                                    ,sigmaColor,sigmaSpace)
                        cv2.imshow("bilateralFilter", frame)
                    elif (self.mode_options_var.get() =="cv2.GaussianBlur" ):
                        GaussianKernelSizeHeight= cv2.getTrackbarPos("Gaussian Kernel Size Height","Settings")
                        GaussianKernelSizeWidth =cv2.getTrackbarPos("Gaussian Kernel Size Width","Settings")
                        KernelStandardDeviationXY = cv2.getTrackbarPos("Kernel Standard Deviation XY", "Settings")
                        if GaussianKernelSizeHeight%2==0:
                            GaussianKernelSizeHeight=GaussianKernelSizeHeight+1
                        if GaussianKernelSizeWidth % 2==0:
                            GaussianKernelSizeWidth=GaussianKernelSizeWidth+1
                        self.seting["GaussianKernelSizeHeight"].set(GaussianKernelSizeHeight)   
                        self.seting["GaussianKernelSizeWidth"].set(GaussianKernelSizeWidth)
                        self.seting["KernelStandardDeviationXY"].set(KernelStandardDeviationXY)  
                        frame = cv2.GaussianBlur(frame, (GaussianKernelSizeHeight,
                                                         GaussianKernelSizeWidth), KernelStandardDeviationXY)
                        cv2.imshow("GaussianBlur", frame)
                    elif (self.mode_options_var.get() =="cv2.blur"):
                        KernelSizeHeight= cv2.getTrackbarPos("Kernel Size Height","Settings")
                        KernelSizeWidth =cv2.getTrackbarPos("Kernel Size Width","Settings")
                        if KernelSizeHeight%2==0:
                            KernelSizeHeight=KernelSizeHeight+1
                        if KernelSizeWidth % 2==0:
                            KernelSizeWidth=KernelSizeWidth+1
                        self.seting["KernelSizeHeight"].set(KernelSizeHeight)  
                        self.seting["KernelSizeWidth"].set(KernelSizeWidth)  
                        frame = cv2.blur(frame, (KernelSizeHeight, KernelSizeWidth))
                        cv2.imshow("SimpleBlur", frame)
                    elif (self.mode_options_var.get() =="cv2.medianBlur"):
                        KernelSize= cv2.getTrackbarPos("Kernel Size","Settings")
                        if KernelSize%2==0:
                            KernelSize=KernelSize+1
                        self.seting["KernelSize"].set(KernelSize)
                        frame= cv2.medianBlur(frame, KernelSize)
                        cv2.imshow("medianBlur", frame)
                    else:
                        print("невірно вказаний метод усунення шуму")
                    print("кількісна оцінка розмитя "+str(blur_score(frame)))
                    #print ("розмитя PSNR: "+str(getPSNR(self.img ,frame)))
                    print("розмитя  ssim: " + str(limit_decimal_places(getMSSISM(self.img ,frame))))  
                if self.contrast_var.get()==1:

                    clipLimit= cv2.getTrackbarPos("clipLimit","Settings")
                    tileGridSizeH=cv2.getTrackbarPos("tileGridSizeH", "Settings")
                    tileGridSizeW=cv2.getTrackbarPos("tileGridSizeW", "Settings")
                    self.seting["clipLimit"].set(clipLimit)
                    self.seting["tileGridSizeH"].set(tileGridSizeH)
                    self.seting["tileGridSizeW"].set(tileGridSizeW)
                    clahe = cv2.createCLAHE( clipLimit/ 10,tileGridSize = (tileGridSizeH,tileGridSizeW))
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  
                    l, a, b = cv2.split(lab) 
                    l2,a2,b2=l,a,b
                    if cv2.getTrackbarPos("isL","Settings")==1 :
                        l2 = clahe.apply(l) 
                    if cv2.getTrackbarPos("isA","Settings")==1 :
                        a2 = clahe.apply(a)
                    if cv2.getTrackbarPos("isB","Settings")==1 :
                        b2 = clahe.apply(b) 
                    lab = cv2.merge((l2,a2,b2))  
                    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) 
                    cv2.imshow('Increased contrast', frame)
                    print("кількісна оцінка контрасту "+str(contrast_score(frame)))
                   # print ("контраст PSNR"+str(getPSNR(self.img ,frame)))
                    print("контраст  ssim: " + str(limit_decimal_places(getMSSISM(self.img ,frame))))   
                if self.is_contours_var.get():
                    frame_to_search_for_contours = frame    
                    threshold = 127  
                    if (self.method_options_var.get() =="cv2.THRESH_BINARY" or
                        self.method_options_var.get() == "cv2.THRESH_BINARY_INV"  or
                        self.method_options_var.get() ==  "cv2.THRESH_TRUNC" or
                        self.method_options_var.get() == "cv2.THRESH_TOZERO" or
                        self.method_options_var.get() == "cv2.THRESH_TOZERO_INV"):
                        
                        frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        threshold = cv2.getTrackbarPos("threshold","Settings")
                        threshold_max = cv2.getTrackbarPos("threshold max","Settings")
                        self.seting["threshold"].set(threshold)  
                        self.seting["threshold_max"].set(threshold_max)  
                        threshold_type = eval(self.method_options_var.get())
                        ret, frame_to_search_for_contours = cv2.threshold(frame,threshold,
                                                                          threshold_max,
                                                                          threshold_type)
                        cv2.imshow("threshold", frame_to_search_for_contours)
                    elif (self.method_options_var.get()=="opening" or self.method_options_var.get()=="closing"):
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        hue=cv2.getTrackbarPos("hue","Settings")
                        saturation=cv2.getTrackbarPos("saturation","Settings")
                        value=cv2.getTrackbarPos("value","Settings")
                        hueL=cv2.getTrackbarPos("hueL","Settings")
                        saturationL=cv2.getTrackbarPos("saturationL","Settings")
                        valueL=cv2.getTrackbarPos("valueL","Settings")
                        self.seting["hue"].set(hue)  
                        self.seting["saturation"].set(saturation)  
                        self.seting["value"].set(value)  
                        self.seting["hueL"].set(hueL)  
                        self.seting["saturationL"].set(saturationL)  
                        self.seting["valueL"].set(valueL)  
                        lower = np.array([hueL,saturationL,valueL])
                        upper = np.array([hue,saturation,value])
                        mask = cv2.inRange(hsv,lower,upper)
                        res =cv2.bitwise_and(frame,frame,mask=mask)
                        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernel)
                        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                        if self.method_options_var.get()=="opening":
                            frame_to_search_for_contours = opening
                            cv2.imshow("opening", res)
                        if self.method_options_var.get()=="closing":
                            frame_to_search_for_contours = closing
                            cv2.imshow("closing", res)
                    elif (self.method_options_var.get()=="cv2.Canny_Edge_Detection"):
                        frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        threshold = cv2.getTrackbarPos("threshold","Settings")
                        threshold_max = cv2.getTrackbarPos("threshold max","Settings")
                        self.seting["threshold"].set(threshold)  
                        self.seting["threshold_max"].set(threshold_max)  
                        frame_to_search_for_contours = cv2.Canny(frame,threshold,threshold_max)
                        #frame_to_search_for_contours=cv2.dilate(frame_to_search_for_contours, kernel, iterations=1)
                        cv2.imshow("Canny_Edge_Detection", frame_to_search_for_contours)
                    else:
                        print("невірно вказано метод метод виділення контурів")
                    #print("виділення контурів ssim: " + str(getMSSISM(frame ,frame_to_search_for_contours)))   

                    
                if self.is_serch_var.get() and not self.is_contours_var.get():
                    print("ввімкніть виділення контурів")
                elif self.is_serch_var.get():
                    minH=self.seting["minH_var"].get()
                    maxH=self.seting["maxH_var"].get()
                    minL=self.seting["minL_var"].get()
                    maxL=self.seting["maxL_var"].get()
                    maxObj=self.seting["maxObj_var"].get()
                    contour_method = eval(self.contour_method_var.get())
                    contour_mode = eval(self.contour_mode_var.get())
                    contours, h = cv2.findContours(frame_to_search_for_contours,contour_method, contour_mode)
                    contours = sorted(contours, key = cv2.contourArea, reverse = True)
                    counter = 1
                    minLine=self.seting["minLine"].get()
                    maxLine=self.seting["maxLine"].get()
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        x,y,w,h=cv2.boundingRect(contour)
                        cv2.drawContours(frame_to_search_for_contours,contour,-1,(200,200,0),3)
                        p=cv2.arcLength(contour,True)
                        num=cv2.approxPolyDP(contour,0.03*p,True)
                        print(len(num))
                        if  len(num)< maxLine and len(num)> minLine and area > minH*minL and area < maxH*maxL :
                            final_video = cv2.rectangle(final_video,(x,y),(x+w,y+h),(0,255,0),2)
                            final_video = cv2.rectangle(final_video,(x,y),(x+60,y-15),(0,0,0),-1)
                            cv2.putText(final_video,(str(counter)+"Obj" + str(len(num))),
                                        (x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
                            counter=counter+1
                            if counter > maxObj:
                                break        
                    if (self.method_options_var.get()=="opening" or self.method_options_var.get()=="closing"):
                        cv2.imshow("mask", mask)


                        
                    cv2.imshow("final_video", final_video)
                end = time.time() - start
                if end < 1/ self.seting["maxFPS_var"].get():
                    time.sleep(1/self.seting["maxFPS_var"].get()-end)
                self.frameTime=end
                print("час обробки: "+str(self.frameTime))
                if cv2.waitKey(30) & 0xFF == 27:  # 27 - код клавіші Esc
                    break
            cv2.destroyAllWindows()
            print("  ssim: " + str(getMSSISM(frame ,frame)))
        else:
            print("Виберіть відео перед початком відслідковування")

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalAnalysisToolkit(root)
    root.mainloop()
