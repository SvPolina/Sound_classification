import librosa
import librosa.display
import IPython.display as ipd
import collections
import csv
import os
import numpy as np
import pandas as pd
import operator
from sklearn.utils import resample
from collections import Counter
import sys
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt 

import warnings
warnings.simplefilter("ignore")


# Constuct data file with possible main features.
def form_data_file(path,output_f_name): 
    df_header=[
               "filename","chroma_stft","chroma_stft_stdev",
               "spectral_centroid","spectral_centroid_stdev",
               "spectral_bandwidth","spectral_bandwidth_stdev",
               "rolloff","rolloff_stdev",
               "zero_crossing_rate","zero_crossing_rate_stdev",
               "pitch","pitch_stdev"
              ]
    for i in range(1, 21):
        df_header.append('mfcc '+str(i))
    file = open(path+'/'+output_f_name, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(df_header) 
        lines=[]
    for f_name in os.listdir(path):
        try:
            line=[]
            sound = path+'/'+f_name
            y, sr = librosa.load(sound, mono=True, duration=30)           
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)            
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            line = [
                    f_name,np.mean(chroma_stft),np.std(chroma_stft), 
                    np.mean(spec_cent),np.std(spec_cent), 
                    np.mean(spec_bw),np.std(spec_bw), 
                    np.mean(rolloff),np.std(rolloff), 
                    np.mean(zcr),np.std(zcr),
                    np.mean(pitches),np.std(pitches)
                    ]        
            for ft in mfcc:
                line.append(np.mean(ft))
            lines.append(line) 
            file = open(path+'/'+output_f_name, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(line)
        except:
            print ("Unexpected error:", sys.exc_info()[0])
            print ("failed loading:", f_name)
            pass    
        
def fft_plot(audio,samp_rate):
    n=int(len(audio))
    T=1/samp_rate
    yf=scipy.fft(audio)
    xf=np.linspace(0,int(1.0//(2.0*T)),int(n//2))
    fig,ax=plt.subplots()
    ax.plot(xf,2.0/n*np.abs(yf[:n//2]))
    plt.grid()
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    return plt.show()        