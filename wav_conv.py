#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wave
import numpy as np
import pandas as pd
import os
import soundfile as sf
import matplotlib.pyplot as plt
from sympy.ntheory import factorint
import librosa
fs2read = 22050
audiolength = 30
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
#%%

# with wave.open("genres_original/blues/blues.00000.wav") as w:
#     rate = w.getframerate()
#     sampwidth = w.getsampwidth()
#     nch = w.getnchannels()
#     frames = w.readframes(fs2read)

# deserialized_bytes = np.frombuffer(frames,dtype=np.int8)

    #dataframe = np.zeros((1,441000))
def createnewdata(outtype=np.float64,sr=2756):
    #1378
    i = 0
    dataframe = pd.DataFrame()
    for root, dirs, files in os.walk("genres_original", topdown=False):
            #for name in files:
            #    print(os.path.join(root, name))
            # print(files)
            for song in files:
                print(song)
                data, samplerate = sf.read(os.path.join(root,song),always_2d=True,dtype=np.float64)
                #print(data.shape)
                n = len(data)
                D = int(samplerate/sr)
                # print(samplerate)
                single_channel = np.array([data[i][0] for i in range(n)])
                #print(single_channel.shape)
                downsampled_data = single_channel[::D]
                maxbuffersize = sr * audiolength
                downsampled_data = downsampled_data[:maxbuffersize]
                split_factor = list(factorint(maxbuffersize))[-1]
                split = np.array_split(downsampled_data,split_factor)
                #print(downsampled_data.shape)
                # with wave.open(os.path.join(root,song)) as w:
                #     print(os.path.join(root,song))
                #     #print(files[0].split('.')[0])
                #     # print(w.getframerate())
                #     # print(w.getsampwidth())
                #     # print(w.getnchannels())
                #     frame = np.frombuffer(w.readframes(fs2read),dtype=np.uint8).reshape(1,w.getsampwidth()*fs2read)
                #downsampled_data.reshape(1,689)
                genre = song.split('.')[0]
                #pdframe = pd.concat([pd.DataFrame(downsampled_data),pd.Series(genre)],axis=0,ignore_index=True).T
                X = pd.DataFrame()
                for j in split:
                    temp = pd.DataFrame(j).T
                    temp.insert(0,"label",pd.Series(genre))
                    X = pd.concat([X,temp],axis=0,ignore_index=True)
                # X = pd.DataFrame(downsampled_data).T
                # X.insert(0,"label",pd.Series(genre))
                #print(pdframe.shape)
                #     #pdframe = pd.DataFrame(frame)
                dataframe = pd.concat([dataframe,X],axis=0,ignore_index=True)
                i += 1
                print(i)
                
    dataframe.fillna(0, inplace=True)
    dataframe.to_csv("/data/music_data/compressed_wavs_2756_split.csv",header=False, index=False)
    return dataframe
#%%

def soundfiletoy():
    data, samplerate = sf.read("/home/marco/NN-projects/music_class/genres_original/jazz/jazz.00001.wav")
    
    n = len(data) #the length of the arrays contained in data
    Fs = samplerate #the sample rate# Working with stereo audio, there are two channels in the audio data.
    
    ch1_Fourier = np.fft.fft(data) #performing Fast Fourier Transform
    abs_ch1_Fourier = np.absolute(ch1_Fourier[:n//2]) #the spectrum
    
    plt.plot(np.linspace(0, Fs / 2, n//2), abs_ch1_Fourier)
    plt.ylabel('Spectrum')
    plt.xlabel('$f$ (Hz)')
    plt.show()
    
    eps = 1e-5
    # Boolean array where each value indicates whether we keep the corresponding frequency
    frequenciesToRemove = (1 - eps) * np.sum(abs_ch1_Fourier) < np.cumsum(abs_ch1_Fourier)
    # The frequency for which we cut the spectrum
    f0 = (len(frequenciesToRemove) - np.sum(frequenciesToRemove) )* (Fs / 2) / (n / 2)
    
    print("f0 : {} Hz".format(int(f0)))# Displaying the spectrum with a vertical line for f0
    
    plt.axvline(f0, color='r')
    plt.plot(np.linspace(0, Fs / 2, n//2), abs_ch1_Fourier)
    plt.ylabel('Spectrum')
    plt.xlabel('$f$ (Hz)')
    plt.show()
    
    #%%
    
    #First we define the names of the output files
    wavCompressedFile = "audio_compressed4.wav"
    #Then we define the downsampling factor
    D = int(Fs / f0)*16
    print("Downsampling factor : {}".format(D))
    new_data = data[::D] #getting the downsampled data#Writing the new data into a wav file
    sf.write(wavCompressedFile, new_data, int(Fs / D), 'PCM_16')
    
    
#%%
def mfcc(downsampling_factor = 1):
    dataset = []
    
    for root, dirs, files in os.walk("genres_original", topdown=False):
            #for name in files:
            #    print(os.path.join(root, name))
            # print(files)
            for song in files:
                print(song)
                data, samplerate = sf.read(os.path.join(root,song),always_2d=False,dtype=np.float64)
                #print(data.shape)
                n = len(data)
                D = int(downsampling_factor)
                final_sr = int(samplerate/downsampling_factor)
                downsampled_data = data[::D]
                maxbuffersize = final_sr * audiolength
                downsampled_data = downsampled_data[:maxbuffersize]
                factors = list(factorint(maxbuffersize))
                split_factor = factors[-1]
                if split_factor < 10 and 2 in factors: split_factor = split_factor * 2
                print(split_factor)
                split = np.array_split(downsampled_data,split_factor)
                genre = genres.index(song.split('.')[0])
                
                for j in split:
                    
                    temp = librosa.feature.mfcc(y=j,sr=final_sr)
                    templabel = np.full((temp.shape[0],1),genre)
                    temp = np.append(temp,templabel,axis=1)
                    dataset.append(temp)
                
    dataset = np.array(dataset)
    np.save("/data/music_data/mfcc_wavs_split.npy",dataset)
    return dataset

#%%
#data, samplerate = sf.read("/home/marco/NN-projects/music_class/genres_original/jazz/jazz.00001.wav")
#y, sr = librosa.load("/home/marco/NN-projects/music_class/genres_original/jazz/jazz.00001.wav")
#dataset = pd.read_csv("/data/music_data/compressed_wavs_2756_split.csv",header=None,index_col=None)
#mf = librosa.feature.mfcc(y=data,sr=sr)
