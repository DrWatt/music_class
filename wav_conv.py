#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wave
import numpy as np
import pandas as pd
import os

fs2read = 220500


# with wave.open("genres_original/blues/blues.00000.wav") as w:
#     rate = w.getframerate()
#     sampwidth = w.getsampwidth()
#     nch = w.getnchannels()
#     frames = w.readframes(fs2read)

# deserialized_bytes = np.frombuffer(frames,dtype=np.int8)

#dataframe = np.zeros((1,441000))
dataframe = pd.DataFrame()
for root, dirs, files in os.walk("genres_original", topdown=False):
        #for name in files:
        #    print(os.path.join(root, name))
        # print(files)
        for song in files:
            try:
                with wave.open(os.path.join(root,song)) as w:
                    print(os.path.join(root,song))
                    #print(files[0].split('.')[0])
                    # print(w.getframerate())
                    # print(w.getsampwidth())
                    # print(w.getnchannels())
                    frame = np.frombuffer(w.readframes(fs2read),dtype=np.uint8).reshape(1,w.getsampwidth()*fs2read)
                    pdframe = pd.concat([pd.DataFrame(frame),pd.Series(song.split('.')[0])],axis=1)
                    #pdframe = pd.DataFrame(frame)
                    dataframe = pd.concat([dataframe,pdframe],axis=0)
            except:
                pass

dataframe.to_csv("/data/music_data/10secwavtoint.csv")