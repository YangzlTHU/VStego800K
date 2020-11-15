#-*-coding:utf-8 -*-
import os
import numpy as np

path = "E:/wav_path/"
pcm_path = "E:/pcm_path/"

for file in os.listdir(path):
	print(file)
	f = 0
	data = 0
	f = open(os.path.join(path,file))
	f.seek(0)
	f.read(44)
	data = np.fromfile(f, dtype=np.int16)
	dataname = file.rstrip('wav')+'pcm'
	data.tofile(os.path.join(pcm_path, dataname))