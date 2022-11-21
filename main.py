import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import sklearn

#读取数据
sample_data_folder = 'E:/pan.baidu/MNE-sample-data'
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_raw.fif')

#读取事件
events_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_raw-eve.fif')
events = mne.read_events(events_file)
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=False)

#挑选EEG通道
raw.pick(['eeg']).load_data()

raw.plot(duration=4, show_scrollbars=False) #打印

print(raw.info)

raw.plot_psd(fmax=180)
raw.plot(duration=5, n_channels=30)

#可视化电极分布
raw.plot_sensors(show_names=True)
fig = raw.plot_sensors('3d')

#滤波与重采样
raw.resample(sfreq=256)
raw.filter(l_freq=0.1, h_freq=70)

#ICA降噪
ica_raw = raw.copy()
#最好输入有多少个通道，分解的独立成分就选择多少，这样可以避免剔除的成分中包含重要信息
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(ica_raw)
#可视化ICA分解的成分，进一步挑选要删除的成分从而实现降噪
ica.plot_components()
ica.plot_sources(raw)

ica.exclude = [0,3,4,5,9,12,15,16,19] #选择要剔除的成分
ica.apply(ica_raw)
ica_raw.plot(duration=4, show_scrollbars=False)

#可视化完整的PSD
raw.plot_psd(fmax=70)
#分频段可视化
raw.plot_psd(fmin=0, fmax=4)
raw.plot_psd(fmin=4, fmax=8)
raw.plot_psd(fmin=8, fmax=12)
raw.plot_psd(fmin=12, fmax=30)
raw.plot_psd(fmin=30, fmax=45)

#等分数据，转为epoch，进行ERP分析
event_dict = {'auditory/left' : 1, 'auditory/right' : 2, 'visual/left' : 3,
              'visual/right' : 4, 'face' : 5, 'buttonpress' : 32}
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.3, tmax=0.7,
                    preload=True)
fig = epochs.plot(events=events)
#一种刺激会做多次，得到多个trials，平均多个trials的结果，可以把噪声平均掉（假设噪声均值为0)，从而得到ERP
l_aud = epochs['visual/left'].average()
figl = l_aud.plot(spatial_colors=True)
l_aud.plot_topomap(times=[-0.2, 0.1, 0.4], average=0.05)

#针对epoch进行频域分析
l_aud.plot_psd()
l_aud.plot_topomap()