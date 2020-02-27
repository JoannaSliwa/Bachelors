import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from bs4 import BeautifulSoup
import pandas

ch = 17 # Number of channels.
Fs = 500.0 # Sampling frequency
# Names of the channels.
ch_names = ['P3','Cz','O2','P4','C3','O1','Pz','C4','ACC_x','ACC_y<',
            'ACC_z','RSSI','Dongle','Head',
            'PC','SCounter','TSS']

# Extracting for each trial start and response times.
times = np.zeros((204,2),dtype=float) # row 0 - starts, row 1 - responses
soup = BeautifulSoup(open('CV_32_flankery.tag'), 'lxml')
event_num = 0
for tag in soup.find_all('tag'):
    if tag['name'] == 'odpowiedz':
        times[event_num,0] = float(tag['position'])
        times[event_num,1] = float(tag['position'])+float(tag['length'])
        event_num+=1
        
# Extracting chosen key and the right key.
with open('CV_32_flankery.csv') as csv_file:
    data = pandas.read_csv(csv_file)
keys = np.zeros((204,5),dtype=str) # row 0 - correct, row 1 - chosen, row 2 - r/f
                                   # row 3  - side, row 4 - center
for event in range(204):
    keys[event,0]=data['CORRECT'][event+1]
    keys[event,1]=data['key_resp_trial.keys'][event+1]
    keys[event,2]=data['key_resp_trial.corr'][event+1]
    keys[event,3]=data['PARAM_G'][event+1]
    keys[event,4]=data['PARAM_S'][event+1]

# Opening file.
with open('CV_32_flankery.raw') as f:
    s=np.fromfile(f, dtype='float32')
    s=np.reshape(s, (int(len(s)/ch),int(ch)))
s=s.T 

# Lengths of stimuli-response.
len_min = int(min(times[:,1]-times[:,0])*Fs)
len_max = int(max(times[:,1]-times[:,0])*Fs)
len_mean = int(np.mean(times[:,1]-times[:,0])*Fs)
len_frag = len_min


# Retrieveing trial fragments - 5sec~response. # 0 - 9Hz, 1- 12 Hz
trials_same = np.zeros((2,125,len_frag))
trials_diff = np.zeros((2,62,len_frag))

sp_same = np.zeros((2,125,len_frag))
sp_diff = np.zeros((2,62,len_frag))

num_same,num_diff = 0,0
spect_mean_9 = 0
spect_mean_12 = 0

[b9,a9]=ss.butter(3,[8.5/(Fs/2), 9.5/(Fs/2)], btype="bandpass")
[b12,a12]=ss.butter(3,[11.5/(Fs/2), 12.5/(Fs/2)], btype="bandpass")
sf9 = ss.filtfilt(b9,a9,s)
sf12= ss.filtfilt(b12,a12,s)

channel = 5
# Selecting trial fragments and inserting into matrix, then calculating instantaneous power with
# Hilbert transform. Then subtracting the mean power.
for event in range(204):
    trial_9 = sf9[channel][(int((times[event,0])*Fs)):(int(times[event,0]*Fs)+len_frag)]
    trial_12 = sf12[channel][(int((times[event,0])*Fs)):(int(times[event,0]*Fs)+len_frag)]
    if (keys[event,2] == '1') and (keys[event,3]==keys[event,4]): 
        trials_same[0,num_same,:]=trial_9
        trials_same[1,num_same,:]=trial_12
        sp_same[0,num_same,:]= abs(ss.hilbert(trial_9))**2
        sp_same[1,num_same,:]=abs(ss.hilbert(trial_12))**2
        spect_mean_9 += sf9[channel][int(times[event,0]*Fs - (0.2)*Fs)]**2
        spect_mean_12 += sf12[channel][int(times[event,0]*Fs - (0.2)*Fs)]**2
        num_same +=1
    elif (keys[event,2] == '1') and (keys[event,3]!=keys[event,4]):
        trials_diff[0,num_diff,:]=trial_9
        trials_diff[1,num_diff,:]=trial_12
        sp_diff[0,num_diff,:]= abs(ss.hilbert(trial_9))**2
        sp_diff[1,num_diff,:]=abs(ss.hilbert(trial_12))**2
        spect_mean_9 += sf9[channel][int(times[event,0]*Fs - (0.2)*Fs)]**2
        spect_mean_12 += sf12[channel][int(times[event,0]*Fs - (0.2)*Fs)]**2
        num_diff +=1
        

sp_s_9 = np.mean(sp_same[0], axis=0)- spect_mean_9/(num_same+num_diff)
sp_s_12 = np.mean(sp_same[1], axis=0) -spect_mean_12/(num_same+num_diff)
sp_d_9 = np.mean(sp_diff[0], axis=0)- spect_mean_9/(num_same+num_diff)
sp_d_12 = np.mean(sp_diff[1], axis=0) -spect_mean_12/(num_same+num_diff)


# Ploting the instantaneous power of fragments.
fig = plt.figure(figsize=(8,3))

plt.subplot(1,2,1)
plt.plot(np.arange(0,len_frag/Fs, 1/Fs),sp_s_9)
plt.plot(np.arange(0,len_frag/Fs, 1/Fs),sp_d_9)
plt.ylim(-150,1000)
plt.title('9 Hz')
plt.grid('on')
plt.subplot(1,2,2)
plt.plot(np.arange(0,len_frag/Fs, 1/Fs),sp_s_12)
plt.plot(np.arange(0,len_frag/Fs, 1/Fs),sp_d_12)
plt.legend(['con', 'incon'])
plt.ylim(-150,1000)
plt.title('12 Hz')
plt.grid('on')

fig.text(0.5, 1.01,'Instantaneous power\n channel '+ch_names[channel], ha='center', fontsize=15)
fig.text(0.5, -0.01, 'Time [s]', ha='center', fontsize=12)
fig.text(-0.01, 0.5,'Amplitude' , va='center', rotation='vertical',fontsize=12)

plt.tight_layout()
plt.show()


