import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from bs4 import BeautifulSoup
import pandas

ch = 17 # Number of channels
Fs = 500.0 # Sampling frequency
# Names of channels.
ch_names = ['P3','Cz','O2','P4','C3','O1','Pz','C4','ACC_x','ACC_y<',
            'ACC_z','RSSI','Dongle','Head',
            'PC','SCounter','TSS']

# Extracting the pressed key, the right key and the letters on target and distractor.
with open('CV_32_flankery.csv') as csv_file:
    data = pandas.read_csv(csv_file)
num_trials = len(data)-1
keys = np.zeros((num_trials,5),dtype=str)   # row 0 - correct, row 1 - chosen, row 2 - r/f
                                            # row 3  - side, row 4 - center               
for event in range(num_trials):
    keys[event,0]=data['CORRECT'][event+1]
    keys[event,1]=data['key_resp_trial.keys'][event+1]
    keys[event,2]=data['key_resp_trial.corr'][event+1]
    keys[event,3]=data['PARAM_G'][event+1]
    keys[event,4]=data['PARAM_S'][event+1]

# Extracting for each trial start and response times.
times = np.zeros((num_trials,2),dtype=float) # row 0 - starts, row 1 - responses
soup = BeautifulSoup(open('CV_32_flankery.tag'), 'lxml')
event_num = 0
for tag in soup.find_all('tag'):
    if tag['name'] == 'odpowiedz':
        times[event_num,0] = float(tag['position'])
        times[event_num,1] = float(tag['position'])+float(tag['length'])
        event_num+=1      

# Opening file with the signal.
with open('CV_32_flankery.raw') as f:
    s=np.fromfile(f, dtype='float32')
    s=np.reshape(s, (int(len(s)/ch),int(ch)))
s=s.T 

# Lengths of stimuli-response.
len_min = int(min(times[:,1]-times[:,0])*Fs)
len_max = int(max(times[:,1]-times[:,0])*Fs)
len_mean = int(np.mean(times[:,1]-times[:,0])*Fs)
len_frag = len_min

# Filtering of the signal - bandpass 8.5-9.5 Hz and 11.5-12.5 Hz.
[b9,a9]=ss.butter(3,[8.5/(Fs/2), 9.5/(Fs/2)], btype="bandpass")
[b12,a12]=ss.butter(3,[11.5/(Fs/2), 12.5/(Fs/2)], btype="bandpass")
sf9 = ss.filtfilt(b9,a9,s)
sf12= ss.filtfilt(b12,a12,s)

# Selecting trial fragments and inserting them into matrix trials_..., then calculating instantaneous
# power with Hilbert transform into sp_..., then subtracting the baseline power spect_mean....
channel=5

trials_same = np.zeros((2,125,len_frag))            # matrices for trials
trials_diff = np.zeros((2,62,len_frag))

sp_same = np.zeros((2,125,len_frag))                # matrices for the instantenous powers of trials
sp_diff = np.zeros((2,62,len_frag))

num_same,num_diff = 0,0                             # count of the trials
spect_mean_9, spect_mean_12 = 0, 0                  # baseline power (0.2 s before stimulus)

for event in range(num_trials):
    trial_9 = sf9[channel][(int((times[event,0])*Fs)):(int(times[event,0]*Fs)+len_frag)]
    trial_12 = sf12[channel][(int((times[event,0])*Fs)):(int(times[event,0]*Fs)+len_frag)]
    if (keys[event,2] == '1') and (keys[event,3]==keys[event,4]): 
        trials_same[0,num_same,:]=trial_9
        trials_same[1,num_same,:]=trial_12
        sp_same[0,num_same,:]= np.abs(ss.hilbert(trial_9))**2
        sp_same[1,num_same,:]=np.abs(ss.hilbert(trial_12))**2
        spect_mean_9 += sf9[channel][int(times[event,0]*Fs - (0.2)*Fs)]**2
        spect_mean_12 += sf12[channel][int(times[event,0]*Fs - (0.2)*Fs)]**2
        num_same +=1
    elif (keys[event,2] == '1') and (keys[event,3]!=keys[event,4]):
        trials_diff[0,num_diff,:]=trial_9
        trials_diff[1,num_diff,:]=trial_12
        sp_diff[0,num_diff,:]= np.abs(ss.hilbert(trial_9))**2
        sp_diff[1,num_diff,:]=np.abs(ss.hilbert(trial_12))**2
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
plt.legend(['congruent', 'incongruent'])
plt.ylim(-150,1000)
plt.title('12 Hz')
plt.grid('on')

fig.text(0.5, 1.01,'Instantaneous power\n channel '+ch_names[channel], ha='center', fontsize=15)
fig.text(0.5, -0.01, 'Time [s]', ha='center', fontsize=12)
fig.text(-0.01, 0.5,'Amplitude' , va='center', rotation='vertical',fontsize=12)

plt.tight_layout()
plt.show()