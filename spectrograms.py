import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from bs4 import BeautifulSoup
import pandas

def plotingEEG(s):
    # Plots a subplot for 8 (EEG sigal) out of 17 channels with time on x-axis.
    fig, ax = plt.subplots(4, 2, figsize=(7,7), sharex=True, sharey=True)
    for i in range(4):
        for j in range(2):
            t=np.arange(0,len(s[0])/500,1/500)
            ax[i,j].plot(t,s[j*4+i])
            ax[i,j].grid(alpha=0.5)
            ax[i,j].set_title('Channel '+str(ch_names[j*4+i]))
    fig.text(0.5, -0.01, 'Time [s]', ha='center', fontsize=15)
    fig.text(-0.01, 0.5,'Amplitude [µV]' , va='center', rotation='vertical',fontsize=15)
    plt.ylim(-2000,2000)
    plt.tight_layout()
    plt.show()
    
def charkterystyki(a,b,f,T,Fs):
    import pylab as py
    from  scipy.signal import freqz, group_delay #funkcja obliczająca funkcję systemu
    from  scipy.signal import firwin, firwin2     # funkcje do projektowania filtrów FIR
    from  scipy.signal import butter, buttord     # funkcje do projektowania filtrów  
    from  scipy.signal import cheby1, cheb1ord    # funkcje do projektowania filtrów 
    from  scipy.signal import cheby2, cheb2ord    # funkcje do projektowania filtrów 
    from  scipy.signal import ellip, ellipord     # funkcje do projektowania filtrów eliptycznych
    from  scipy.signal import lfilter, filtfilt # funkcje do aplikowania filtrów
    # przyda nam się oś czasu od -T do T sekund
    t = np.arange(-T, T, 1/Fs)
    
    # oś częstości przeliczamy na radiany
    w = 2*np.pi* f/Fs #wektor od 0 do pi, f/Fs to od 0 do 0.5
    
    # obliczamy transmitancję
    w, h = freqz(b,a,worN=w) #w to znormalizowane częstosci, h to odpowiedz

    # obliczamy moduł transmitancji
    m = np.abs(h)
        
    # obliczamy fazę i "rozwijamy" ją   
    faza = np.unwrap(np.angle(h)) #dopelnienie do wielokrotnosci pi

    # obliczamy opóźnienie fazowe
    opoznienieFazowe = - faza/w

    # obliczamy opóźnienie grupowe
    df = np.diff(faza)
    idx, = np.where(np.abs(df-np.pi)<0.05) #to zabezpieczenie na błędy przy "rozwijaniu" fazy
    df[idx] = (df[idx+1]+df[idx-1])/2
    grupowe = - df/np.diff(w)
 
    # obliczamy odpowiedź impulsową
    x = np.zeros(len(t))
    x[len(t)//2] = 1 # impuls
    y = lfilter(b,a,x) # przepuszczamy impuls przez filtr i obserwujemy odpowiedź impulsową
    
    # obliczamy odpowiedź schodkową
    s = np.zeros(len(t))
    s[len(t)//2:] = 1 # schodek
    ys = lfilter(b,a,s) # przepuszczamy schodek przez filtr i obserwujemy odpowiedź schodkową
    
    # rysujemy
    fig = py.figure()
    py.subplot(3,2,1)
    py.title('moduł transmitancji')
    py.plot(f,20*np.log10(m))
    py.ylabel('[dB]')
    py.grid('on')
    
    py.subplot(3,2,3)
    py.title('opóźnienie fazowe')
    py.plot(f, opoznienieFazowe)
    py.ylabel('próbki')
    py.grid('on')
    
    py.subplot(3,2,5)
    py.title('opóźnienie grupowe')
    py.plot(f[:-1],grupowe)
    py.ylabel('próbki')
    py.xlabel('Częstość [Hz]')
    py.grid('on')
    #py.ylim([0, np.max(grupowe)+1])
    
    py.subplot(3,2,2)
    py.title('odpowiedź impulsowa')
    py.plot(t, x)
    py.plot(t, y)
    py.xlim([-T/2,T])
    py.grid('on')
    
    py.subplot(3,2,4)
    py.title('odpowiedź schodkowa')
    py.plot(t, s)
    py.plot(t, ys)
    py.xlim([-T/2,T])
    py.xlabel('Czas [s]')
    py.grid('on')
    
    fig.subplots_adjust(hspace=.5)
    py.show()
    


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
keys = np.zeros((204,3),dtype=str) # row 0 - correct, row 1 - chosen, row 2 - r/f
for event in range(204):
    keys[event,0]=data['CORRECT'][event+1]
    keys[event,1]=data['key_resp_trial.keys'][event+1]
    keys[event,2]=data['key_resp_trial.corr'][event+1]

# Opening file.
with open('CV_32_flankery.raw') as f:
    s=np.fromfile(f, dtype='float32')
    s=np.reshape(s, (int(len(s)/ch),int(ch)))
s=s.T

# Filtering with butterworth filter - highpass form 0.1 Hz.
#[b,a]=ss.butter(3,0.1/(Fs/2), btype="highpass")
#sf = ss.filtfilt(b,a,s)
[b,a]=ss.butter(3,[1/(Fs/2), 45/(Fs/2)], btype="bandpass")
sf = ss.filtfilt(b,a,s)
#charkterystyki(a,b,np.arange(0.01,Fs/2,0.01) ,T=0.2,Fs=500.0)
#plotingEEG(sf)

# Minumum length of stimuli-response.
len_min = int(min(times[:,1]-times[:,0])*Fs)
len_max = int(max(times[:,1]-times[:,0])*Fs)
len_mean = int(np.mean(times[:,1]-times[:,0])*Fs)
len_frag = len_min

# Retrieveing trial fragments - 5sec~response.
trials_rZ = np.zeros((92,len_frag))
trials_rM = np.zeros((93,len_frag))
trials_wZ = np.zeros((10,len_frag))
trials_wM = np.zeros((9,len_frag))
num,num2,num3,num4=0,0,0,0

channel = 5
# Inserting signals from z and m, right and wrong ones to matrix.
for event in range(204):
    #trial = sf[channel][(int((times[event,1])*Fs)-len_frag):int(times[event,1]*Fs)]
    trial = sf[channel][(int((times[event,0])*Fs)):(int(times[event,0]*Fs)+len_frag)]
    if keys[event,0]=='z' and keys[event,2] == '1': 
        trials_rZ[num,:]=trial
        num +=1
    elif keys[event,0]=='m' and keys[event,2] == '1': 
        trials_rM[num2,:]=trial
        num2 +=1
    elif keys[event,0]=='z' and keys[event,2] == '0': 
        trials_wZ[num3,:]=trial
        num3 +=1
    elif keys[event,0]=='m' and keys[event,2] == '0': 
        trials_wM[num4,:]=trial
        num4 +=1
'''
# Plotting the single trials from each group and averaged signal.
colors = cm.autumn(np.linspace(0, 0.7, 92))
colors2 = cm.autumn(np.linspace(0, 0.7, 93))
colors3 = cm.autumn(np.linspace(0, 0.7, 10))
colors4 = cm.autumn(np.linspace(0, 0.7, 9))

plt.figure()    
plt.subplot(2,2,1)
for i in range(92):
    plt.plot(trials_rZ[i],color=colors[i])
plt.plot(np.mean(trials_rZ,axis=0), color='b')
plt.ylim(-550,550)
plt.title('Right Z')
plt.subplot(2,2,2)
for i in range(93):
    plt.plot(trials_rM[i],color=colors2[i])
plt.plot(np.mean(trials_rM,axis=0), color='b')
plt.title('Right M')
plt.ylim(-550,550)
plt.subplot(2,2,3)
for i in range(10):
    plt.plot(trials_wZ[i],color=colors3[i])
plt.plot(np.mean(trials_wZ,axis=0), color='b')
plt.title('Wrong Z')
plt.ylim(-550,550)
plt.subplot(2,2,4)
for i in range(9):
    plt.plot(trials_wM[i],color=colors4[i])
plt.plot(np.mean(trials_wM,axis=0), color='b')
plt.title('Wrong M')
plt.tight_layout()
plt.ylim(-550,550)
plt.show()


'''
#Calculating spectrogram for each category


#II - Averaging the signals from each group and calculating the specrum from the mean. 
spectra = np.zeros((4,len_frag))
spectra[0,:] = np.abs(np.fft.fft(np.mean(trials_rZ,axis=0)))
spectra[1,:] = np.abs(np.fft.fft(np.mean(trials_rM,axis=0)))
spectra[2,:] = np.abs(np.fft.fft(np.mean(trials_wZ,axis=0)))
spectra[3,:] = np.abs(np.fft.fft(np.mean(trials_wM,axis=0)))
freq = np.fft.fftfreq(len_frag,1/Fs)
'''
#I - Calculating the spectrum for each signal form every trial and averaging the spectra from each group.
spectra = np.zeros((4,len_frag))
spi_trials_rZ = np.zeros((92,len_frag))
spi_trials_rM = np.zeros((93,len_frag))
spi_trials_wZ = np.zeros((10,len_frag))
spi_trials_wM = np.zeros((9,len_frag))

for i in range(len(trials_rZ)):
    spi_trials_rZ[i,:] = np.abs(np.fft.fft(trials_rZ[i,:]))
for i in range(len(trials_rM)):
    spi_trials_rM[i,:] = np.abs(np.fft.fft(trials_rM[i,:]))
for i in range(len(trials_wZ)):
    spi_trials_wZ[i,:] = np.abs(np.fft.fft(trials_wZ[i,:]))
for i in range(len(trials_wM)):
    spi_trials_wM[i,:] = np.abs(np.fft.fft(trials_wM[i,:]))
    
spectra[0,:] = np.mean(spi_trials_rZ,axis=0)
spectra[1,:] = np.mean(spi_trials_rM,axis=0)
spectra[2,:] = np.mean(spi_trials_wZ,axis=0)
spectra[3,:] = np.mean(spi_trials_wM,axis=0)
freq = np.fft.fftfreq(len_frag,1/Fs)
'''
# Plotting calculated specra.
fig = plt.figure(figsize=(12,8))    
plt.subplot(2,2,1)
plt.plot(spectra[0,:])
plt.title('Right Z')
plt.xlim(0,40)
plt.axvline(x=9, color='red')
plt.axvline(x=12, color='green')
plt.axvline(x=2*9, color='red',ls=':')
plt.axvline(x=2*12, color='green',ls=':')
plt.axvline(x=3*9, color='red',ls=':')
plt.axvline(x=3*12, color='green',ls=':')
plt.ylim(0,8000)
plt.grid('on',alpha=0.7)
plt.subplot(2,2,2)
plt.plot(spectra[1,:])
plt.title('Right M')
plt.xlim(0,40)
plt.axvline(x=9, color='red')
plt.axvline(x=12, color='green')
plt.axvline(x=2*9, color='red',ls=':')
plt.axvline(x=2*12, color='green',ls=':')
plt.axvline(x=3*9, color='red',ls=':')
plt.axvline(x=3*12, color='green',ls=':')
plt.ylim(0,8000)
plt.grid('on',alpha=0.7)
plt.subplot(2,2,3)
plt.plot(spectra[2,:])
plt.title('Wrong Z')
plt.xlim(0,40)
plt.axvline(x=9, color='red')
plt.axvline(x=12, color='green')
plt.axvline(x=2*9, color='red',ls=':')
plt.axvline(x=2*12, color='green',ls=':')
plt.axvline(x=3*9, color='red',ls=':')
plt.axvline(x=3*12, color='green',ls=':')
plt.ylim(0,8000)
plt.grid('on',alpha=0.7)
plt.subplot(2,2,4)
plt.plot(spectra[3,:])
plt.title('Wrong M')
plt.xlim(0,40)
plt.axvline(x=9, color='red')
plt.axvline(x=12, color='green')
plt.axvline(x=2*9, color='red',ls=':')
plt.axvline(x=2*12, color='green',ls=':')
plt.axvline(x=3*9, color='red',ls=':')
plt.axvline(x=3*12, color='green',ls=':')
plt.ylim(0,8000)
plt.legend(['spectrum','9 Hz', '12 Hz', '9 Hz harmonic', '12 Hz harmonic'])
plt.grid('on',alpha=0.7)
fig.text(0.5, -0.01, 'Frequencies [Hz]', ha='center', fontsize=12)
fig.text(-0.01, 0.5,'Amplitude' , va='center', rotation='vertical',fontsize=12)
plt.tight_layout()
plt.show()
