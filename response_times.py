import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas

# Extracting data if the pressed key was correct
with open('CV_32_flankery.csv') as csv_file:
    data = pandas.read_csv(csv_file) 
num_trials = len(data)-1
keys = np.zeros((num_trials,3),dtype=str) # row 0 - r/f, row 1  - side, row 2 - center
for event in range(num_trials):
    keys[event,0]=data['key_resp_trial.corr'][event+1]
    keys[event,1]=data['PARAM_G'][event+1]
    keys[event,2]=data['PARAM_S'][event+1]

# Extracting for each trial the response time (from the start of flckering til pressing the key).
times = np.zeros(num_trials,dtype=float) 
soup = BeautifulSoup(open('CV_32_flankery.tag'), 'lxml')
event_num = 0
for tag in soup.find_all('tag'):
    if tag['name'] == 'odpowiedz':
        times[event_num] = float(tag['length'])
        event_num+=1
           
num_same,num_diff = 0,0
times_same, times_diff = [], []

for event in range(num_trials):
    if (keys[event,0] == '1') and (keys[event,1]==keys[event,2]): 
        times_same.append(times[event]*1000)
        num_same +=1
    elif (keys[event,0] == '1') and (keys[event,1]!=keys[event,2]):
        times_diff.append(times[event]*1000)
        num_diff +=1
con = np.array(times_same)
incon = np.array(times_diff)

# Ploting the instantaneous power of fragments.
fig = plt.figure(figsize=(5,3))

plt.grid('on', alpha=0.6)
plt.hist(con, bins=int(num_same/2.7), color='green', alpha=0.7)
plt.hist(incon, bins=int(num_diff/2.7), color='red', alpha=0.7)
plt.legend(['congruent', 'incongruent'])


fig.text(0.5, 1.01,'Response times', ha='center', fontsize=15)
fig.text(0.5, -0.01, 'Time [ms]', ha='center', fontsize=12)
fig.text(-0.01, 0.5,'Number of trials' , va='center', rotation='vertical',fontsize=12)

plt.tight_layout()
plt.show()