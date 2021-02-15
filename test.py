# -*- coding: utf-8 -*-
from collections import defaultdict
import pickle, glob
import pandas as pd
from obspy import read, Trace
import numpy as np
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy import signal

def scale_finder(st):
	# Scaling the data is important since not all the data is recorded in cm/s^2. This function returns the scaling for a given obspy stream.
	if st[0].stats.network == 'TW' or st[0].stats.network == 'UV' or st[0].stats.network == 'CN' or st[
		0].stats.network == 'Seisram':  # or st[0].stats.network == 'IG'
		scale = 10 ** -7
	elif st[0].stats.network == 'GE' or st[0].stats.network == 'HL' or st[0].stats.network == 'HC' or st[
		0].stats.network == 'HT' or st[0].stats.network == 'KO' \
			or st[0].stats.network == 'AV' or st[0].stats.network == 'AK' or st[0].stats.network == 'TA' or st[
		0].stats.network == 'PB' or st[0].stats.network == 'CI' \
			or st[0].stats.network == 'US' or st[0].stats.network == 'G' or st[0].stats.network == 'X1' or st[
		0].stats.network == 'BK' or st[0].stats.network == 'UW' \
			or st[0].stats.network == 'TS' or st[0].stats.network == 'AZ' or st[0].stats.network == 'IU' or st[
		0].stats.network == 'TU' or st[0].stats.network == 'NP' \
			or st[0].stats.network == 'YZ' or st[0].stats.network == 'YO' or st[0].stats.network == 'II' or st[
		0].stats.network == 'OV' or st[0].stats.network == 'TC' \
			or st[0].stats.network == 'AT' or st[0].stats.network == 'HV':
		scale = 100
	elif st[0].stats.network == 'BO':
		scale = 100
	elif st[0].stats.network == 'GEONET':
		scale = 0.1
	else:
		scale = 1
	return scale


df = pd.read_csv("/home/dertuncay/Pulse_Identification/Pulse_Identification/DBRerun/db_final.csv")
# df = pd.read_csv("/home/dertuncay/Total DB creator/totaldb.csv")

progress = ProgressBar(max_value=df.shape[0])
output = []
# Synthetic (0=Real,1=Synthetic), Manual_Pulse, Baker_Tp, Chang_Tp, Deniz_Tp
labels = []
startandend = []
aerror = 0
ierror = 0
SRATE = 20.0
SLEN = 60.0


# Real Signals
for index, row in progress(df.iterrows()):
# for index, row in df.iterrows():
	try:
		st = read(row['Path'], format='SAC')
		# st.normalize()
		# bandpass 0.05 to 5
		st[0].filter('bandpass', freqmin=0.05, freqmax=5)
		st[0].detrend("demean")
		st[0].detrend("linear")
		st[0].integrate(method='cumtrapz')
		st[0].detrend("demean")
		st[0].detrend("linear")
		st.normalize()

		# Resample the data to 20Hz.
		if st[0].stats.sampling_rate != SRATE:
			st[0].resample(SRATE)
		sr = st[0].stats.sampling_rate

		# Start from P wave arrival
		try:
			start_time = st[0].stats.starttime + st[0].stats.sac.a
			# Cut the data between P wave arrival and 40 seconds from P wave arrival. 60 seconds is arbitrary but major earthquakes last around 40 - 60 seconds.
			st.trim(starttime=start_time, endtime=start_time + SLEN - 0.01, pad=True, fill_value=0)
			# If the signal has duration less than 40 seconds, add zeros to the end.
			length = int(SLEN * SRATE)
			if len(st[0].data) < SLEN * int(sr):
				tr_zeros = Trace(data=np.zeros((length - len(st[0].data),)))
				n_trace = np.concatenate((st[0].data, tr_zeros), axis=0)
			else:
				n_trace = st[0].data[:length]
			# # Scale the waveform
			# scale = scale_finder(st)
			eq_data = Trace(n_trace)# * scale)
			# if row['Manual_Pulse'] == 1:
			# 	plt.plot(eq_data)
			# 	plt.savefig('Impulse_Figures/Real/' + st[0].stats.station + '_' + st[0].stats.channel + '.png',dpi=300)
			# 	plt.close('all')

			if not np.isnan(row['Manual_Pulse']):
			# if not np.isnan(row['Manual Pulse']):
				output.append(eq_data.data)
				labels.append([row['Manual_Pulse'],row['Baker_Tp'],row['Chang_Tp'],row['Deniz_Tp']])
				# labels.append([row['Manual Pulse'],row['Tp Baker'],row['Tp Chang'],row['Tp Deniz']])
		except AttributeError:
			aerror += 1
		except IndexError:
			ierror += 1
	except IOError:
		print(row['Path'])


output = np.array(output)
labels = np.array(labels)
print("Output dimensions:", output.shape)
with open('data/inputs.pkl', 'wb') as f:
	pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
with open('data/labels.pkl', 'wb') as f:
	pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

#Synthetic Signals
# Root
output_syn = [];
syns = glob.glob('/home/dertuncay/Synthetic_Results/Synthetic_Pulse/*')
# import random
# syns = random.choices(population=syns, k=17620-tmp_pos)

# sacs1 = glob.glob('/home/dertuncay/Pulse_Identification/SyntheticPulses/pulses/*')
# sacs2 = glob.glob('/home/dertuncay/Pulse_Identification/SyntheticPulses/pulses2/*')
# syns  = sacs1 + sacs2
progress = ProgressBar(max_value=len(syns))
for syn in progress(syns):

	st = read(syn, format='SAC')
	# bandpass 0.05 to 10
	st[0].filter('bandpass',freqmin=0.05,freqmax=5)
	st[0].detrend("demean")
	st[0].detrend("linear")
	st[0].integrate(method='cumtrapz')
	st[0].detrend("demean")
	st[0].detrend("linear")
	st.normalize()
	# Resample the data to 20Hz.
	if st[0].stats.sampling_rate != SRATE:
		st[0].resample(SRATE)
	sr = st[0].stats.sampling_rate
	st.trim(starttime=st[0].stats.starttime, endtime=st[0].stats.starttime + SLEN - 0.01, pad=True, fill_value=0)
	# noise = np.random.normal(0,np.amax(np.abs(st[0].data)) * 0.01,int(SLEN * SRATE))
	# st[0].data = st[0].data * 0.1
	# if np.amax(np.abs(st[0].data)) > 30:
	if len(st[0].data) > SLEN * SRATE:
		output_syn.append(st[0].data[:int(SLEN*SRATE-len(st[0].data))])# + noise)
	else:
		output_syn.append(st[0].data)# + noise)#[:-1]
	# labels.append([1,1,1,1,1])
	# plt.plot(st[0].data[:1200] + noise)
	# plt.savefig('Impulse_Figures/Syn/' + syn.split('/')[-1] + '.png',dpi=300)
	# plt.close('all')

output_syn = np.array(output_syn)
print("Output Synthetic dimensions:", output_syn.shape)
with open('data/synthetic.pkl', 'wb') as f:
	pickle.dump(output_syn, f, pickle.HIGHEST_PROTOCOL)
