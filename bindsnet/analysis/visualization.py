import sys
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

def spike_trains_main(data=None, n_ex=None, top_k=None, indices=None):
	for j in range(top_k):
		plot_spike_trains_examples(data, n_ex=n_ex)

def create_movie(weights):
	fig = plt.figure()
	im = plt.imshow(weights[:, :, 0], cmap='hot_r', animated=True, vmin=0, vmax=1)
	plt.axis('off'); plt.colorbar(im)
		
	def update(j):
		im.set_data(weights[:, :, j])
		return [im]
	
	global ani; ani=0
	ani = animation.FuncAnimation(fig, update, frames=weights.shape[-1], interval=1000, blit=True)
	plt.show()

def plot_weights_movie(weights):
	n_examples = weights.shape[0]
	ws = []
	for i in range(n_examples):
		ws.append(get_weights_for_example(weights[i]))
	ws = np.concatenate(ws, axis=2)
	create_movie(ws)
		
def get_weights_for_example(w, sample_every=1):
	return w[:, :, range(0, w.shape[2], sample_every)]

def plot_spike_trains_for_example(spikes, n_ex=None, top_k=None, indices=None):
	'''
	Plot spike trains for top-k neurons or for specific indices.
	
	Inputs:
	
		| :code:`spikes` (:code:`torch.tensor (N_examples, N_neurons, time)`): Spiking train data for a population of neurons for one example
		| :code:`n_ex` (:code:`int`): Allows user to pick which example to plot spikes for. Must be >= 0
		| :code:`top_k` (:code:`int`): Plot k neurons that spiked the most for n_ex example
		| :code:`indices` (:code:`list(int)`): Plot specific neurons' spiking activity instead of top_k. Meant to replace top_k. 
	'''

	assert (n_ex is not None and n_ex >= 0 and n_ex < spikes.shape[0])
	
	plt.figure()
	
	if top_k is None and indices is None: # Plot all neurons' spiking activity
		spike_per_neuron = [np.argwhere(i==1).flatten() for i in spikes[n_ex, :, :]]
		
	elif top_k is None: # Plot based on indices parameter
		assert (indices is not None)
		spike_per_neuron = [np.argwhere(i==1).flatten() for i in spikes[n_ex, indices, :]]
		
	elif indices is None: # Plot based on top_k parameter
		assert (top_k is not None)
		# Obtain the top k neurons that fired the most
		top_k_loc = np.argsort(np.sum(spikes[n_ex,:,:], axis=1), axis=0)[::-1]
		spike_per_neuron = [np.argwhere(i==1).flatten() for i in spikes[n_ex, top_k_loc[0:top_k], :]]
		plt.title('Spiking activity for top %d neurons'%top_k)
		
	plt.eventplot(spike_per_neuron, linelengths= [0.5]*len(spike_per_neuron))
	plt.xlabel('Simulation Time'); plt.ylabel('Neuron index')
	plt.show()

def plot_voltage(voltage, n_ex=0, n_neuron=0, time=None, threshold=None):
	'''
	Plot voltage for a single neuron on a specific example.
	
	Inputs:

		| :code:`voltage` (:code:`torch.Tensor`): Membrane voltage data for a population of neurons.
		| :code:`n_ex` (:code:`int`): Allows user to pick which example to plot voltage for.
		| :code:`n_neuron` (:code:`int`): Neuron index for which to plot voltages for
		| :code:`time` (:code:`tuple(int)`): Plot spiking activity of neurons between the given range of time. 
		| :code:`threshold` (:code:`float`): Neuron spiking threshold. Will be shown on the plot.
	'''
	
	assert (n_ex >= 0 and n_neuron >= 0)
	assert (n_ex < voltage.shape[0] and n_neuron < voltage.shape[1])

	if time is None:
		time = (0, voltage.shape[-1])
	else:
		assert (time[0] < time[1])
		assert (time[1] <= voltage.shape[-1])
	
	timer = np.arange(time[0], time[1])
	time_ticks = np.arange(time[0], time[1]+1, 10)
	
	plt.figure()
	plt.plot(voltage[n_ex, n_neuron, timer])
	plt.xlabel('Simulation Time'); plt.ylabel('Voltage'); plt.title('Membrane voltage of neuron %d for example %d'%(n_neuron, n_ex+1))
	locs, labels = plt.xticks()
	locs = range(int(locs[1]), int(locs[-1]), 10)
	plt.xticks(locs, time_ticks)
	
	# Draw threshold line only if given
	if threshold is not None:
		plt.axhline(threshold, linestyle='--', color='black', zorder=0)
		
	plt.show()
	
def main():
	np.random.seed(0)
	data = np.random.binomial(1, 0.25, size=(5, 20, 100))
	data = np.random.uniform(0, 30, size=(5, 20, 100))
	data = np.sort(data, axis=2)
	plot_voltages(data, n_ex=2, n_neuron=15, threshold=20)
	
if __name__ == '__main__':
	main()
		
	
