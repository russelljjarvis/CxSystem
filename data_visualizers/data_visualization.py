from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import zlib
import os
from brian2 import *
import bz2
import pandas as pd

time_for_visualization = np.array([0, 0.09])   #+ 0.00001  # To accept 0 as starting point. Rounding error for the end.
# dt = 0.1 * ms
plot_dt = 1 * ms

state_variable_to_monitor = 'vm'
# state_variable_to_monitor = 'wght'
# state_variable_to_monitor = 'spike_sensor'
# state_variable_to_monitor = 'synaptic_scaling_factor'
# state_variable_to_monitor = 'Apre'
# state_variable_to_monitor = 'Apost'

state_variable = state_variable_to_monitor + '_all'

# data_file_name = '../CX_OUTPUT/CX_Output_20161108_11000084_Python_1000ms.gz'
directory = '/opt/Laskenta/Output/CX_Output'
# directory = '/opt3/CX_Output'

# TODO build frequency vs scaling_factor plot. Check that cell_group_wise scaling factor
class DataVisualization:

    def __init__(self):
        self.neuron_groups = ['NG0_relay_video', 'NG1_PC_L4toL2', 'NG3_BC_L4', 'NG2_PC_L2toL1', 'NG4_BC_L2']

    def data_loader(self, input_path):
        if '.gz' in input_path:
            with open(input_path, 'rb') as fb:
                data = zlib.decompress(fb.read())
                loaded_data = pickle.loads(data)
        elif '.bz2' in input_path:
            with bz2.BZ2File(input_path, 'rb') as fb:
                loaded_data = pickle.load(fb)
        elif 'pickle' in input_path:
            with open(input_path, 'rb') as fb:
                loaded_data = pickle.load(fb)
        return loaded_data

    def firing_rate_histograms(self, simulation_data, figure_title):
        spikes_all = simulation_data['spikes_all']
        total_time = simulation_data['time_vector'][-1]  # Last point in the time vector
        state_variable_data_dict = simulation_data['synaptic_scaling_factor_all']

        fig = plt.figure()
        figure_title_overview = 'Firing rates and synaptic scaling for neuron groups; %s' % figure_title
        fig.suptitle(figure_title_overview, fontsize=12)
        n_rows = len(spikes_all)
        n_columns = 3  # The three plots side-by-side
        title_font = {'fontname': 'Arial', 'size': '12', 'color': 'black', 'weight': 'normal',
                      'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space

        for subplot_index, neuron_group in enumerate(self.neuron_groups):
            number_of_neurons = simulation_data['number_of_neurons'][neuron_group]

            spike_indices = spikes_all[neuron_group][0]
            spike_counts, bin_edges = np.histogram(spike_indices, bins=np.arange(number_of_neurons+1))
            frequencies = spike_counts / total_time


            # the histogram of the data
            plt.subplot(n_rows, n_columns, subplot_index * n_columns + 1)
            plt.bar(np.arange(number_of_neurons), frequencies)
            plt.title('%s spike frequencies' % neuron_group, **title_font)
            if subplot_index == n_rows - 1:
                plt.ylabel('Firing rate (Hz)')
                plt.xlabel('Neuron index')

            plt.subplot(n_rows, n_columns, subplot_index * n_columns + 2)
            plt.hist(frequencies, facecolor='green', alpha=0.75)
            if subplot_index == n_rows - 1:
                plt.ylabel('Number of neurons')
                plt.xlabel('Firing rate (Hz)')

            if neuron_group in state_variable_data_dict.keys():
                # Parsing the sampled_cells array from the csv monitor substring. Unfortunately very difficult.
                df=simulation_data['Anatomy_configuration']
                bool_idx = df.applymap(lambda x: True if state_variable_to_monitor + '[rec]' in str(x) else False)  #Find matching cells
                index_to_pattern = np.where(bool_idx)
                appropriate_rows = df.ix[index_to_pattern[0]]  # Assumes that the neuron group monitors are in the same column in the csv file
                bool_correct_row = appropriate_rows[1].str.contains(neuron_group[2]) # Select correct row of dataframe based on NG number
                correct_row_data = appropriate_rows[bool_correct_row]
                correct_cell_data = str(correct_row_data.values[0][index_to_pattern[1][0]])
                assert state_variable_to_monitor in correct_cell_data, "State variable to monitor not found"
                starting_index = correct_cell_data.index(state_variable_to_monitor + '[rec]')
                ending_index = correct_cell_data.index(')', starting_index) + 1
                exestring = 'np.arange' + correct_cell_data[
                                          starting_index + len(state_variable_to_monitor + '[rec]'):ending_index]
                final_exestring = 'sampled_cells = ' + exestring.replace('-', ',')
                exec final_exestring in globals(), locals()

                plt.subplot(n_rows, n_columns, subplot_index * n_columns + 3)
                plt.plot(frequencies[sampled_cells], state_variable_data_dict[neuron_group][:,-1],'.')
            if subplot_index == n_rows - 1:
                plt.ylabel(state_variable_to_monitor)
                plt.xlabel('Firing rate (Hz)')

        plt.subplots_adjust(hspace=0.4)

    def make_figure(self):
        extensions = ['.gz', '.bz2', '.pickle']
        most_recent_data_file_name = max([os.path.join(directory, files) for files in os.listdir(directory)
                                          if files.endswith(tuple(extensions))], key=os.path.getmtime)
        neuron_groups = self.neuron_groups

        metadata = self.data_loader(most_recent_data_file_name)

        if type(metadata) is dict:
            n_figures = [0]
        elif type(metadata) is not dict:
            n_figures = np.arange(len(metadata))

        # For loop for N entries in df. For each, load data, create figure
        for row_index in n_figures:

            if type(metadata) is dict:
                data_file_name = most_recent_data_file_name
            elif type(metadata) is not dict:
                data_file_name = metadata.iloc[row_index]['Full path']

            simulation_data = self.data_loader(data_file_name)
            positions = simulation_data['positions_all']
            spikes_all = simulation_data['spikes_all']
            stvar_of_interest = simulation_data[state_variable]
            dt = simulation_data['time_vector'][1] - simulation_data['time_vector'][0]

            if type(metadata) is not dict:
                figure_title = '%s = %s' % (metadata.iloc[row_index]['Dimension-1 Parameter'],
                                            str(metadata.iloc[row_index]['Dimension-1 Value']))
            elif type(metadata) is dict:
                figure_title = 'A nice figure'

            self.firing_rate_histograms(simulation_data, figure_title)

            fig = plt.figure()

            fig.suptitle(figure_title, fontsize=12)

            col_max = 1
            for group_index, neuron_group in enumerate(neuron_groups):
                repeat_col = sum([1 for pp in stvar_of_interest.keys() if (neuron_group+'__') in pp])
                col_max = repeat_col if repeat_col > col_max else col_max

            for plot_index, neuron_group in enumerate(neuron_groups):
                n_columns = 2 + col_max
                plt.subplot(len(neuron_groups), n_columns, plot_index * n_columns + 1)
                plt.plot(np.real(positions['w_coord'][neuron_group]), np.imag(positions['w_coord'][neuron_group]), '.')
                plt.title('%s positions' % neuron_group)

                spikes = spikes_all[neuron_group]
                plt.subplot(len(neuron_groups), n_columns, plot_index * n_columns + 2)
                # plt.plot(spikes[1][::100],spikes[0][::100], '.k')
                plt.plot(spikes[1], spikes[0], '.k')
                plt.xlim([time_for_visualization[0], time_for_visualization[1]])
                plt.title('%s spikes' % neuron_group)

                if neuron_group in stvar_of_interest.keys():
                    stvar_value = stvar_of_interest[neuron_group]
                    plt.subplot(len(neuron_groups), n_columns, plot_index * n_columns + 3)
                    data_indices_for_plot = np.array([time_for_visualization[0] * (1 * second)
                                                      / dt, time_for_visualization[1] * (1 * second) / dt], dtype=int)
                    subsample_step = int(plot_dt / dt)
                    subsampled_data_epoch_for_plot = stvar_value[:, data_indices_for_plot[0]:
                                                                 data_indices_for_plot[1]:subsample_step]
                    time_vector = np.arange(time_for_visualization[0],
                                            time_for_visualization[1], plot_dt / (1 * second))
                    plt.plot(time_vector, subsampled_data_epoch_for_plot.T, '-')
                    plt.title('%s %s' % (neuron_group,state_variable_to_monitor))

                elif any([(neuron_group+'__') in syn_name for syn_name in stvar_of_interest.keys()]) or \
                     any([(neuron_group+'__') in syn_name for syn_name in stvar_of_interest.keys()]):
                    target_syn_idx = [syn_idx for syn_idx, syn_name in enumerate(stvar_of_interest.keys())
                                      if (neuron_group+'__') in syn_name]
                    for index_of_syn_idx, syn_idx in enumerate(target_syn_idx) :
                        stvar = stvar_of_interest.keys()[syn_idx]
                        stvar_value = stvar_of_interest[stvar]
                        plt.subplot(len(neuron_groups), n_columns, plot_index * n_columns + 3 + index_of_syn_idx)
                        # N_time_points = len(stvar_value[0])
                        data_indices_for_plot = np.array([time_for_visualization[0] * (1 * second) / dt,
                                                          time_for_visualization[1] * (1 * second) / dt], dtype=int)
                        subsample_step = int(plot_dt / dt)
                        subsampled_data_epoch_for_plot = stvar_value[:, data_indices_for_plot[0]:
                                                                     data_indices_for_plot[1]:subsample_step]
                        time_vector = np.arange(time_for_visualization[0], time_for_visualization[1],
                                                plot_dt / (1 * second))
                        plt.plot(time_vector, subsampled_data_epoch_for_plot.T, '-')
                        title_text = '%s %s' % (stvar_of_interest.keys()[syn_idx], state_variable_to_monitor)
                        if len(title_text) > 26:
                            title_font = {'fontname': 'Arial', 'size': '8', 'color': 'black', 'weight': 'normal',
                                          'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space

                        plt.title(title_text,**title_font)
                else:
                    pass

                plt.subplots_adjust(hspace=0.3)

        plt.show()

if __name__ == '__main__':
    dv = DataVisualization()
    dv.make_figure()