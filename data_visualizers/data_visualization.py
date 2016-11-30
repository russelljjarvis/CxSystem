import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import zlib
import os
from brian2 import *
import bz2
import pandas as pd

time_for_visualization = np.array([0, 0.36]) + 0.000001  # To accept 0 as starting point. Rounding error for the end.
dt = 0.1 * ms
plot_dt = 0.1 * ms
# state_variable_to_monitor = 'vm_all'
state_variable_to_monitor = 'wght_all'
# state_variable_to_monitor = 'apre_all'

# data_file_name = '../CX_OUTPUT/CX_Output_20161108_11000084_Python_1000ms.gz'
directory = '/opt/Laskenta/Output/CX_Output'
# directory = '/opt3/CX_Output'


class DataVisualization:

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

    def make_figure(self):
        extensions = ['.gz', '.bz2', '.pickle']
        most_recent_data_file_name = max([os.path.join(directory, files) for files in os.listdir(directory)
                                          if files.endswith(tuple(extensions))], key=os.path.getmtime)
        neuron_groups = ['NG0_relay_video', 'NG1_PC_L4toL2', 'NG3_BC_L4', 'NG2_PC_L2toL1', 'NG4_BC_L2']

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
            stvar_of_interest = simulation_data[state_variable_to_monitor]

            fig = plt.figure()

            if type(metadata) is not dict:
                figure_title = '%s = %s' % (metadata.iloc[row_index]['Level-1 Parameter'],
                                            str(metadata.iloc[row_index]['Level-1 Value']))
            elif type(metadata) is dict:
                figure_title = 'A nice figure'

            fig.suptitle(figure_title, fontsize=12)

            for plot_index, neuron_group in enumerate(neuron_groups):

                n_columns = 3
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
                    plt.title('%s vm' % neuron_group)

                elif any([(neuron_group+'__') in syn_name for syn_name in stvar_of_interest.keys()]) or \
                     any([(neuron_group+'__') in syn_name for syn_name in stvar_of_interest.keys()]):
                    target_syn_idx = [syn_idx for syn_idx,syn_name in enumerate(stvar_of_interest.keys()) if (neuron_group+'__') in syn_name][0]
                    stvar = stvar_of_interest.keys()[target_syn_idx]
                    stvar_value = stvar_of_interest[stvar]
                    plt.subplot(len(neuron_groups), n_columns, plot_index * n_columns + 3)
                    # N_time_points = len(stvar_value[0])
                    data_indices_for_plot = np.array([time_for_visualization[0] * (1 * second)
                                                      / dt, time_for_visualization[1] * (1 * second) / dt], dtype=int)
                    subsample_step = int(plot_dt / dt)
                    subsampled_data_epoch_for_plot = stvar_value[:, data_indices_for_plot[0]:data_indices_for_plot[1]:
                                                                 subsample_step]
                    time_vector = np.arange(time_for_visualization[0], time_for_visualization[1],
                                            plot_dt / (1 * second))
                    plt.plot(time_vector, subsampled_data_epoch_for_plot.T, '-')
                    plt.title('%s wght' % stvar_of_interest.keys()[target_syn_idx])
                else:
                    pass

        plt.show()

if __name__ == '__main__':
    dv = DataVisualization()
    dv.make_figure()