from __future__ import division
import json
import os.path
import re

# Mapping from Markram et al cell groups to own cell groups
cell_group_dict = {
    'L1_DAC': 'L1_I', 'L1_DLAC': 'L1_I', 'L1_HAC': 'L1_I', 'L1_NGC-DA': 'L1_I', 'L1_NGC-SA': 'L1_I', 'L1_SLAC': 'L1_I',
    'L23_BP': 'L23_UM_I', 'L23_BTC': 'L23_UM_I', 'L23_ChC': 'L23_UM_I', 'L23_DBC': 'L23_UM_I', 'L23_LBC': 'L23_BC',
    'L23_MC': 'L23_MC', 'L23_NBC': 'L23_BC', 'L23_NGC': 'L23_UM_I', 'L23_PC': 'L23_PC', 'L23_SBC': 'L23_BC',
    'L4_BP': 'L4_UM_I', 'L4_BTC': 'L4_UM_I', 'L4_ChC': 'L4_UM_I', 'L4_DBC': 'L4_UM_I', 'L4_LBC': 'L4_BC',
    'L4_MC': 'L4_MC', 'L4_NBC': 'L4_BC', 'L4_NGC': 'L4_UM_I', 'L4_PC': 'L4_PC1', 'L4_SBC': 'L4_BC', 'L4_SP': 'L4_PC2',
    'L4_SS': 'L4_SS', 'L5_BP': 'L5_UM_I', 'L5_BTC': 'L5_UM_I', 'L5_ChC': 'L5_UM_I', 'L5_DBC': 'L5_UM_I',
    'L5_LBC': 'L5_BC', 'L5_MC': 'L5_MC', 'L5_NBC': 'L5_BC', 'L5_NGC': 'L5_UM_I', 'L5_SBC': 'L5_BC',
    'L5_STPC': 'L5_PC', 'L5_TTPC1': 'L5_PC', 'L5_TTPC2': 'L5_PC', 'L5_UTPC': 'L5_PC', 'L6_BP': 'L6_UM_I',
    'L6_BPC': 'L6_PC1', 'L6_BTC': 'L6_UM_I', 'L6_ChC': 'L6_UM_I', 'L6_DBC': 'L6_UM_I', 'L6_IPC': 'L6_PC1',
    'L6_LBC': 'L6_BC', 'L6_MC': 'L6_MC', 'L6_NBC': 'L6_BC', 'L6_NGC': 'L6_UM_I', 'L6_SBC': 'L6_BC',
    'L6_TPC_L1': 'L6_PC2', 'L6_TPC_L4': 'L6_PC1', 'L6_UTPC': 'L6_PC1',
    }


# Data from NMC portal. The sum L23 is the L2 and L3 data summed together
cell_N_dict = {
    'L1_DAC': 58, 'L1_DLAC': 24, 'L1_HAC': 91, 'L1_NGC-SA': 52, 'L1_NGC-DA': 72, 'L1_SLAC': 41, 'L23_DBC': 106+69,
    'L23_BP': 16+12, 'L23_LBC': 277+179, 'L23_NGC': 34+22, 'L23_NBC': 160+108, 'L23_SBC': 99+67, 'L23_ChC': 37+24,
    'L23_BTC': 63+41, 'L23_PC': 2421+3456, 'L23_MC': 202+131, 'L4_ChC': 8, 'L4_NBC': 96, 'L4_LBC': 122, 'L4_MC': 118,
    'L4_SS': 406, 'L4_SBC': 60, 'L4_DBC': 40, 'L4_SP': 1098, 'L4_PC': 2674, 'L4_BTC': 20, 'L4_BP': 8, 'L4_NGC': 6,
    'L5_DBC': 96, 'L5_BP': 34, 'L5_LBC': 210, 'L5_STPC': 302, 'L5_NGC': 8, 'L5_SBC': 25, 'L5_NBC': 201, 'L5_ChC': 19,
    'L5_BTC': 76, 'L5_TTPC1': 2403, 'L5_MC': 395, 'L5_UTPC': 342, 'L5_TTPC2': 2003, 'L6_MC': 336, 'L6_ChC': 16,
    'L6_SBC': 67, 'L6_NGC': 17, 'L6_LBC': 463, 'L6_BTC': 54, 'L6_NBC': 198, 'L6_BPC': 3174, 'L6_IPC': 3476,
    'L6_TPC_L1': 1637, 'L6_DBC': 31, 'L6_TPC_L4': 1440, 'L6_UTPC': 1735, 'L6_BP': 7}

own_cell_groups=set(cell_group_dict.values())

# Get a list of Markram neuron groups mathing own search group

markram_groups_tmp=[]
own_Ncell_dict={}
for own_cell_group in own_cell_groups:
    [markram_groups_tmp.append(name) for name, group in cell_group_dict.items() if group == own_cell_group] # search cell_group_dict by value
    Ncells = sum([cell_N_dict[N] for N in markram_groups_tmp]) # sum N cells in the groups
    own_Ncell_dict[own_cell_group]=Ncells
    markram_groups_tmp=[]

#Assign all unmarked inhibitory neurons to basket and Martinotti cells in their respective relations
layers=['23','4','5','6']
markram_groups_tmp=[]
for layer in layers:
    UM_I_key='L{}_UM_I'.format(layer)
    Ncells = own_Ncell_dict[UM_I_key]
    # Calculate MC/BC relations
    MC_key = 'L{}_MC'.format(layer)
    N_MC_cells=own_Ncell_dict[MC_key]
    BC_key = 'L{}_BC'.format(layer)
    N_BC_cells=own_Ncell_dict[BC_key]
    proportion_MC = N_MC_cells / (N_MC_cells + N_BC_cells)
    proportion_BC = N_BC_cells / (N_MC_cells + N_BC_cells)
    N_MC_cells_final = N_MC_cells + round(proportion_MC * Ncells)
    N_BC_cells_final = N_BC_cells + round(proportion_BC * Ncells)
    # assign back to own cell groups and delete um keys
    own_Ncell_dict[MC_key] = int(N_MC_cells_final)
    own_Ncell_dict[BC_key] = int(N_BC_cells_final)
    del own_Ncell_dict[UM_I_key] # remove now unnecessary UM_I_key

ordered_own_Ncell_list=sorted(own_Ncell_dict.items())
#check the sum
# for layer in layers:
#     print layer
#     print sum(list(v for k, v in own_Ncell_dict.iteritems() if layer in k.lower()))
# cellsum=0
# for group_tuple in ordered_own_Ncell_list:
#     cellsum += group_tuple[1]

# Here you have all the N neurons
print ordered_own_Ncell_list
# print cellsum

# # And now for the synapses
# basepath = 'C:\Users\Simo\Laskenta\Models\Markram'
# filename = 'pathways_anatomy_factsheets_simplified.json'
# fullname = os.path.join(basepath, filename)
#
#
# with open(fullname) as data_file:
#     data = json.load(data_file)
#
# # Names second level
# flags_second_level = {'common_neighbor_bias': 0, 'connection_probability': 0,
#                       'mean_number_of_synapse_per_connection': 0, 'number_of_convergent_neuron_mean': 0,
#                       'number_of_convergent_neuron_std': 0, 'number_of_divergent_neuron_mean': 0,
#                       'number_of_divergent_neuron_std': 0, 'number_of_synapse_per_connection_std': 0,
#                       'total_synapse_count': 0}
#
# # TODO  and synapses for own cell groups
#
# search_pre_layer = '^L23'
# search_post_layer = ':L23'
# search_pre_group = '_NBC:'
# search_post_group = '_NBC$'
# search_any_pre_group = '^.*:'
#
# count = 0
# all_matches=[]
#
# for i in range(len(data.keys())):
#     # for j in range(len(data.keys()[i])):
#     # match=re.search(search_pattern, data.keys()[i][j])
#     # match = re.search(search_pre_layer, data.keys()[i])
#     # match = re.search(search_post_layer, data.keys()[i])
#     # match = re.search(search_pre_group, data.keys()[i])
#     # match = re.search(search_post_group, data.keys()[i])
#     match = re.search(search_any_pre_group, data.keys()[i])
#     if match:
#         all_matches.append(match.group()[:-1])
#         # print str(match.group())
#         # print data.keys()[i]
#         # pprint(data.values()[i])
#         # break
#         count += 1
# # print all_matches
# group_names = set(all_matches)
#
# sorted_group_names =list(group_names)
# sorted_group_names.sort()
# # print sorted_group_names
# print 'count is = %s' % str(count)
# print 'N group names = %s' % str(len(group_names))
# print 'Group names are: \n'
# for gn in sorted_group_names:
#     print gn + ', '