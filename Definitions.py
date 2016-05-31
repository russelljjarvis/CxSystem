__author__ = 'V_AD'
from brian2 import *
from namespaces import *


class customized_neuron(object):
    '''Using this class you will get a dictionary containing all parameters and variables that are needed to \
    create a group of that customized cell. This dictionary will eventually be used to build the cortical module.'''
    # This vairable is to keep track of all customized neurons do be able to draw it
    def __init__(self, number_of_neurons,  cell_type,  layers_idx , resolution = 0.1, network_center = 0 + 0j):
        '''
        :param cell_type: type of cell which is either PC, SS, BC, MC, Inh[?].
        :type cell_type: string
        :param layers_idx: This tuple numpy array defines the first and last layers in which the neuron resides. So array([4,1]) means that the\
        soma resides in layer 4 and the apical dendrites which are (2 compartments) extend to layer 2/3 and 1. To avoid confusion, layer 2\
        is used as the indicator of layer 2/3. Hence, if the last compartment of a neuron is in layer 2/3, use number 2.
        :type layers_idx: numpy array
        '''
        # check cell_type
        customized_neuron._celltypes = array(['PC', 'SS', 'BC', 'MC','L1i'])
        assert cell_type in customized_neuron._celltypes, "Error: cell type '%s' is not defined" %cell_type
        # check layers_idx
        assert len(layers_idx) < 3 , "Error: length of layers_idx array is larger than 2"
        if len (layers_idx) == 2 :
            assert layers_idx[1] < layers_idx [0] , "Error: indices of the layer_idx array are not descending"
        elif len (layers_idx) == 1 :
            assert cell_type != 'PC' , "Cell type is PC but the start and end of the neuron is not defined in layers_idx"
        # final neuron is the output neuron containing equation, parameters , etc TODO
        self.output_neuron = {}
        self.output_neuron['number_of_neurons'] = int(number_of_neurons)
        self.output_neuron['threshold']  = 'vm>Vcut'
        self.output_neuron['reset'] = 'vm=V_res'
        self.output_neuron['refractory'] =  '4 * ms'
        self.output_neuron['type'] = cell_type
        self.output_neuron['soma_layer'] = int(layers_idx[0])
        # _comparts_tmp1 & 2 are for extracting the layer of the compartments if applicable
        if self.output_neuron['type'] == 'PC':
            self._comparts_tmp1 = array(range (layers_idx[0]-1,layers_idx[1]-1,-1))
            self._comparts_tmp2 = delete(self._comparts_tmp1,where(self._comparts_tmp1==3)) if 3 in self._comparts_tmp1 else self._comparts_tmp1
            self.output_neuron['dends_layer'] = self._comparts_tmp2
            self.output_neuron['dend_comp_num'] = len (self.output_neuron['dends_layer'])
            self.output_neuron['total_comp_num'] = self.output_neuron['dend_comp_num'] + 3
        else:
            self.output_neuron['dends_layer'] = self.output_neuron['soma_layer']
            self.output_neuron['dend_comp_num'] = array([0])
            self.output_neuron['total_comp_num'] = array([1])
            # number of compartments if applicable

        self.output_neuron['namespace'] = neuron_namespaces(self.output_neuron).output_namespace
        self.output_neuron['equation'] = ''
        getattr(self, '_' + self.output_neuron['type'])()
        _M_V1 = 2.3
        _dx = _M_V1* resolution
        _grid_size = sqrt(self.output_neuron['number_of_neurons'])*_dx
        self.output_neuron['positions'] = self._get_positions(self.output_neuron['number_of_neurons'],_grid_size,1,'array', network_center)
        print "Customized " + str(cell_type) + " neuron in layer "+ str(layers_idx) + " initialized"


    def _get_positions(self, N, grid_size, scale, layout, networkcenter):

        _pos_ndx=array(range(N))

        # print N, grid_size, scale, layout

    #        _positions = empty((N, 2))

        _side = int(sqrt(N))
        if layout == 'array':
            _positions = (_pos_ndx/_side) + 1j * ( _pos_ndx%_side)
            _positions = _positions/float(_side)*grid_size + networkcenter #SV change 030915
            print "grid_size: " + str(grid_size)

        else:
            if layout == 'random':
                _positions=grid_size*numpy.random.rand(N, 2)

        return (_positions-0.5*grid_size*(1+1j))



    def _PC(self):
        '''
        :param namespace_type: defines the category of the equation.
        :type namespace_type: str
        :param n_comp: number of compartments in the neuron
        :type n_comp: int
        :param layer_idx: indices of the layers in which neuron resides.
        :type layer_idx: array
        :param eq_template_soma: Contains template somatic equation used in Brian2.

        ::

            dgeX/dt = -geX/tau_eX : siemens
            dgealphaX/dt = (geX-gealphaX)/tau_eX : siemens
            dgi/dt = -gi/tau_i : siemens
            dgialpha/dt = (gi-gialpha)/tau_i : siemens

        :param eq_template_dend: Contains template somatic equation used in Brian2.
        :type eq_template_dend: str
        :param test_param: something here
        :type test_param: some type here
        '''

        #: The template for the somatic equations used in multi compartmental neurons, the inside values could be replaced later using "Equation" function in brian2.
        eq_template_soma = '''
        dvm/dt = (gL*(EL-vm) + gealpha * (Ee-vm) + gealphaX * (Ee-vm) + gialpha * (Ei-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) +I_dendr) / C : volt (unless refractory)
        dge/dt = -ge/tau_e : siemens
        dgealpha/dt = (ge-gealpha)/tau_e : siemens
        dgeX/dt = -geX/tau_eX : siemens
        dgealphaX/dt = (geX-gealphaX)/tau_eX : siemens
        dgi/dt = -gi/tau_i : siemens
        dgialpha/dt = (gi-gialpha)/tau_i : siemens
        '''
        #: The template for the dendritic equations used in multi compartmental neurons, the inside values could be replaced later using "Equation" function in brian2.
        eq_template_dend = '''
        dvm/dt = (gL*(EL-vm) + gealpha * (Ee-vm) + gealphaX * (Ee-vm) + gialpha * (Ei-vm) +I_dendr) / C : volt
        dge/dt = -ge/tau_e : siemens
        dgealpha/dt = (ge-gealpha)/tau_e : siemens
        dgeX/dt = -geX/tau_eX : siemens
        dgealphaX/dt = (geX-gealphaX)/tau_eX : siemens
        dgi/dt = -gi/tau_i : siemens
        dgialpha/dt = (gi-gialpha)/tau_i : siemens
        '''

        self.output_neuron['equation'] = Equations(eq_template_dend, vm="vm_basal", ge="ge_basal", gealpha="gealpha_basal",
                                         C=self.output_neuron['namespace']['C'][0], gL=self.output_neuron['namespace']['gL'][0],
                                         gi="gi_basal", geX="geX_basal", gialpha="gialpha_basal",
                                         gealphaX="gealphaX_basal", I_dendr="Idendr_basal")
        self.output_neuron['equation'] += Equations(eq_template_soma, gL=self.output_neuron['namespace']['gL'][1],
                                         ge='ge_soma', geX='geX_soma', gi='gi_soma', gealpha='gealpha_soma',
                                         gealphaX='gealphaX_soma',
                                         gialpha='gialpha_soma', C=self.output_neuron['namespace']['C'][1],
                                         I_dendr='Idendr_soma')
        for _ii in range(self.output_neuron['dend_comp_num'] + 1):  # extra dendritic compartment in the same level of soma
            self.output_neuron['equation'] += Equations(eq_template_dend, vm="vm_a%d" % _ii,
                                             C=self.output_neuron['namespace']['C'][_ii],
                                             gL=self.output_neuron['namespace']['gL'][_ii], ge="ge_a%d" % _ii,
                                             gi="gi_a%d" % _ii, geX="geX_a%d" % _ii,
                                             gealpha="gealpha_a%d" % _ii, gialpha="gialpha_a%d" % _ii,
                                             gealphaX="gealphaX_a%d" % _ii, I_dendr="Idendr_a%d" % _ii)

        # basal self connection
        self.output_neuron['equation'] += Equations('I_dendr = gapre*(vmpre-vmself)  : amp',
                                         gapre=1 / (self.output_neuron['namespace']['Ra'][0]),
                                         I_dendr="Idendr_basal", vmself="vm_basal", vmpre="vm")
        self.output_neuron['equation'] += Equations('I_dendr = gapre*(vmpre-vmself)  + gapost*(vmpost-vmself) : amp',
                                         gapre=1 / (self.output_neuron['namespace']['Ra'][1]),
                                         gapost=1 / (self.output_neuron['namespace']['Ra'][0]),
                                         I_dendr="Idendr_soma", vmself="vm",
                                         vmpre="vm_a0", vmpost="vm_basal")
        self.output_neuron['equation'] += Equations('I_dendr = gapre*(vmpre-vmself) + gapost*(vmpost-vmself) : amp',
                                         gapre=1 / (self.output_neuron['namespace']['Ra'][2]),
                                         gapost=1 / (self.output_neuron['namespace']['Ra'][1]),
                                         I_dendr="Idendr_a0", vmself="vm_a0", vmpre="vm_a1", vmpost="vm")

        for _ii in arange(1, self.output_neuron['dend_comp_num']):
            self.output_neuron['equation'] += Equations('I_dendr = gapre*(vmpre-vmself) + gapost*(vmpost-vmself) : amp',
                                             gapre=1 / (self.output_neuron['namespace']['Ra'][_ii]),
                                             gapost=1 / (self.output_neuron['namespace']['Ra'][_ii - 1]),
                                             I_dendr="Idendr_a%d" % _ii, vmself="vm_a%d" % _ii,
                                             vmpre="vm_a%d" % (_ii + 1), vmpost="vm_a%d" % (_ii - 1))

        self.output_neuron['equation'] += Equations('I_dendr = gapost*(vmpost-vmself) : amp',
                                         I_dendr="Idendr_a%d" % self.output_neuron['dend_comp_num'],
                                         gapost=1 / (self.output_neuron['namespace']['Ra'][-1]),
                                         vmself="vm_a%d" % self.output_neuron['dend_comp_num'],
                                         vmpost="vm_a%d" % (self.output_neuron['dend_comp_num'] - 1))

        self.output_neuron['equation'] +=  Equations('''x : meter
                            y : meter''')


    def _BC(self):
        self.output_neuron['equation'] =   Equations(  '''
            dvm/dt = (gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) + ge * (Ee-vm) + gi * (Ei-vm)) / C : volt (unless refractory)
            dge/dt = -ge/tau_e : siemens  # This goes to synapses object in B2
            dgi/dt = -gi/tau_i : siemens  # This goes to synapses object in B2
            ''', ge = 'ge_soma',gi='gi_soma')

        self.output_neuron['equation'] += Equations('''x : meter
            y : meter''')


    def _L1i(self):
        self.output_neuron['equation'] =   Equations(  '''
            dvm/dt = (gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) + ge * (Ee-vm) + gi * (Ei-vm)) / C : volt (unless refractory)
            dge/dt = -ge/tau_e : siemens  # This goes to synapses object in B2
            dgi/dt = -gi/tau_i : siemens  # This goes to synapses object in B2
            ''', ge = 'ge_soma',gi='gi_soma')

        self.output_neuron['equation'] += Equations('''x : meter
            y : meter''')



    def _MC(self):
        self.output_neuron['equation'] = Equations(  '''
            dvm/dt = (gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) + ge * (Ee-vm) + gi * (Ei-vm)) / C : volt (unless refractory)
            dge/dt = -ge/tau_e : siemens  # This goes to synapses object in B2
            dgi/dt = -gi/tau_i : siemens  # This goes to synapses object in B2
            ''', ge = 'ge_soma',gi='gi_soma')

        self.output_neuron['equation'] += Equations('''x : meter
            y : meter''')
    def _SS (self):
        self.output_neuron['equation'] = Equations(  '''
            dvm/dt = (gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) + ge * (Ee-vm) + gi * (Ei-vm)) / C : volt (unless refractory)
            dge/dt = -ge/tau_e : siemens  # This goes to synapses object in B2
            dgi/dt = -gi/tau_i : siemens  # This goes to synapses object in B2
            ''', ge = 'ge_soma',gi='gi_soma')

        self.output_neuron['equation'] += Equations('''x : meter
            y : meter''')

#################
#################
################# Synapses
#################
#################



class customized_synapse(object):
    def __init__(self, receptor,pre_group_idx ,post_group_idx, syn_type,pre_type,post_type, post_comp_name='_soma'):
        customized_synapse.syntypes = array(['STDP'])
        assert syn_type in customized_synapse.syntypes, "Error: cell type '%s' is not defined" % syn_type
        self.output_synapse = {}
        self.output_synapse['type'] = syn_type
        self.output_synapse['receptor'] = receptor
        # self.output_synapse['namespace_type'] = namespace_type
        # self.output_synapse['pre_type'] = pre_group_type
        self.output_synapse['pre_group_idx'] = int(pre_group_idx)
        self.output_synapse['pre_group_type'] = pre_type
        # self.output_synapse['post_type'] = post_group_type
        self.output_synapse['post_group_idx'] = int(post_group_idx)
        self.output_synapse['post_group_type'] = post_type
        self.output_synapse['post_comp_name'] = post_comp_name
        _name_space = synapse_namespaces(self.output_synapse)
        self.output_synapse['namespace'] = {}
        self.output_synapse['namespace'] = _name_space.output_namespace
        self.output_synapse['sparseness'] = _name_space.sparseness
        self.output_synapse['ilam'] = _name_space.ilam
        getattr(self, self.output_synapse['type'])()

    def STDP(self):

        self.output_synapse['equation'] = Equations('''
            wght:siemens
            dapre/dt = -apre/taupre : siemens (event-driven)
            dapost/dt = -apost/taupost : siemens (event-driven)
            ''')

        if self.output_synapse['namespace']['Apre'] >= 0:
            self.output_synapse['pre_eq'] = '''
                        %s+=wght
                        apre += Apre * wght0 * Cp
                        wght = clip(wght + apost, 0, wght_max)
                        ''' % (self.output_synapse['receptor'] + self.output_synapse['post_comp_name'] +  '_post')
        else:
            self.output_synapse['pre_eq'] = '''
                        %s+=wght
                        apre += Apre * wght * Cd
                        wght = clip(wght + apost, 0, wght_max)
                        ''' % (self.output_synapse['receptor']+self.output_synapse['post_comp_name'] + '_post')
        if self.output_synapse['namespace']['Apost'] <= 0:
            self.output_synapse['post_eq'] = '''
                        apost += Apost * wght * Cd
                        wght = clip(wght + apre, 0, wght_max)
                        '''
        else:
            self.output_synapse['post_eq'] = '''
                        apost += Apost * wght0 * Cp
                        wght = clip(wght + apre, 0, wght_max)
                        '''
