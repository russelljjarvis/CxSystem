import cortical_system as CX
from brian2 import *
import multiprocessing
import time
import shutil
import os

class array_run(object):

    def __init__(self,anatomy_df,physiology_df):
        self.anatomy_df = anatomy_df
        self.physiology_df =physiology_df
        try :
            self.multidimension_array_run = int(self.parameter_finder(self.anatomy_df,'multidimension_array_run'))
        except TypeError:
            self.multidimension_array_run = 0
        anatomy_array_search_result = anatomy_df[anatomy_df.applymap(lambda x: True if ('|' in str(x) or '&' in str(x)) else False)]
        physio_array_search_result = physiology_df[physiology_df.applymap(lambda x: True if ('|' in str(x) or '&' in str(x)) else False)]
        arrays_idx_anatomy = where(anatomy_array_search_result.isnull().values != True)
        arrays_idx_anatomy = [(arrays_idx_anatomy[0][i],arrays_idx_anatomy[1][i]) for i in range(len(arrays_idx_anatomy[0]))]
        arrays_idx_physio = where(physio_array_search_result.isnull().values != True)
        arrays_idx_physio = [(arrays_idx_physio[0][i], arrays_idx_physio[1][i]) for i in range(len(arrays_idx_physio[0]))]
        self.df_anat_final_array = []
        self.df_phys_final_array = []
        self.final_messages = []
        self.sum_of_array_runs = len(arrays_idx_anatomy) + len(arrays_idx_physio)
        if self.sum_of_array_runs > 1 and not self.multidimension_array_run:
            anatomy_default = self.df_default_finder(anatomy_df)
            physio_default = self.df_default_finder(physiology_df)
        else:
            anatomy_default = anatomy_df
            physio_default = physiology_df
        if self.multidimension_array_run:
            if arrays_idx_anatomy:
                anat_variations, anat_messages = self.df_builder_for_array_run( anatomy_df, arrays_idx_anatomy)
            if arrays_idx_physio:
                physio_variations, physio_messages = self.df_builder_for_array_run( physiology_df, arrays_idx_physio)
            if arrays_idx_anatomy and arrays_idx_physio:
                for anat_idx, anat_df in enumerate(anat_variations):
                    for physio_idx, physio_df in enumerate(physio_variations):
                        self.df_anat_final_array.append(anat_df)
                        self.df_phys_final_array.append(physio_df)
                        self.final_messages.append(anat_messages[anat_idx]+physio_messages[physio_idx])
            elif arrays_idx_anatomy:
                for anat_idx, anat_df in enumerate(anat_variations):
                    self.df_anat_final_array.append(anat_df)
                    self.df_phys_final_array.append(physio_default)
                    self.final_messages.append(anat_messages[anat_idx])
            elif arrays_idx_physio:
                for physio_idx, physio_df in enumerate(physio_variations):
                    self.df_phys_final_array.append(physio_df)
                    self.final_messages.append(physio_messages[physio_idx])


        else:
            if arrays_idx_anatomy:
                df_anat_array, anat_messages = self.df_builder_for_array_run(anatomy_df, arrays_idx_anatomy)
                self.df_anat_final_array.extend(df_anat_array)
                self.df_phys_final_array.extend([physio_default for _ in range(len(self.df_anat_final_array))])
                self.final_messages.extend(anat_messages)
            if arrays_idx_physio:
                df_phys_array, physio_messages = self.df_builder_for_array_run(physiology_df, arrays_idx_physio)
                self.df_phys_final_array.extend(df_phys_array)
                self.df_anat_final_array.extend([anatomy_default for _ in range(len(self.df_phys_final_array))])
                self.final_messages.extend(physio_messages)

        print "Info: array of Dataframes for anatomical and physiological configuration are ready"
        self.spawner()

    def arr_run(self,idx, working):
        working.value += 1
        np.random.seed(idx)
        idx = idx/self.trials_per_config
        tr = idx % self.trials_per_config
        print "################### Trial %d started running for simulation number %d: %s ##########################" % (tr ,idx,self.final_messages[idx][1:])
        cm = CX.cortical_system(self.df_anat_final_array[idx],self.df_phys_final_array[idx],output_file_suffix = self.final_messages[idx])
        cm.run()
        if self.number_of_process ==1 and self.do_benchmark == 1 and self.device == 'Python':
            # this should be used to clear the cache of weave for benchmarking. otherwise weave will mess it up
            shutil.rmtree(os.path.join(os.environ['HOME'],'.cache/scipy'))

        working.value -= 1

    def spawner(self):
        try:
            self.number_of_process = int(self.parameter_finder(self.anatomy_df, 'number_of_process'))
        except TypeError:
            self.number_of_process = int(multiprocessing.cpu_count() * 3 / 4)
            print "\nWarning: number_of_process is not defined in the configuration file, the default number of processes are 3/4*number of CPU cores: %d processes\n" % self.number_of_process
        try:
            self.do_benchmark = int(self.parameter_finder(self.anatomy_df,'do_benchmark'))
        except TypeError:
            self.do_benchmark = 0
        try:
            self.trials_per_config = int(self.parameter_finder(self.anatomy_df,'trials_per_config'))
        except TypeError:
            self.trials_per_config = 1
        try:
            self.device = self.parameter_finder(self.anatomy_df, 'device')
        except TypeError:
            self.device = 'Python'

        print "following configurations are going to be simulated with %d processes using %s device (printed only in letters and numbers): " \
              "\n %s"%(self.number_of_process,self.device,str(self.final_messages).replace('_',''))
        manager = multiprocessing.Manager()
        jobs = []
        working = manager.Value('i', 0)
        number_of_runs = len(self.final_messages) * self.trials_per_config
        assert len(self.final_messages) < 1000 , 'The array run is trying to run more than 1000 simulations, this is not allowed unless you REALLY want it and if you REALLY want it you should konw what to do.'
        while len(jobs) < number_of_runs:
            time.sleep(0.3)
            if working.value < self.number_of_process:
                p = multiprocessing.Process(target=self.arr_run, args=(len(jobs), working,))
                jobs.append(p)
                p.start()
        for j in jobs:
            j.join()

    def parameter_finder(self,df,keyword):
        location = where(df.values == keyword)
        if location:
            counter = int(location[0])+1
            while counter < df.shape[0] :
                if '#' not in str(self.anatomy_df.ix[counter][int(location[1])]):
                    value = self.anatomy_df.ix[counter][int(location[1])]
                    break
                else:
                    counter+=1
            return value

    def df_builder_for_array_run(self, original_df, index_of_array_variable,message=''):
        array_of_dfs = []
        run_messages = []
        array_variable = original_df.ix[index_of_array_variable[0][0]][index_of_array_variable[0][1]]
        opening_braket_idx = array_variable.index('{')
        if (not self.multidimension_array_run and self.sum_of_array_runs>1) or (self.sum_of_array_runs==1 and ':' in array_variable):
            colon_idx = array_variable.index(':')
            array_variable = array_variable.replace(array_variable[opening_braket_idx + 1:colon_idx + 1],'') # removing default value
        elif ':' in array_variable:
            print "\nWarning: the default value set for %s is omitted since the array run is multidimentional (multidimension_array_run flag is set to 1)\n" %array_variable
            colon_idx = array_variable.index(':')
            array_variable = array_variable.replace(array_variable[opening_braket_idx + 1:colon_idx + 1], '')  # removing default value
        closing_braket_idx = array_variable.index('}')
        template_of_variable = array_variable[:opening_braket_idx] + '^^^' + array_variable[closing_braket_idx + 1:]
        assert not ('|' in array_variable and '&' in array_variable), "The following array run should be defined either using | or & not both of them:" %array_variable
        if '|' in array_variable :
            changing_part = array_variable[opening_braket_idx + 1:closing_braket_idx].replace('|', ',')
            tmp_str = 'arange(' + changing_part + ')'
            variables_to_iterate = eval(tmp_str)
        elif '&' in array_variable:
            variables_to_iterate = eval('[' + array_variable[opening_braket_idx + 1:closing_braket_idx].replace('&', ',') + ']')
        variables_to_iterate = [template_of_variable.replace('^^^', str(vv)) for vv in variables_to_iterate]
        for var in variables_to_iterate:
            temp_df = original_df.copy()
            temp_df.ix[index_of_array_variable[0][0]][index_of_array_variable[0][1]] = var
            if self.multidimension_array_run and len(index_of_array_variable)>1:
                tmp_message = self.message_finder(temp_df, index_of_array_variable)
                temp_df, messages = self.df_builder_for_array_run(temp_df, index_of_array_variable[1:],tmp_message)
            else:
                temp_df = [self.df_default_finder(temp_df)]
                messages = [message+self.message_finder(temp_df[0], index_of_array_variable)]
            array_of_dfs.extend(temp_df)
            run_messages.extend(messages)
        if not self.multidimension_array_run and len(index_of_array_variable)>1:
            temp_df, messages = self.df_builder_for_array_run(original_df, index_of_array_variable[1:])
            array_of_dfs.extend(temp_df)
            run_messages.extend(messages)
        return array_of_dfs, run_messages


    def df_default_finder(self,df_):
        df = df_.copy()
        df_search_result = df[df.applymap(lambda x: True if ('|' in str(x) or '&' in str(x)) else False)]
        df_search_result = where(df_search_result.isnull().values == False)
        arrays_idx_ = [(df_search_result[0][i], df_search_result[1][i]) for i in range(len(df_search_result[0]))]
        for to_default_idx in arrays_idx_:
            value_to_default = df.ix[to_default_idx[0]][to_default_idx[1]]
            assert ':' in value_to_default, "The default value should be defined for %s , or make sure multidimension_array_run in configuraiton file is set to 1." % value_to_default
            default = value_to_default[value_to_default.index('{')+1:value_to_default.index(':')]
            df.ix[to_default_idx[0]][to_default_idx[1]] = df.ix[to_default_idx[0]][to_default_idx[1]].replace(value_to_default[value_to_default.index('{'):value_to_default.index('}')+1],default)
        return df

    def message_finder(self,df, idx):
        idx = idx[0]
        whitelist = set('abcdefghijklmnopqrstuvwxyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890._-')
        try:
            if any(df[0].str.contains('row_type')):
                definition_rows_indices = array(df[0][df[0] == 'row_type'].index.tolist())
                target_row = max(where(definition_rows_indices < idx[0])[0])
                title = str(df.loc[target_row][idx[1]])
                value = str(df.loc[idx[0]][idx[1]])
                message = '_' + title + ''.join(filter(whitelist.__contains__, value))
        except KeyError:
            if 'Variable' in df.columns:
                try:
                    if not math.isnan(df['Key'][idx[0]]):
                        title = str(df['Key'][idx[0]])
                        value = str(df[idx[0]][idx[1]])
                    else:
                        title = str(df['Variable'][idx[0]])
                        value = str(df.ix[idx[0]][idx[1]])
                except TypeError:
                    title = str(df['Key'][idx[0]])
                    value = str(df.ix[idx[0]][idx[1]])
                message = '_' + title + ''.join(filter(whitelist.__contains__, value))
        return message

