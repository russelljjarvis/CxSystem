import cortical_system as CX
import os
from brian_genn_version  import *


CM = CX.cortical_system (os.path.dirname(os.path.realpath(__file__)) + '/Markram_config_file.csv' , os.getcwd(), result_filename ='Custom_Name.mat',use_genn=0,runtime=1000*ms )



# you can either use the "runtime" argument in definition of the CM or remove that and uncomment the following lines:
# run(500*ms,report = 'text')
# if CM.use_genn == 1:
#     device.build(directory=os.path.join(CM.save_path, 'GeNN_Output'),
#                  compile=True,
#                  run=True,
#                  use_GPU=True)
# CM.gather_result()


# some drawings:
for group in CM.monitor_name_bank:
    mon_num = len(CM.monitor_name_bank[group])
    exec "f, axarr = plt.subplots(%d, sharex=True)" % mon_num
    for item_idx, item in enumerate(CM.monitor_name_bank[group]):
        if 'SpMon' in item:
            if len(CM.monitor_name_bank[group]) == 1:
                exec "axarr.plot(%s.t/ms,%s.i,'.k')" % (item, item);
                exec "axarr.set_title('%s')" % (item);
            else:
                exec "axarr[%d].plot(%s.t/ms,%s.i,'.k')" % (item_idx, item, item)
                exec "axarr[%d].set_title('%s')" % (item_idx, item)
        elif 'StMon' in item:
            underscore = item.index('__')
            variable = item[underscore + 2:]
            exec 'y_num=len(%s.%s)' % (item, variable)
            try:
                exec "CM.multi_y_plotter(axarr[%d] , y_num , '%s',%s , '%s')" % (item_idx, variable, item, item)
            except:
                exec "CM.multi_y_plotter(axarr , y_num , '%s',%s , '%s')" % (variable, item, item)
show()