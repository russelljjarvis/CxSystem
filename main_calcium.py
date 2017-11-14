import CxSystem as cxs
import os
from brian2  import *
import brian2tools.nmlexport
import datetime

default_runtime = 500*ms

time_start = datetime.datetime.now()
CM = cxs.CxSystem(anatomy_and_system_config = os.path.dirname(os.path.realpath(__file__)) + '/config_files/corem_config_file.csv',
                        physiology_config = os.path.dirname(os.path.realpath(__file__)) + '/config_files/Physiological_Parameters_for_calcium_June17.csv')

# CM = cxs.CxSystem(anatomy_and_system_config = os.path.dirname(os.path.realpath(__file__)) + '/config_files/Burbank_config_test.csv',
#                         physiology_config = os.path.dirname(os.path.realpath(__file__)) + '/config_files/Physiological_Parameters_for_Burbank.csv')

time_before_run = datetime.datetime.now()
set_device('neuroml2', filename="cxsystem.nml")
CM.run()
time_end = datetime.datetime.now()

duration_generation = int((time_before_run - time_start).total_seconds())
duration_simulation = int((time_end - time_before_run).total_seconds())
duration_total = int((time_end - time_start).total_seconds())

print 'Duration of network generation: %d min %d s' % (duration_generation//60, duration_generation%60)
print 'Duration of actual simulation: %d min %d s' % (duration_simulation//60, duration_simulation%60)
print 'TOTAL %d min %d s' % (duration_total//60, duration_total%60)
print '=> %d times realtime' % (duration_total*second / default_runtime)