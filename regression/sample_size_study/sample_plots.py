import pandas as pd
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.plotting import get_plots_size_sample

get_plots_size_sample('termination_time')
get_plots_size_sample('efficiency')
get_plots_size_sample('lifetime')