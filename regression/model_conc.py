from utils import get_data_from_json, data_time
import matplotlib.pyplot as plt

ml_data = get_data_from_json('performance_indicators/performance_indicators_phi_1.0.json')
data = data_time(400, ['Volume Liquid'], ml_data)
data = data_time(400,['Last Concentration'], data)
