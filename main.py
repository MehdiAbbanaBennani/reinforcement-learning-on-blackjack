from run import run
import time

from config import plot_parameters
from config import parameters
from config import rule_parameters

start_time = time.time()

# You can change the parameters in the config file
run(parameters=parameters,
    plot_parameters=plot_parameters,
    rules_parameters=rule_parameters)

print("--- %s seconds ---" % (time.time() - start_time))
