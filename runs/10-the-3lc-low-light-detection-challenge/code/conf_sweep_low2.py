import sys
import conf_sweep
for conf in (0.005, 0.01):
    conf_sweep.run_conf(sys.argv[1], conf)
