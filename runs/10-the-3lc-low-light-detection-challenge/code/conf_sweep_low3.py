import sys
import conf_sweep
for conf in (0.001, 0.002, 0.003):
    conf_sweep.run_conf(sys.argv[1], conf)
