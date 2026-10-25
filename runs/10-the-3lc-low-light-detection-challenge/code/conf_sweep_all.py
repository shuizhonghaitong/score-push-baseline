import sys
import conf_sweep
for conf in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25):
    conf_sweep.run_conf(sys.argv[1], conf)
