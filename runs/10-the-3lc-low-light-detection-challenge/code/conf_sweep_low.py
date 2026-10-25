import sys
sys.argv = [sys.argv[0], sys.argv[1]]
import conf_sweep
for conf in (0.02, 0.05, 0.08):
    conf_sweep.run_conf(sys.argv[1], conf)
