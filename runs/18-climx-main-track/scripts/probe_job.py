import time,sys
t0=time.time()
while time.time()-t0<60:
    time.sleep(5)
    print("alive", flush=True)
print("PROBE_DONE", flush=True)
