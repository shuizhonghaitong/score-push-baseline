import os
print("CWD:", os.getcwd(), flush=True)
print("PWD env:", os.environ.get("PWD"), flush=True)
print("argv0-ish executable:", os.path.realpath(__file__), flush=True)
try:
    os.makedirs("wd_test_dir", exist_ok=True)
    print("cwd writable: True", flush=True)
except Exception as e:
    print("cwd writable: False", repr(e), flush=True)
