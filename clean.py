import common
import cfg
import os

def run(argv):
    orig    = cfg.config["paths"]["npy-orig"]
    fixed   = cfg.config["paths"]["npy-fixed"]
    shifted = cfg.config["paths"]["shifted"]
    
    for path in [orig, fixed, shifted]:
        files = common.listfiles(path, ".zip")
        for _, filename in files:
            print(filename)
            os.remove(filename)
