import os, glob
import subprocess
os.environ['CUDA_VISIBLE_DEVICES']='0'
import fire
import random as rnd
from big_spose_sleep import Imagine
from pathlib import Path
import sys


if __name__=="__main__":

    reps = 5
    sposeprofiles = glob.glob( "*profile.txt" )

    for ri in range(1, reps): 
        for sposeprofilefn in sposeprofiles: 
            print(sposeprofilefn)
            subprocess.call( "/LOCAL/kamue/anaconda3/bin/python start_test_run.py " + sposeprofilefn )
            os.rename( 'spose_'+sposeprofilefn+".png", sposeprofilefn + "_" + str(ri) + ".png" )