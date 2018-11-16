import os
import numpy as np

size = 16
sigmas = np.array([.1,.15,.2,.25,.3,.35,.45,.55])

def execute(size, sigma):
    os.system("python3 stratified_curriculum-starter.py --name multi --N 50 --sigma "+str(sigma)+" --frames 20000000")

def main():
    for _sigma in sigmas:
        execute(size, _sigma)

main()
