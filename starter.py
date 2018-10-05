import os
import shutil
import numpy as np
import json

frames = 5000000

def train(env, name):
  os.system("python3 -m scripts.train --algo ppo --env "+env+" --no-instr --tb --frames="+str(frames)+" --model "+name+" --save-interval 10")

def mkdir(filename):
    newpath = "storage/"+filename
    if not os.path.exists(newpath):
        os.makedirs(newpath)

def copy_agent(src, dest):
    shutil.copy2('storage//'+src+'//model.pt', 'storage//'+dest+'//model.pt')
    with open('storage//'+dest+'//status.json', 'w') as outfile:
        json.dump({"num_frames": 0, "update": 0}, outfile)

def main():
    train("MiniGrid-DoorKey-5x5-v0","DoorKey-5x5")
    train("MiniGrid-DoorKey-6x6-v0","DoorKey-6x6")
    train("MiniGrid-DoorKey-8x8-v0","DoorKey-8x8")
    train("MiniGrid-DoorKey-16x16-v0","DoorKey-16x16")

    mkdir("DoorKey-6x6-with-5x5")
    copy_agent("DoorKey-5x5","DoorKey-6x6-with-5x5")
    train("MiniGrid-DoorKey-6x6-v0","DoorKey-6x6-with-5x5")

    mkdir("DoorKey-8x8-with-6x6")
    copy_agent("DoorKey-6x6","DoorKey-8x8-with-6x6")
    train("MiniGrid-DoorKey-8x8-v0","DoorKey-8x8-with-6x6")

    mkdir("DoorKey-16x16-with-8x8")
    copy_agent("DoorKey-8x8","DoorKey-16x16-with-8x8")
    train("MiniGrid-DoorKey-16x16-v0","DoorKey-16x16-with-8x8")

main()
