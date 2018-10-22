import os
import shutil
import numpy as np
import json
import argparse
import logging

restart=False
enjoy_cmds = []

#env and name are full env and full model name
def train(env, model):
    global logger
    os.system("python3 -m scripts.train --algo ppo --env "+env+" --no-instr --tb --frames="+str(frames)+" --model "+model+" --save-interval 10")
    logger.info('Trained '+env+' using ppo to folder '+model+' with '+str(frames)+' frames')
    enjoy_cmds.append('python3 -m scripts.enjoy --env ' + env + ' --model ' + model)

#if there is not a folder, creates it
def mkdir(filename):
    newpath = "storage/"+filename
    if not os.path.exists(newpath):
        os.makedirs(newpath)

#if we have to restart or we don't have that model, we copy it
def copy_agent(src, dest):
    global logger
    mkdir(dest)
    not_exists = not os.path.isfile('storage/'+src+'/model.pt')
    if not_exists or restart:
        shutil.copy2('storage/'+src+'/model.pt', 'storage/'+dest+'/model.pt')
        with open('storage/'+dest+'/status.json', 'w') as outfile:
            json.dump({"num_frames": 0, "update": 0}, outfile)        
        logger.info('Copied '+src + ' to '+dest)

#initializes logger
def init_logger():
    logger = logging.getLogger('myapp')
    hdlr = logging.FileHandler('starter_log.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.WARNING)
    return logger

#saves commands to enjoy all trained models
def save_enjoy_commands():
    with open('cmds_full_curriculum.txt', 'w') as outenjoy:
        for cmd in enjoy_cmds:
            outenjoy.write("%s\n" % cmd)

def main():
    global frames
    frames=2*10**6
    train("MiniGrid-DoorKey-5x5-v0","DoorKey-5x5")
    train("MiniGrid-DoorKey-6x6-v0","DoorKey-6x6")
    train("MiniGrid-DoorKey-8x8-v0","DoorKey-8x8")
    train("MiniGrid-DoorKey-10x10-v0","DoorKey-10x10")
    frames=3*10**6
    train("MiniGrid-DoorKey-12x12-v0","DoorKey-12x12")
    train("MiniGrid-DoorKey-14x14-v0","DoorKey-14x14")
    train("MiniGrid-DoorKey-16x16-v0","DoorKey-16x16")
    #end simple training for DoorKey

    frames=2*10**6
    copy_agent("DoorKey-5x5","DoorKey-6x6-with-5x5")
    train("MiniGrid-DoorKey-6x6-v0","DoorKey-6x6-with-5x5")
    #remains the same frame number
    copy_agent("DoorKey-6x6-with-5x5","DoorKey-8x8-with-6x6")
    train("MiniGrid-DoorKey-8x8-v0","DoorKey-8x8-with-6x6")
    #remains the same frame number
    copy_agent("DoorKey-8x8-with-6x6","DoorKey-10x10-with-8x8")
    train("MiniGrid-DoorKey-10x10-v0","DoorKey-10x10-with-8x8")
    frames=3*10**6
    copy_agent("DoorKey-10x10-with-8x8","DoorKey-12x12-with-10x10")
    train("MiniGrid-DoorKey-12x12-v0","DoorKey-12x12-with-10x10")
    frames=9*10**6
    copy_agent("DoorKey-12x12-with-10x10","DoorKey-14x14-with-12x12")
    train("MiniGrid-DoorKey-14x14-v0","DoorKey-14x14-with-12x12")
    #remains the same frame number
    copy_agent("DoorKey-14x14-with-12x12","DoorKey-16x16-with-14x14")
    train("MiniGrid-DoorKey-16x16-v0","DoorKey-16x16-with-14x14")
    #End curriculum training for DoorKey

    save_enjoy_commands()

logger = init_logger()
main()
