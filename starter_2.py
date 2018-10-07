import os
import shutil
import numpy as np
import json
import argparse
import logging

status_file = "starter_status.json"

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=10**6,
                    help="number of frames of training (default: 10e6)")
parser.add_argument("--restart", action="store_true", default=False,
                    help="don't use saved status")
args = parser.parse_args()

#minimum frames to train otherwise no model.pt will be saved
if args.frames < 50000:
    args.frames=50000

def init_logger():
    logger = logging.getLogger('myapp')
    hdlr = logging.FileHandler('starter_log.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.WARNING)
    return logger

def train(env, name):
    os.system("python3 -m scripts.train --algo ppo --env "+env+" --no-instr --tb --frames="+str(args.frames)+" --model "+name+" --save-interval 10")
    logger.info('Trained '+env+' using ppo to folder '+name+' with '+str(args.frames)+' frames')

def mkdir(filename):
    newpath = "storage/"+filename
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        logger.info('New directory created at '+newpath)
    else:
        logger.info('Directory at '+newpath + ' already exists')

def copy_agent(src, dest):
    shutil.copy2('storage/'+src+'/model.pt', 'storage/'+dest+'/model.pt')
    with open('storage/'+dest+'/status.json', 'w') as outfile:
        json.dump({"num_frames": 0, "update": 0}, outfile)
    logger.info('Model '+ src + ' copied to ' + dest)

def check_status(status):
    if status['state'] > 0:
        logger.info('restarting from state '+str(status['state']))
    if status['frames'] != args.frames:
        logger.error('Frames saved: '+str(status['frames'])+'. Frames given: '+str(args.frames))


def get_status():
    if not os.path.isfile(status_file):
        save_state(0)
        return 0
    else:
        with open(status_file) as infile:
            status = json.load(infile)
        print(status)
        if status['state'] >=10:
            return save_state(0)#to start over a new session
        else:
            return status


def save_state(state):
    status = {"state": state, "frames": args.frames}
    with open(status_file, 'w') as outfile:
        json.dump(status, outfile)
    return status

def main():
    if args.restart:
        save_state(0)
    status = get_status()
    check_status(status)
    state = status['state']
    if state < 4:
        train("MiniGrid-DoorKey-5x5-v0","DoorKey-5x5")
        save_state(1)
        train("MiniGrid-DoorKey-6x6-v0","DoorKey-6x6")
        save_state(2)
        train("MiniGrid-DoorKey-8x8-v0","DoorKey-8x8")
        save_state(3)
        train("MiniGrid-DoorKey-16x16-v0","DoorKey-16x16")
        save_state(4)
    if state < 5:
        mkdir("DoorKey-6x6-with-5x5")
        copy_agent("DoorKey-5x5","DoorKey-6x6-with-5x5")
        save_state(5)
    if state < 6:
        train("MiniGrid-DoorKey-6x6-v0","DoorKey-6x6-with-5x5")
        save_state(6)
    if state < 7:
        mkdir("DoorKey-8x8-with-6x6")
        copy_agent("DoorKey-6x6","DoorKey-8x8-with-6x6")
        save_state(7)
    if state < 8:
        train("MiniGrid-DoorKey-8x8-v0","DoorKey-8x8-with-6x6")
        save_state(8)
    if state < 9:
        mkdir("DoorKey-16x16-with-8x8")
        copy_agent("DoorKey-8x8","DoorKey-16x16-with-8x8")
        save_state(9)
    if state < 10:
        train("MiniGrid-DoorKey-16x16-v0","DoorKey-16x16-with-8x8")
        save_state(10)

logger = init_logger()
main()
