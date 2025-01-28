# from tensorboard.backend.event_processing import event_accumulator
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import subprocess
import time

# create a python file that launches a tensorboard server for each subdirectory in a folder
directory = 'v2/log/GANASTRO'
starting_port = 9200
base_cmd = ['tensorboard', '--logdir']
for idx, folder in enumerate(os.listdir(directory)):
    entry_point = os.path.join(directory, folder)
    cmd = base_cmd + [entry_point] + ['--bind_all'] + ['--port', starting_port + idx], ['&']
    print(cmd)
    subprocess.call(" ".format(chunk for chunk in cmd), shell=True)
    print(f"ok {starting_port + idx}")

while True:
    time.sleep(1)