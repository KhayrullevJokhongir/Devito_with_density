import subprocess
  
pytonProcess = subprocess.check_output("ps ax | grep Ac_wave_sim_hetero_media.py",shell=True).decode()
pytonProcess = pytonProcess.split('\n')
  
for process in pytonProcess:
    print(process) 