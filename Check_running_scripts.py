import subprocess
  
pytonProcess = subprocess.check_output("ps ax | grep Daat_acquis_by_ac_wave_sim_dens_func_based.py",shell=True).decode()
pytonProcess = pytonProcess.split('\n')
  
for process in pytonProcess:
    print(process) 