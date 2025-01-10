import os, sys
sys.path.append(os.path.abspath('..'))

from qft import *
from qiskit import qasm2
import pandas as pd
from Enola.route import *
import time
data_list = []
n = 50
repeat_num = 10
qasm_path = "/home/gaodc/code/qft/random_p_qaoa/qasm"
code_path = "/home/gaodc/code/qft/random_p_qaoa/code"
# os.makedirs(qasm_path,exist_ok=True)
# os.makedirs(f"{code_path}/LNN",exist_ok=True)
# os.makedirs(f"{code_path}/zigzag",exist_ok=True)

for p in range(95,0,-5):
    cir = QFT(n)
    w = -1
    maps = cir.LNN_maps(w)
    cir.to_random_p_qaoa(p/100)
    for i in range(repeat_num):
        cir.export_to_qasm(f"{qasm_path}/qft{n}_{p}_{i}.qasm")
    print(f"creat {p}")
    
for p in range(95,0,-5):
    for i in range(repeat_num):
        data = {"prob": p ,"i": i}
        edges = read_edges_from_qasm(f"{qasm_path}/qft{n}_{p}_{i}.qasm")
    
        w = -1
        print(f"qft_{n} with c={w} for p={p}")
        cir = QFT(n)
        time_s = time.time()
        maps = cir.LNN_maps(w)
        cir.to_random_p_qaoa(p/100,edges)
        
        route = QuantumRouter(cir.num_qubits,cir.maps,cir.gate_list, [cir.num_qubits+2,3],cir.ignore_gates,"others")
        route.run()
        route.save_program(f"{code_path}/LNN/qft{n}_{p}_{i}_full_code.json")
        data["LNN time 1"] = time.time() - time_s
        
        w = 10
        print(f"qft_{n} with c={w} for p={p}")
        cir = QFT(n)
        time_s = time.time()
        maps = cir.LNN_maps(w)
        cir.to_random_p_qaoa(p/100,edges)
        
        route = QuantumRouter(cir.num_qubits,cir.maps,cir.gate_list, [12,12],cir.ignore_gates,"others")
        route.run()
        route.save_program(f"{code_path}/zigzag/qft{n}_{p}_{i}_full_code.json")
        data["zigzag time"] = time.time() - time_s
        
        data_list.append(data)

data_df = pd.DataFrame(data_list)
data_df.to_csv("time.csv",index=False)
