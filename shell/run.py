import os, sys
sys.path.append(os.path.abspath('..'))

from qft import *
from qiskit import qasm2
import pandas as pd
from Enola.route import *
import time
data_list = []
for n in range(5,51):
    data = {"num": n}
    cir = QFT(n)
    
    w = -1
    print(f"qft_{n} with c={w}")
    time_s = time.time()
    maps = cir.LNN_maps(w)
    route = QuantumRouter(cir.num_qubits,maps,cir.gate_list, [cir.num_qubits+2,3],None,"others")
    route.run()
    route.save_program(f"{'linear_map' if w < 0 else 'fold_map'}/qft_{n}_full_code.json")
    data["LNN time"] = time.time() - time_s
    
    w = 10
    print(f"qft_{n} with c={w}")
    time_s = time.time()
    maps = cir.LNN_maps(w)
    route = QuantumRouter(cir.num_qubits,maps,cir.gate_list, [12,12],None,"others")
    route.run()
    route.save_program(f"{'linear_map' if w < 0 else 'fold_map'}/qft_{n}_full_code.json")
    data["fold time"] = time.time() - time_s
    data_list.append(data)

data_df = pd.DataFrame(data_list)
data_df.to_csv("time.csv",index=False)