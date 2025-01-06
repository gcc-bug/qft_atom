from qft import *
from qiskit import qasm2
import pandas as pd
from Enola.route import *
import time
data_list = []
n = 50
# for d in range(1,0,-1):
#     data = {"degree": d}
#     cir = QFT(n)
    
#     w = -1
#     print(f"qft_{n} with c={w} for d={d}")
#     time_s = time.time()
#     maps = cir.LNN_maps(w)
#     cir.to_random_qaoa(d)
#     cir.export_to_qasm(f"random_qaoa/qasm/qft{n}_{d}.qasm")
#     route = QuantumRouter(cir.num_qubits,cir.maps,cir.gate_list, [cir.num_qubits+2,3],cir.ignore_gates,"others")
#     route.run()
#     route.save_program(f"random_qaoa/code/LNN/qft{n}_{d}_full_code.json")
#     data["LNN time 1"] = time.time() - time_s

for d in range(49,0,-1):
    data = {"degree": d}
    cir = QFT(n)
    
    w = 10
    print(f"qft_{n} with c={w} for d={d}")
    time_s = time.time()
    maps = cir.LNN_maps(w)
    
    edges = read_edges_from_qasm(f"random_qaoa/qasm/qft{n}_{d}.qasm")
    cir.to_random_qaoa(d,edges)
    route = QuantumRouter(cir.num_qubits,cir.maps,cir.gate_list, [12,12],cir.ignore_gates,"others")
    route.run()
    route.save_program(f"random_qaoa/code/zigzag/qft{n}_{d}_full_code.json")
    data["zigzag time"] = time.time() - time_s
