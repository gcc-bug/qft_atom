
from qft import *
from qiskit import qasm2
import pandas as pd
from Enola.route import *
for n in range(5,101):
    cir = QFT(n)
    
    w = -1
    print(f"qft_{n} with c={w}")
    maps = cir.LNN_maps(w)
    route = QuantumRouter(cir.num_qubits,maps,cir.gate_list, [cir.num_qubits+2,3],"others")
    route.run()
    route.save_program(f"{'linear_map' if w < 0 else 'fold_map'}/qft_{n}_full_code.json")
    
    w = 10
    print(f"qft_{n} with c={w}")
    maps = cir.LNN_maps(w)
    route = QuantumRouter(cir.num_qubits,maps,cir.gate_list, [12,12],"others")
    route.run()
    route.save_program(f"{'linear_map' if w < 0 else 'fold_map'}/qft_{n}_full_code.json")


for n in range(200,1001,100):
    cir = QFT(n)
    
    w = -1
    print(f"qft_{n} with c={w}")
    maps = cir.LNN_maps(w)
    route = QuantumRouter(cir.num_qubits,maps,cir.gate_list, [cir.num_qubits+2,3],"others")
    route.run()
    route.save_program(f"{'linear_map' if w < 0 else 'fold_map'}/qft_{n}_full_code.json")
    
    w = math.ceil(math.sqrt(n))
    d = math.ceil(n//(w-2))
    print(f"qft_{n} with c={w}")
    maps = cir.LNN_maps(w)
    route = QuantumRouter(cir.num_qubits,maps,cir.gate_list, [w+2,d+2],"others")
    route.run()
    route.save_program(f"{'linear_map' if w < 0 else 'fold_map'}/qft_{n}_full_code.json")