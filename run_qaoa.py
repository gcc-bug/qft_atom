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
    cir.to_qaoa()
    route = QuantumRouter(cir.num_qubits,cir.maps,cir.gate_list, [cir.num_qubits+2,3],cir.ignore_gates,"others")
    route.run()
    route.save_program(f"qaoa/code/LNN/qft_{n}_{n-1}_full_code.json")
    data["LNN time 1"] = time.time() - time_s
    
    cir = QFT(n)
    print(f"qft_{n} with c={w}")
    time_s = time.time()
    maps = cir.LNN_maps(w)
    cir.to_qaoa(True)
    route = QuantumRouter(cir.num_qubits,cir.maps,cir.gate_list, [cir.num_qubits+2,3],cir.ignore_gates,"others")
    route.run()
    route.save_program(f"qaoa/code/LNN/qft_{n}_{n-3}_full_code.json")
    data["LNN time 2"] = time.time() - time_s
    # w = 10
    # print(f"qft_{n} with c={w}")
    # time_s = time.time()
    # maps = cir.LNN_maps(w)
    # route = QuantumRouter(cir.num_qubits,maps,cir.gate_list, [12,12],"others")
    # route.run()
    # route.save_program(f"{'linear_map' if w < 0 else 'fold_map'}/qft_{n}_full_code.json")
    # data["fold time"] = time.time() - time_s
    # data_list.append(data)

# for n in range(200,1001,100):
#     cir = QFT(n)
    
#     w = -1
#     print(f"qft_{n} with c={w}")
#     maps = cir.LNN_maps(w)
#     route = QuantumRouter(cir.num_qubits,maps,cir.gate_list, [cir.num_qubits+2,3],"others")
#     route.run()
#     route.save_program(f"{'linear_map' if w < 0 else 'fold_map'}/qft_{n}_full_code.json")
    
#     w = math.ceil(math.sqrt(n))
#     d = math.ceil(n//(w-2))
#     print(f"qft_{n} with c={w}")
#     maps = cir.LNN_maps(w)
#     route = QuantumRouter(cir.num_qubits,maps,cir.gate_list, [w+2,d+2],"others")
#     route.run()
#     route.save_program(f"{'linear_map' if w < 0 else 'fold_map'}/qft_{n}_full_code.json")

data_df = pd.DataFrame(data_list)
data_df.to_csv("time_qaoa.csv",index=False)