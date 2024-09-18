from qiskit import QuantumCircuit,qasm2
from numpy import pi
import copy
class QFT:
    def __init__(self, num_qubits: int) -> None:
        """
        Initialize the Quantum Fourier Transform (QFT) class with a specified number of qubits.

        :param num_qubits: int
            The number of qubits for the quantum circuit.
        """
        self.num_qubits = num_qubits
        self.circuit,self.gate_list = self._initialize_circuit()
        self.num_gate = len(self.circuit.data)

    def _initialize_circuit(self) -> tuple[QuantumCircuit, list[list[int]]]:
        """
        Private method to create and return a Quantum Circuit for QFT.

        Returns:
            QuantumCircuit
                The initialized quantum circuit with QFT applied.
        """
        quantum_circuit = QuantumCircuit(self.num_qubits)

        # Apply Controlled-Z gates in a specific pattern across all qubits
        for i in range(self.num_qubits):
            quantum_circuit.h(i)
            for j in range(i+1,self.num_qubits):
                theta = pi/(2**(j-i))
                quantum_circuit.cp(theta,i,j)

        return quantum_circuit
    
    
    def draw(self) -> None:
        print(self.circuit.draw())
    
    def export_to_qasm(self, filename: str) -> None:
        """
        Export the quantum circuit to an OpenQASM file.

        :param filename: str
            The filename for the QASM output.
        """
        if not filename.endswith('.qasm'):
            raise ValueError("The filename must end with '.qasm' to ensure proper QASM file format.")
        qasm2.dump(self.circuit, filename)
        print(f"QASM file '{filename}' has been created.")