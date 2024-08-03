from qiskit import QuantumCircuit,qasm2
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

    def _initialize_circuit(self) -> tuple[QuantumCircuit, list[list[int]]]:
        """
        Private method to create and return a Quantum Circuit for QFT.

        Returns:
            QuantumCircuit
                The initialized quantum circuit with QFT applied.
        """
        quantum_circuit = QuantumCircuit(self.num_qubits)
        seprate_gate = [[]]

        # Apply Controlled-Z gates in a specific pattern across all qubits
        for i in range(1, self.num_qubits):
            gate_list = []
            gate_list.append([0,i])
            quantum_circuit.cz(0, i)  # Apply CZ between the first qubit and others
            j = 1
            while j < i / 2:
                gate_list.append([j,i-j])
                quantum_circuit.cz(j, i - j)  # Apply CZ in a staggered manner within a subset of qubits
                j += 1
            # seprate_gate.append(gate_list)
            # gate_list = []
            gate_list.append([0,i])
            quantum_circuit.cz(0, i)  # Apply CZ between the first qubit and others
            j = 1
            while j < i / 2:
                gate_list.append([j,i-j])
                quantum_circuit.cz(j, i - j)  # Apply CZ in a staggered manner within a subset of qubits
                j += 1
            seprate_gate.append(gate_list)

        # Further apply Controlled-Z gates to ensure complete QFT
        for i in range(1, self.num_qubits - 1):
            gate_list = []
            gate_list.append([i, self.num_qubits - 1])
            quantum_circuit.cz(i, self.num_qubits - 1)  # Connect each qubit to the last one with CZ
            j = i + 1
            while j < (self.num_qubits - 1 + i) / 2:
                gate_list.append([j, self.num_qubits - 1 + i - j])
                quantum_circuit.cz(j, self.num_qubits - 1 + i - j)  # Staggered CZ gates towards the end
                j += 1
            # seprate_gate.append(gate_list)
            # gate_list = []
            gate_list.append([i, self.num_qubits - 1])
            quantum_circuit.cz(i, self.num_qubits - 1)  # Connect each qubit to the last one with CZ
            j = i + 1
            while j < (self.num_qubits - 1 + i) / 2:
                gate_list.append([j, self.num_qubits - 1 + i - j])
                quantum_circuit.cz(j, self.num_qubits - 1 + i - j)  # Staggered CZ gates towards the end
                j += 1
            seprate_gate.append(gate_list)

        return quantum_circuit,seprate_gate
    
    
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