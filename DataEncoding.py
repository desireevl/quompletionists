# parameters should be of the form params = np.random.uniform(size=(num_layers, num_qubits, 3))

def encoding(x):
    for layer in range(layers)
        for wire in range(wires):
            qml.Rot(x[3*wire: 3*(wire+1)] wires=[wire])     #Encode three features in each qubit
        for wire1 in range(wires):                          #Then add entanglers
            for wire2 in range(wire+1, wires):
                qml.CRZ(pi, wires = [wire1, wire2])
