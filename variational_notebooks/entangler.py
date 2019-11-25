def get_entangler_map(map_type='full', num_qubits=8):
    """
        Utility method to get an entangler map among qubits.
        
        Args:
            map_type (str): 'full' entangles each qubit with all the subsequent ones
                           'linear' entangles each qubit with the next
            num_qubits (int): Number of qubits for which the map is needed
        Returns:
            A map of qubit index to an array of indexes to which this should be entangled
        Raises:
            ValueError: if map_type is not valid.
    """
    
    ret = []
    
    if num_qubits > 1:
        if map_type == 'full':
            ret = [[i, j] for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        elif map_type == 'linear':
            ret = [[i, i + 1] for i in range(num_qubits - 1)]
        else:
            raise ValueError("map_type only supports 'full' or 'linear' type.")
    else:
        raise ValueError("num_qubits is positive non null")
    return ret