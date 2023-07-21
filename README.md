# thermalization_dynamics
This is a research code by Nikolay Gnezdilov. It implements the evolution for a few-qubit system after a local quench using exact diagonalization to explore thermalization properties of the system. The quench induces random long-range couplings between the qubits.
"SYK2_thermalizer.py" contains the functions for the post-quench evolution.
"thermal_state.py" contains the functions to compute the observables with a reference thermal state.
"run_quench_protocol.ipynb" includes a brief description of the quench protocol. It executes the protocol and generates the data.
"plotting.ipynb" describes the temperature determination procedure and produces the plots.
