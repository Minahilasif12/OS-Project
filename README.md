# Multi-Core Neural Network Simulation Using Processes and Threads

## Project Overview
This project simulates a neural network architecture on a multi-core processor using OS-level constructs such as processes and threads. Each layer is implemented as a separate process, while individual neurons within a layer are represented as threads.

**Course:** Operating Systems  
**Objective:** Demonstrate multi-core parallelism using processes, threads, IPC, and synchronization

## Team Members
**Team information**
-Ali Hamza 22i-2566
-Minahil Asif 22i-2710

## Features
- **Multi-Process Architecture**: Each layer runs as a separate process
- **Multi-Threaded Neurons**: Each neuron within a layer runs as a thread
- **IPC via Pipes**: Layers communicate using unnamed pipes
- **Thread Synchronization**: Uses pthread mutexes for safe concurrent access
- **Dynamic Configuration**: Number of layers and neurons configurable at runtime
- **Forward Pass**: Computes weighted sums through all layers
- **Backward Pass Simulation**: Demonstrates backward propagation flow
- **File I/O**: Reads weights from input.txt and writes results to output.txt

## System Requirements
- Linux OS (Ubuntu recommended)
- g++ compiler with C++11 support
- pthread library
- Make utility

## File Structure
```
OS-Project/
├── neural_network.cpp    # Main implementation file
├── input.txt            # Input data and weights
├── Makefile            # Build configuration
├── README.md           # This file
└── output.txt          # Generated output (created after running)
```

## Input File Format
The `input.txt` file should contain:
1. **Line 1**: Initial input values (comma-separated)
2. **Lines 2-3**: Weights for 2 input layer neurons
3. **Next N lines**: Weights for hidden layer 1 neurons
4. **Next N lines**: Weights for hidden layer 2 neurons
5. **Next N lines**: Weights for output layer neurons

Where N = number of neurons in each hidden/output layer.

### Example (2 hidden layers, 8 neurons each):
```
1.2, 0.5
0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
...
```

## Compilation and Execution

### Using Make:
```bash
# Compile the program
make

# Run the program
make run

# Clean build files
make clean
```

### Manual Compilation:
```bash
# Compile
g++ -std=c++11 -pthread -o neural_network neural_network.cpp

# Run
./neural_network
```

### Runtime Input:
When you run the program, you'll be prompted:
```
Enter number of hidden layers: 2
Enter number of neurons in each hidden/output layer: 8
```

## How It Works

### 1. Input Layer
- Creates 2 neuron threads
- Reads initial inputs from file
- Computes weighted sum for each neuron
- Sends outputs to hidden layer 1 via pipe

### 2. Hidden Layers
- Each hidden layer is a separate process
- Creates N neuron threads (N specified by user)
- Receives inputs from previous layer via pipe
- Computes weighted sums in parallel
- Sends outputs to next layer via pipe

### 3. Output Layer
- Creates N neuron threads
- Receives inputs from last hidden layer
- Computes weighted sums
- Calculates f(x1) and f(x2):
  - f(x1) = (sum² + sum + 1) / 2
  - f(x2) = (sum² - sum) / 2
- Sends backward signals

### 4. Backward Pass
- Simulates backpropagation
- Propagates f(x1) and f(x2) back through layers
- Displays intermediate values at each layer
- No actual weight updates performed

### 5. Second Forward Pass
- Uses f(x1) and f(x2) as new inputs
- Performs complete forward pass again
- Writes all results to output.txt

## Output
The program generates:
1. **Console Output**: Real-time progress of each layer
2. **output.txt**: Contains all forward/backward pass results

### Sample Console Output:
```
=== INPUT LAYER PROCESS STARTED ===
Input Layer - Initial inputs: 1.2 0.5
Input Layer - Neuron outputs: 3.6 2.8
=== INPUT LAYER COMPLETED ===

=== HIDDEN LAYER 1 PROCESS STARTED ===
Hidden Layer 1 - Received inputs: 3.6 2.8
Hidden Layer 1 - Neuron outputs: ...
=== HIDDEN LAYER 1 COMPLETED ===
...
```

## Key OS Concepts Used

### 1. Process Management
- `fork()`: Creates child processes for each layer
- `wait()` / `waitpid()`: Synchronizes process completion
- Each layer runs independently as a process

### 2. Thread Management
- `pthread_create()`: Creates threads for neurons
- `pthread_join()`: Waits for thread completion
- Parallel neuron computation within layers

### 3. Inter-Process Communication (IPC)
- `pipe()`: Creates unnamed pipes for data transfer
- `read()` / `write()`: Transfers data between processes
- Bidirectional communication for forward/backward passes

### 4. Synchronization
- `pthread_mutex_t`: Protects shared neuron outputs
- `pthread_mutex_lock()` / `pthread_mutex_unlock()`: Critical sections
- Prevents race conditions during parallel computation

### 5. Resource Management
- Proper cleanup of pipes, mutexes, and file descriptors
- Dynamic memory allocation/deallocation
- Process termination handling

## Architecture Benefits
1. **Parallelism**: Multiple neurons compute simultaneously across cores
2. **Scalability**: Easy to add more layers or neurons
3. **Modularity**: Each layer is an independent process
4. **Efficiency**: Leverages multi-core processors effectively

## Troubleshooting

### Compilation Errors:
- Ensure g++ is installed: `sudo apt-get install g++`
- Verify pthread support: `g++ --version`

### Runtime Errors:
- Check input.txt exists and has correct format
- Ensure sufficient lines in input.txt for selected configuration
- Verify file permissions: `chmod +x neural_network`

### Pipe Errors:
- May indicate process synchronization issues
- Check that all pipe file descriptors are properly closed

## Testing
Test with different configurations:
```
1 hidden layer, 4 neurons
2 hidden layers, 8 neurons
3 hidden layers, 16 neurons
```

## Implementation Notes
- The program reads weights line-by-line from input.txt
- Thread-safe access to shared outputs using mutexes
- Pipe communication ensures proper data flow
- All processes are properly synchronized
- Clean resource cleanup prevents memory leaks

## Limitations
- No actual gradient computation or weight updates
- Simplified backward pass (demonstration only)
- No activation functions applied
- Fixed input layer size (2 neurons)

## Future Enhancements
- Implement actual backpropagation with weight updates
- Add activation functions (sigmoid, ReLU, tanh)
- Support for different layer sizes
- Error checking and validation
- Performance metrics and timing

## References
- POSIX Threads Programming: https://computing.llnl.gov/tutorials/pthreads/
- Linux System Calls: man pages (fork, pipe, wait)
- Neural Networks: Basic concepts and architecture

## License
This project is for educational purposes as part of OS course project.
