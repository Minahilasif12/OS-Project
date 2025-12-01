/* 
 * CODE EXPLANATION GUIDE
 * Neural Network Simulation - Key Components
 * 
 * This file explains the important parts of neural_network_complete.cpp
 */

// ============================================================================
// 1. HEADER FILES & INCLUDES
// ============================================================================

#include <pthread.h>      // For thread operations (pthread_create, pthread_join)
#include <unistd.h>       // For fork(), pipe(), read(), write()
#include <sys/wait.h>     // For wait(), waitpid()

// ============================================================================
// 2. DATA STRUCTURES
// ============================================================================

struct NeuronData {
    vector<double> inputs;     // Input values to this neuron
    vector<double> weights;    // Weights for this neuron
    double output;             // Computed output
    int neuron_id;            // Neuron identifier
};

// ============================================================================
// 3. THREAD FUNCTION - NEURON COMPUTATION
// ============================================================================

void* neuron_compute(void* arg) {
    /*
     * This function runs in each neuron thread
     * 
     * WHAT IT DOES:
     * - Receives inputs and weights
     * - Computes weighted sum: Σ(input[i] × weight[i])
     * - Returns result
     * 
     * THREADING:
     * - Multiple instances run in parallel
     * - Each computes one neuron's output
     */
    
    NeuronData* data = (NeuronData*)arg;
    
    double sum = 0.0;
    for (size_t i = 0; i < data->inputs.size(); i++) {
        sum += data->inputs[i] * data->weights[i];
    }
    
    data->output = sum;
    pthread_exit(NULL);
}

// ============================================================================
// 4. PIPE COMMUNICATION FUNCTIONS
// ============================================================================

void writeToPipe(int fd, const vector<double>& data) {
    /*
     * PROTOCOL:
     * 1. Write size (int) first
     * 2. Write data array second
     * 
     * This ensures receiver knows how much to read
     */
    
    int size = data.size();
    write(fd, &size, sizeof(int));           // Send size
    write(fd, data.data(), size * sizeof(double)); // Send data
}

vector<double> readFromPipe(int fd) {
    /*
     * PROTOCOL:
     * 1. Read size first
     * 2. Allocate array
     * 3. Read data
     */
    
    int size;
    read(fd, &size, sizeof(int));           // Receive size
    
    vector<double> data(size);
    read(fd, data.data(), size * sizeof(double)); // Receive data
    
    return data;
}

// ============================================================================
// 5. INPUT LAYER PROCESS
// ============================================================================

void inputLayerProcess(int pipe_fd[2], ...) {
    /*
     * RUNS IN: Separate child process (after fork)
     * 
     * PROCESS:
     * 1. Read initial inputs from file
     * 2. Create 2 threads (2 input neurons)
     * 3. Each thread computes weighted sum
     * 4. Collect outputs
     * 5. Send to next layer via pipe
     * 
     * THREADING EXAMPLE:
     */
    
    pthread_t threads[2];
    NeuronData neuron_data[2];
    
    // Create threads
    for (int i = 0; i < 2; i++) {
        pthread_create(&threads[i], NULL, neuron_compute, &neuron_data[i]);
    }
    
    // Wait for completion
    for (int i = 0; i < 2; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Send results to next layer
    writeToPipe(pipe_fd[1], outputs);
}

// ============================================================================
// 6. HIDDEN/OUTPUT LAYER PROCESS
// ============================================================================

void layerProcess(int read_fd, int write_fd, ...) {
    /*
     * RUNS IN: Separate child process
     * 
     * PROCESS:
     * 1. Read inputs from previous layer (via pipe)
     * 2. Create N threads (N neurons)
     * 3. Each thread computes weighted sum in parallel
     * 4. Collect outputs
     * 5. Send to next layer (or compute f(x1), f(x2) if output layer)
     * 
     * SYNCHRONIZATION:
     * - Uses mutex to protect shared output collection
     */
    
    // Read from previous layer
    vector<double> inputs = readFromPipe(read_fd);
    
    // Create threads for neurons
    pthread_t* threads = new pthread_t[num_neurons];
    
    for (int i = 0; i < num_neurons; i++) {
        pthread_create(&threads[i], NULL, neuron_compute, &neuron_data[i]);
    }
    
    // Wait and collect
    for (int i = 0; i < num_neurons; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Send to next layer
    writeToPipe(write_fd, outputs);
}

// ============================================================================
// 7. MAIN FUNCTION - PROCESS ORCHESTRATION
// ============================================================================

int main() {
    /*
     * MAIN PROCESS RESPONSIBILITIES:
     * 1. Get user configuration (layers, neurons)
     * 2. Create pipes for IPC
     * 3. Fork processes for each layer
     * 4. Wait for all processes to complete
     * 5. Coordinate backward pass
     * 6. Run second forward pass
     */
    
    // Get configuration
    cin >> num_hidden_layers >> neurons_per_layer;
    
    // Create pipes
    int forward_pipes[total_layers][2];
    for (int i = 0; i < total_layers; i++) {
        pipe(forward_pipes[i]);  // Creates pipe[0]=read, pipe[1]=write
    }
    
    // ========================================================================
    // FORK INPUT LAYER
    // ========================================================================
    
    pid_t input_pid = fork();
    if (input_pid == 0) {
        // CHILD PROCESS: Input Layer
        close(forward_pipes[0][0]);  // Close unused read end
        
        inputLayerProcess(forward_pipes[0], ...);
        
        close(forward_pipes[0][1]);
        exit(0);  // Child terminates
    }
    
    // PARENT PROCESS continues
    close(forward_pipes[0][1]);  // Close unused write end
    
    // ========================================================================
    // FORK HIDDEN LAYERS (in loop)
    // ========================================================================
    
    for (int i = 0; i < num_hidden_layers; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            // CHILD PROCESS: Hidden Layer i
            layerProcess(forward_pipes[i][0], forward_pipes[i+1][1], ...);
            exit(0);
        }
        
        // PARENT: Close used pipes
        close(forward_pipes[i][0]);
        close(forward_pipes[i+1][1]);
    }
    
    // ========================================================================
    // FORK OUTPUT LAYER
    // ========================================================================
    
    pid_t output_pid = fork();
    if (output_pid == 0) {
        // CHILD PROCESS: Output Layer
        layerProcess(..., true);  // true = is_output_layer
        exit(0);
    }
    
    // ========================================================================
    // WAIT FOR ALL PROCESSES
    // ========================================================================
    
    waitpid(input_pid, NULL, 0);     // Wait for input layer
    for (pid in hidden_layers) {
        waitpid(pid, NULL, 0);        // Wait for each hidden layer
    }
    waitpid(output_pid, NULL, 0);    // Wait for output layer
    
    // ========================================================================
    // BACKWARD PASS & SECOND FORWARD PASS
    // ========================================================================
    
    // Read f(x1) and f(x2) from output layer
    // Display backward propagation
    // Repeat forward pass with new inputs
    
    return 0;
}

// ============================================================================
// KEY OS CONCEPTS MAPPED TO CODE
// ============================================================================

/*
 * PROCESS CREATION:
 *   fork() - Creates child process
 *   Child gets copy of parent's memory
 *   Returns 0 in child, PID in parent
 * 
 * PROCESS SYNCHRONIZATION:
 *   waitpid() - Parent waits for specific child
 *   exit() - Child terminates
 * 
 * INTER-PROCESS COMMUNICATION:
 *   pipe() - Creates one-way communication channel
 *   pipe[0] - Read end
 *   pipe[1] - Write end
 *   Must close unused ends!
 * 
 * THREAD CREATION:
 *   pthread_create() - Spawns new thread
 *   Threads share memory space
 *   Lightweight vs processes
 * 
 * THREAD SYNCHRONIZATION:
 *   pthread_join() - Wait for thread completion
 *   pthread_mutex - Protect shared data
 * 
 * MUTEX USAGE:
 *   pthread_mutex_lock() - Acquire lock
 *   [Critical section]
 *   pthread_mutex_unlock() - Release lock
 */

// ============================================================================
// EXECUTION FLOW
// ============================================================================

/*
 * FORWARD PASS:
 * 
 *   Main Process
 *       │
 *       ├─ fork() → Input Layer Process
 *       │              ├─ pthread_create() → Neuron 0 Thread
 *       │              ├─ pthread_create() → Neuron 1 Thread
 *       │              ├─ pthread_join() - wait for threads
 *       │              └─ write to pipe → outputs
 *       │
 *       ├─ fork() → Hidden Layer 1 Process
 *       │              ├─ read from pipe ← inputs
 *       │              ├─ pthread_create() × N → Neuron Threads
 *       │              ├─ pthread_join() - wait for all
 *       │              └─ write to pipe → outputs
 *       │
 *       ├─ fork() → Hidden Layer 2 Process
 *       │              └─ (same as above)
 *       │
 *       └─ fork() → Output Layer Process
 *                      ├─ read from pipe ← inputs
 *                      ├─ pthread_create() × N → Neuron Threads
 *                      ├─ pthread_join() - wait for all
 *                      ├─ compute f(x1), f(x2)
 *                      └─ write to backward pipe
 * 
 *   Main Process
 *       └─ waitpid() for all children
 *       └─ read backward values
 *       └─ display backward propagation
 *       └─ repeat forward pass with f(x1), f(x2)
 */

// ============================================================================
// PIPE COMMUNICATION EXAMPLE
// ============================================================================

/*
 * Layer 1 Process              Pipe               Layer 2 Process
 * ┌──────────────┐                               ┌──────────────┐
 * │ Compute done │                               │              │
 * │ outputs =    │                               │              │
 * │ {2.5, 3.7}   │                               │              │
 * │              │                               │              │
 * │ write(pipe)  │ ──────────────────────────>   │ read(pipe)   │
 * │   size=2     │ ═══════════════════════════>  │   size=2     │
 * │   data[0]=2.5│ ═══════════════════════════>  │   data[0]    │
 * │   data[1]=3.7│ ═══════════════════════════>  │   data[1]    │
 * │              │                               │              │
 * │ close(pipe)  │                               │ Use inputs   │
 * │ exit(0)      │                               │ {2.5, 3.7}   │
 * └──────────────┘                               └──────────────┘
 */

// ============================================================================
// THREADING EXAMPLE
// ============================================================================

/*
 * Layer Process (e.g., Hidden Layer with 4 neurons)
 * 
 * Main thread:
 *   ├─ Read inputs from pipe
 *   ├─ Create thread 0 → compute neuron 0 output
 *   ├─ Create thread 1 → compute neuron 1 output  } Run in
 *   ├─ Create thread 2 → compute neuron 2 output  } parallel
 *   ├─ Create thread 3 → compute neuron 3 output  } on different
 *   │                                              } CPU cores
 *   ├─ pthread_join(thread 0) - wait
 *   ├─ pthread_join(thread 1) - wait
 *   ├─ pthread_join(thread 2) - wait
 *   ├─ pthread_join(thread 3) - wait
 *   └─ All done, send outputs to next layer
 */

// ============================================================================
// END OF CODE EXPLANATION
// ============================================================================
