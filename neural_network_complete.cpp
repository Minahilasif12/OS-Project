#include <iostream>
#include <fstream>
#include <vector>
#include <pthread.h>
#include <unistd.h>
#include <sys/wait.h>
#include <cstring>
#include <cmath>
#include <sstream>
#include <iomanip>

using namespace std;

// Structure for neuron thread data
struct NeuronData {
    vector<double> inputs;
    vector<double> weights;
    double output;
    int neuron_id;
};

// Mutex for thread synchronization
pthread_mutex_t output_mutex = PTHREAD_MUTEX_INITIALIZER;
vector<double> layer_outputs;

// Thread function for neuron computation
void* neuron_compute(void* arg) {
    NeuronData* data = (NeuronData*)arg;
    
    double sum = 0.0;
    for (size_t i = 0; i < data->inputs.size(); i++) {
        sum += data->inputs[i] * data->weights[i];
    }
    
    data->output = sum;
    
    cout << "  Neuron " << data->neuron_id << " computed: " << fixed << setprecision(4) << sum << endl;
    
    pthread_exit(NULL);
}

// Read a line and parse comma-separated doubles
vector<double> parseLine(const string& line) {
    vector<double> values;
    stringstream ss(line);
    double value;
    char comma;
    
    while (ss >> value) {
        values.push_back(value);
        ss >> comma;
    }
    
    return values;
}

// Read weights from file starting at a specific line
vector<vector<double>> readWeights(const string& filename, int start_line, int num_lines) {
    vector<vector<double>> weights;
    ifstream file(filename);
    string line;
    int current_line = 0;
    
    while (getline(file, line)) {
        if (current_line >= start_line && current_line < start_line + num_lines) {
            weights.push_back(parseLine(line));
        }
        current_line++;
    }
    
    file.close();
    return weights;
}

// Write data to pipe
void writeToPipe(int fd, const vector<double>& data) {
    int size = data.size();
    write(fd, &size, sizeof(int));
    if (size > 0) {
        write(fd, data.data(), size * sizeof(double));
    }
}

// Read data from pipe
vector<double> readFromPipe(int fd) {
    int size;
    read(fd, &size, sizeof(int));
    
    vector<double> data(size);
    if (size > 0) {
        read(fd, data.data(), size * sizeof(double));
    }
    
    return data;
}

// Input Layer Process
void inputLayerProcess(int pipe_fd[2], const vector<double>& initial_inputs, 
                       const vector<vector<double>>& weights, ofstream& logFile) {
    cout << "\n=== INPUT LAYER (Process) ===" << endl;
    logFile << "\n=== INPUT LAYER ===" << endl;
    
    cout << "Initial inputs: ";
    for (double val : initial_inputs) {
        cout << val << " ";
    }
    cout << endl;
    
    // Create threads for 2 input neurons
    pthread_t threads[2];
    NeuronData neuron_data[2];
    
    for (int i = 0; i < 2; i++) {
        neuron_data[i].inputs = initial_inputs;
        neuron_data[i].weights = weights[i];
        neuron_data[i].neuron_id = i;
        neuron_data[i].output = 0.0;
        
        pthread_create(&threads[i], NULL, neuron_compute, &neuron_data[i]);
    }
    
    // Wait for threads and collect outputs
    vector<double> outputs;
    for (int i = 0; i < 2; i++) {
        pthread_join(threads[i], NULL);
        outputs.push_back(neuron_data[i].output);
    }
    
    cout << "Output: ";
    logFile << "Outputs: ";
    for (double val : outputs) {
        cout << fixed << setprecision(4) << val << " ";
        logFile << fixed << setprecision(4) << val << " ";
    }
    cout << endl;
    logFile << endl;
    
    // Send to next layer
    writeToPipe(pipe_fd[1], outputs);
    close(pipe_fd[1]);
}

// Hidden/Output Layer Process
void layerProcess(int read_fd, int write_fd, int layer_num, int num_neurons,
                 const vector<vector<double>>& weights, bool is_output,
                 ofstream& logFile, vector<double>& backward_values) {
    
    cout << "\n=== " << (is_output ? "OUTPUT" : "HIDDEN") << " LAYER " 
         << layer_num << " (Process) ===" << endl;
    logFile << "\n=== " << (is_output ? "OUTPUT" : "HIDDEN") << " LAYER " 
            << layer_num << " ===" << endl;
    
    // Read inputs from previous layer
    vector<double> inputs = readFromPipe(read_fd);
    close(read_fd);
    
    cout << "Received inputs (" << inputs.size() << "): ";
    for (double val : inputs) {
        cout << fixed << setprecision(4) << val << " ";
    }
    cout << endl;
    
    // Create threads for neurons
    pthread_t* threads = new pthread_t[num_neurons];
    NeuronData* neuron_data = new NeuronData[num_neurons];
    
    for (int i = 0; i < num_neurons; i++) {
        neuron_data[i].inputs = inputs;
        neuron_data[i].weights = weights[i];
        neuron_data[i].neuron_id = i;
        neuron_data[i].output = 0.0;
        
        pthread_create(&threads[i], NULL, neuron_compute, &neuron_data[i]);
    }
    
    // Wait for threads and collect outputs
    vector<double> outputs;
    for (int i = 0; i < num_neurons; i++) {
        pthread_join(threads[i], NULL);
        outputs.push_back(neuron_data[i].output);
    }
    
    cout << "Output: ";
    logFile << "Outputs: ";
    for (double val : outputs) {
        cout << fixed << setprecision(4) << val << " ";
        logFile << fixed << setprecision(4) << val << " ";
    }
    cout << endl;
    logFile << endl;
    
    // If output layer, compute f(x1) and f(x2)
    if (is_output) {
        double sum = 0.0;
        for (double val : outputs) {
            sum += val;
        }
        
        double fx1 = (sum * sum + sum + 1) / 2.0;
        double fx2 = (sum * sum - sum) / 2.0;
        
        cout << "\nComputed f(x1) = " << fixed << setprecision(4) << fx1 << endl;
        cout << "Computed f(x2) = " << fixed << setprecision(4) << fx2 << endl;
        
        logFile << "f(x1) = " << fixed << setprecision(4) << fx1 << endl;
        logFile << "f(x2) = " << fixed << setprecision(4) << fx2 << endl;
        
        backward_values = {fx1, fx2};
        writeToPipe(write_fd, backward_values);
    } else {
        writeToPipe(write_fd, outputs);
    }
    
    close(write_fd);
    
    delete[] threads;
    delete[] neuron_data;
}

// Backward propagation display
void displayBackwardProp(int layer_num, const vector<double>& values) {
    cout << "\n[BACKWARD] Layer " << layer_num << " received: ";
    for (double val : values) {
        cout << fixed << setprecision(4) << val << " ";
    }
    cout << endl;
}

int main() {
    string filename = "input.txt";
    int num_hidden_layers;
    int neurons_per_layer;
    
    cout << "========================================" << endl;
    cout << "  NEURAL NETWORK SIMULATION" << endl;
    cout << "  Multi-Core Process & Thread Based" << endl;
    cout << "========================================" << endl;
    
    cout << "\nEnter number of hidden layers: ";
    cin >> num_hidden_layers;
    
    cout << "Enter number of neurons in each hidden/output layer: ";
    cin >> neurons_per_layer;
    
    // Open output file
    ofstream output_file("output.txt");
    output_file << "=== NEURAL NETWORK SIMULATION ===" << endl;
    output_file << "Configuration:" << endl;
    output_file << "  Hidden Layers: " << num_hidden_layers << endl;
    output_file << "  Neurons per layer: " << neurons_per_layer << endl;
    
    // Read input file
    ifstream input_file(filename);
    if (!input_file.is_open()) {
        cerr << "Error: Cannot open " << filename << endl;
        return 1;
    }
    
    // Read initial inputs (line 0)
    string line;
    getline(input_file, line);
    vector<double> initial_inputs = parseLine(line);
    input_file.close();
    
    cout << "\n*** FORWARD PASS ***" << endl;
    output_file << "\n*** FORWARD PASS ***" << endl;
    
    // Calculate line offsets for weights
    int line_offset = 1;  // Start after input line
    
    // Read weights for input layer (2 neurons)
    vector<vector<double>> input_weights = readWeights(filename, line_offset, 2);
    line_offset += 2;
    
    // Create pipes for forward pass
    int total_layers = 1 + num_hidden_layers + 1; // input + hidden + output
    int forward_pipes[total_layers][2];
    
    for (int i = 0; i < total_layers; i++) {
        pipe(forward_pipes[i]);
    }
    
    // Fork input layer
    pid_t input_pid = fork();
    if (input_pid == 0) {
        // Child: Input layer
        close(forward_pipes[0][0]); // Close read end
        
        inputLayerProcess(forward_pipes[0], initial_inputs, input_weights, output_file);
        
        output_file.close();
        exit(0);
    }
    
    // Parent continues
    close(forward_pipes[0][1]); // Close write end
    
    // Fork hidden layers
    vector<pid_t> layer_pids;
    for (int i = 0; i < num_hidden_layers; i++) {
        vector<vector<double>> layer_weights = readWeights(filename, line_offset, neurons_per_layer);
        line_offset += neurons_per_layer;
        
        pid_t pid = fork();
        if (pid == 0) {
            // Child: Hidden layer
            close(forward_pipes[i+1][0]); // Close read end of output pipe
            
            vector<double> dummy;
            layerProcess(forward_pipes[i][0], forward_pipes[i+1][1], 
                        i+1, neurons_per_layer, layer_weights, false, output_file, dummy);
            
            output_file.close();
            exit(0);
        }
        
        layer_pids.push_back(pid);
        close(forward_pipes[i][0]);     // Close read end
        close(forward_pipes[i+1][1]);   // Close write end
    }
    
    // Fork output layer
    vector<vector<double>> output_weights = readWeights(filename, line_offset, neurons_per_layer);
    vector<double> backward_values;
    
    // Create backward pipe
    int backward_pipe[2];
    pipe(backward_pipe);
    
    pid_t output_pid = fork();
    if (output_pid == 0) {
        // Child: Output layer
        close(backward_pipe[0]); // Close read end
        
        layerProcess(forward_pipes[num_hidden_layers][0], backward_pipe[1],
                    num_hidden_layers + 1, neurons_per_layer, output_weights, 
                    true, output_file, backward_values);
        
        output_file.close();
        exit(0);
    }
    
    close(forward_pipes[num_hidden_layers][0]);
    close(backward_pipe[1]);
    
    // Wait for all processes
    waitpid(input_pid, NULL, 0);
    for (pid_t pid : layer_pids) {
        waitpid(pid, NULL, 0);
    }
    waitpid(output_pid, NULL, 0);
    
    // Read backward values
    backward_values = readFromPipe(backward_pipe[0]);
    close(backward_pipe[0]);
    
    // Display backward propagation
    cout << "\n*** BACKWARD PASS (Simulation) ***" << endl;
    output_file << "\n*** BACKWARD PASS ***" << endl;
    
    for (int i = num_hidden_layers; i >= 0; i--) {
        displayBackwardProp(i, backward_values);
        output_file << "Layer " << i << " backward: ";
        for (double val : backward_values) {
            output_file << fixed << setprecision(4) << val << " ";
        }
        output_file << endl;
    }
    
    // Second forward pass with f(x1) and f(x2)
    cout << "\n*** SECOND FORWARD PASS with f(x1) and f(x2) ***" << endl;
    output_file << "\n*** SECOND FORWARD PASS with f(x1) and f(x2) ***" << endl;
    
    // Create new pipes for second forward pass
    for (int i = 0; i < total_layers; i++) {
        pipe(forward_pipes[i]);
    }
    
    // Fork input layer with new inputs
    input_pid = fork();
    if (input_pid == 0) {
        close(forward_pipes[0][0]);
        
        inputLayerProcess(forward_pipes[0], backward_values, input_weights, output_file);
        
        output_file.close();
        exit(0);
    }
    
    close(forward_pipes[0][1]);
    
    // Fork hidden layers again
    line_offset = 3; // Reset to first hidden layer weights
    for (int i = 0; i < num_hidden_layers; i++) {
        vector<vector<double>> layer_weights = readWeights(filename, line_offset, neurons_per_layer);
        line_offset += neurons_per_layer;
        
        pid_t pid = fork();
        if (pid == 0) {
            close(forward_pipes[i+1][0]);
            
            vector<double> dummy;
            layerProcess(forward_pipes[i][0], forward_pipes[i+1][1], 
                        i+1, neurons_per_layer, layer_weights, false, output_file, dummy);
            
            output_file.close();
            exit(0);
        }
        
        close(forward_pipes[i][0]);
        close(forward_pipes[i+1][1]);
    }
    
    // Fork output layer again
    pipe(backward_pipe);
    
    output_pid = fork();
    if (output_pid == 0) {
        close(backward_pipe[0]);
        
        vector<double> dummy;
        layerProcess(forward_pipes[num_hidden_layers][0], backward_pipe[1],
                    num_hidden_layers + 1, neurons_per_layer, output_weights, 
                    true, output_file, dummy);
        
        output_file.close();
        exit(0);
    }
    
    close(forward_pipes[num_hidden_layers][0]);
    close(backward_pipe[1]);
    
    // Wait for all second pass processes
    waitpid(input_pid, NULL, 0);
    for (int i = 0; i < num_hidden_layers; i++) {
        wait(NULL);
    }
    waitpid(output_pid, NULL, 0);
    
    close(backward_pipe[0]);
    
    cout << "\n========================================" << endl;
    cout << "  SIMULATION COMPLETED" << endl;
    cout << "  Results saved to output.txt" << endl;
    cout << "========================================" << endl;
    
    output_file << "\n=== SIMULATION COMPLETED ===" << endl;
    output_file.close();
    
    pthread_mutex_destroy(&output_mutex);
    
    return 0;
}
