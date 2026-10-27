#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>

#include <mpi.h>

const int IS_OUTPUT_NODE = 1;
const int IS_HIDDEN_NODE = 0;
const int IS_OUTPUT_LAYER = 1;
const int IS_HIDDEN_LAYER = 0;

int number_of_procs;
int proc_id;

struct hidden_layer_node {
    std::vector<float> weights;
    std::vector<float> inputs;
    float output;
    float post_activation_output;
    int is_output_node; // 1 if output node, 0 if hidden node
    std::vector<float> error_term;
    float (*func)(float);
};

struct hidden_layer {
    std::vector<hidden_layer_node> layer_weights;
    std::vector<float> outputs;
    int is_output_layer; // 1 if output layer, 0 if hidden layer
};

MPI_Datatype node_scalars_t;

void create_node_scalars_type() {
    hidden_layer_node probe;
    MPI_Aint base, addr_out, addr_post;
    MPI_Get_address(&probe,                        &base);
    MPI_Get_address(&probe.output,                 &addr_out);
    MPI_Get_address(&probe.post_activation_output, &addr_post);

    int          blocklens[2] = {1, 1};
    MPI_Aint     offsets[2]   = {addr_out - base, addr_post - base};
    MPI_Datatype types[2]     = {MPI_FLOAT, MPI_FLOAT};

    MPI_Datatype raw;
    MPI_Type_create_struct(2, blocklens, offsets, types, &raw);
    MPI_Type_create_resized(raw, 0, sizeof(hidden_layer_node), &node_scalars_t);
    MPI_Type_commit(&node_scalars_t);
    MPI_Type_free(&raw);
}

float tanh_calc(float value) {
    return tanh(value);
}

float relu_calc(float value) {
    if(value > 0) {
        return value;
    }
    else {
        return 0;
    }
}

float linear_calc(float value) {
    return value;
}

float tanh_derivative(float value) {
    return 1 - (tanh_calc(value) * tanh_calc(value));
}

float relu_derivative(float value) {
    if(value > 0) {
        return 1;
    }
    else {
        return 0;
    }
}

float linear_derivative(float value) {
    return 1;
}

float squared_error_SISO(float output, float target) {
    float error = output - target;
    return (error * error) / 2;
}

float squared_error_MISO(std::vector<float> outputs, std::vector<float> targets) {
    float sum = 0;
    for(int i = 0; i < outputs.size(); i++) {
        float error = outputs.at(i) - targets.at(i);
        sum += (error * error);
    }
    return sum / outputs.size();
}

float MSE_derivative(float output, float target) {
    return output - target;
}

float forward_pass_single_node( std::vector<float> curr_weights, std::vector<float> inputs, float (*func)(float) ) {

    //for(auto k : curr_weights) {
    //    printf("Size of Node: %f", k);
    //}
    //printf("\n");

    if(inputs.size() != curr_weights.size()) { // input no. must equal number of weights in the node
        //std::cout << "Input: " << inputs.size() << " Weights " << curr_weights.size() << std::endl;
        return errno;
    }

    float output = 1.0;
    for(int i = 0; i < curr_weights.size(); i++) { // for all the weights in the node
        //std::cout << "Input: " << inputs.at(i) << " Weight: " << curr_weights.at(i) << " Makes " << inputs.at(i) * curr_weights.at(i) << std::endl;
        output += inputs.at(i) * curr_weights.at(i); // multiply the input with that input's corresponding weight
        // and sum these products together
    }

    // output = func(output);
    return output; // one singular output
}

hidden_layer forward_pass_layer(hidden_layer hl, std::vector<float> inputs) {

    int total_nodes = hl.layer_weights.size();

    std::vector<int> recvcounts(number_of_procs);
    std::vector<int> displs(number_of_procs);
    for(int r = 0; r < number_of_procs; r++) {
        int first = (int)(((long long)r       * total_nodes) / number_of_procs);
        int last  = (int)(((long long)(r + 1) * total_nodes) / number_of_procs);
        displs[r]     = first;
        recvcounts[r] = last - first;
    }

    int start = displs[proc_id];
    int end   = start + recvcounts[proc_id];

    for(int i = start; i < end; i++) {
        float (*func)(float) = hl.layer_weights.at(i).func;
        hl.layer_weights.at(i).output = forward_pass_single_node(hl.layer_weights.at(i).weights, inputs, func);
        hl.layer_weights.at(i).post_activation_output = func(hl.layer_weights.at(i).output);
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   hl.layer_weights.data(), recvcounts.data(), displs.data(), node_scalars_t,
                   MPI_COMM_WORLD);

    hl.outputs.assign(total_nodes, 0.0f);
    for(int i = 0; i < total_nodes; i++) {
        hl.layer_weights.at(i).inputs = inputs;
        hl.outputs[i] = hl.layer_weights.at(i).post_activation_output;
    }

    //std::cout << "Ending Forward Pass" << std::endl;

    return hl;
}

float find_derivative(hidden_layer_node node) {
    if(node.func == tanh_calc) {
        return tanh_derivative(node.output);
    }
    else if(node.func == relu_calc) {
        return relu_derivative(node.output);
    }
    else if(node.func == linear_calc) {
        return linear_derivative(node.output);
    }
    else {
        return 0;
    }
}

std::vector<float> find_column_error_terms(hidden_layer hl, int column) {
    std::vector<float> error_terms;
    for(int i = 0; i < hl.layer_weights.size(); i++) {
        error_terms.push_back(hl.layer_weights.at(i).error_term.at(column));
    }
    return error_terms;
}

std::vector<float> get_column_error_terms(hidden_layer hl, int column) {
    std::vector<float> error_terms;
    for(int i = 0; i < hl.layer_weights.size(); i++) {
        error_terms.push_back(hl.layer_weights.at(i).error_term.at(column));
    }
    return error_terms;
}

hidden_layer_node backward_prop_single_node(hidden_layer_node node, float learning_rate, 
    std::vector<float> targets, std::vector<float> prev_layer_weights = std::vector<float>()) {
    
    float loss = 0;

    //std::cout << "Finding Loss" << std::endl;
    if(node.is_output_node == IS_OUTPUT_NODE) {
        //std::cout << "Output: " << node.output << std::endl;
        //std::cout << "Using MSE with target " << targets.at(0) << std::endl;
            
        loss = MSE_derivative(node.post_activation_output, targets.at(0));
    }
    else {
        for(int k = 0; k < targets.size(); k++) {
            loss += targets.at(k);
        }
    }

    for(int i = 0; i < node.weights.size(); i++) {
        
        //std::cout << "Loss: " << loss << std::endl;
        //std::cout << "Output: " << node.output << std::endl;
        
        float err_term = find_derivative(node) * loss;
        //std::cout << "Beta: " << node.error_term.empty() << std::endl;
        if(node.error_term.empty()) {
            //std::cout << "Alpha: " << std::endl;
            node.error_term.push_back(err_term * node.weights.at(i));
        } 
        else {
            
            if(node.error_term.size() > i) {
                node.error_term.at(i) = err_term * node.weights.at(i);
            }
            else {
                node.error_term.push_back(err_term * node.weights.at(i));
            }
            //std::cout << "Delta: " << node.error_term.at(i) << std::endl;
        }
        

        for(auto i : node.inputs) {
            //std::cout << "Input: " << i << std::endl;
        }
        //std::cout << "Error Term: " << err_term << std::endl;
        
        float weight_change = learning_rate * node.inputs.at(i) * err_term;

        //std::cout << "Weight Change: " << weight_change << std::endl;
        node.weights.at(i) -= weight_change;
    }

    return node;
}

// which nodes each rank owns: rank r owns counts[r] nodes, starting at node firsts[r]
void build_partition(int total_nodes, std::vector<int>& counts, std::vector<int>& firsts) {
    counts.assign(number_of_procs, 0);
    firsts.assign(number_of_procs, 0);
    for(int r = 0; r < number_of_procs; r++) {
        int first = (int)(((long long)r       * total_nodes) / number_of_procs);
        int last  = (int)(((long long)(r + 1) * total_nodes) / number_of_procs);
        firsts[r] = first;
        counts[r] = last - first;
    }
}

// after backward: share every node's updated weights and error_term, so every rank holds the full layer
void sync_layer_after_backward(hidden_layer& hl, const std::vector<int>& counts, const std::vector<int>& firsts) {
    int total_nodes = hl.layer_weights.size();
    int width  = hl.layer_weights.at(0).weights.size();   // weights per node
    int stride = 2 * width;                               // floats per node: weights, then error_term

    std::vector<int> recvcounts(number_of_procs), displs(number_of_procs);
    for(int r = 0; r < number_of_procs; r++) {
        recvcounts[r] = counts[r] * stride;               // in FLOATS, not nodes
        displs[r]     = firsts[r] * stride;
    }

    int start = firsts[proc_id];
    int end   = start + counts[proc_id];

    std::vector<float> packed((size_t)total_nodes * stride);
    for(int i = start; i < end; i++) {
        hidden_layer_node& n = hl.layer_weights.at(i);
        if(n.weights.size() != width || n.error_term.size() != width) {
            throw std::runtime_error("sync_layer_after_backward: unexpected vector size");
        }
        std::copy(n.weights.begin(),    n.weights.end(),    packed.begin() + (size_t)i * stride);
        std::copy(n.error_term.begin(), n.error_term.end(), packed.begin() + (size_t)i * stride + width);
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   packed.data(), recvcounts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

    for(int i = 0; i < total_nodes; i++) {
        hidden_layer_node& n = hl.layer_weights.at(i);
        n.weights.assign(packed.begin() + (size_t)i * stride,            packed.begin() + (size_t)i * stride + width);
        n.error_term.assign(packed.begin() + (size_t)i * stride + width, packed.begin() + (size_t)(i + 1) * stride);
    }
}

hidden_layer backward_prop_layer(hidden_layer hl, float learning_rate, std::vector<float> targets, hidden_layer prev_layer = hidden_layer()) {

    std::vector<int> counts, firsts;
    build_partition(hl.layer_weights.size(), counts, firsts);
    int start = firsts[proc_id];
    int end   = start + counts[proc_id];

    if(hl.is_output_layer == IS_OUTPUT_LAYER) {
        for(int i = start; i < end; i++) {
            std::vector<float> single_target = {targets.at(i)};
            hl.layer_weights.at(i) = backward_prop_single_node(hl.layer_weights.at(i), learning_rate, single_target);
        }
    }
    else {
        for(int i = start; i < end; i++) {
            std::vector<float> column_error_terms = find_column_error_terms(prev_layer, i);
            hl.layer_weights.at(i) = backward_prop_single_node(hl.layer_weights.at(i), learning_rate, column_error_terms);
        }
    }

    sync_layer_after_backward(hl, counts, firsts);
    return hl;
}


int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

    create_node_scalars_type();

    //printf("Hello World!");

    // Key Rules
    //      1. the number of weights per node equals the number of nodes from the previous layer
    //      2. you can go bananas with the number of nodes per layer as long as the weight count is maintained

    std::vector<float> inputs = {1.0, 2.0, 3.0}; // three inputs
    std::vector<float> targets = {1.0, 2.0}; 

    std::vector<std::vector<float>> training_inputs(100, std::vector<float>(1, 0)); // three inputs
    std::vector<std::vector<float>> training_targets(100, std::vector<float>(1, 0));

    std::cout << "Generating Test Cases" << std::endl;
    int count = 100;
    for(int i = 0; i < count; i++) {
        //std::cout << training_inputs.at(i).at(i) << std::endl;;
        training_inputs.at(i).at(0) = (float)i / 100.0;
        training_targets.at(i).at(0) = (float)(i * i) / (100.0 * 100.0);

    }

    float learning_rate = 0.1; 

    std::cout << "creating Layer 1" << std::endl;
    hidden_layer_node h0_1;
    h0_1.weights = {1.0}; // nodes with 3 weights for 3 inputs
    h0_1.func = tanh_calc; // activation function for this node
    h0_1.is_output_node = IS_HIDDEN_NODE; // this is a hidden node
    hidden_layer_node h0_2;
    h0_2.weights = {1.0}; // nodes with 3 weights for 3 inputs
    h0_2.func = tanh_calc; // activation function for this node
    h0_2.is_output_node = IS_HIDDEN_NODE; // this is a hidden node
    hidden_layer_node h0_3;
    h0_3.weights = {1.0}; // nodes with 3 weights for 3 inputs
    h0_3.func = tanh_calc; // activation function for this node
    h0_3.is_output_node = IS_HIDDEN_NODE; // this is a hidden node

    hidden_layer h0;
    h0.layer_weights = {h0_1, h0_2, h0_3};
    h0.is_output_layer = IS_HIDDEN_LAYER;

    hidden_layer_node h1_1;
    h1_1.weights = {1.0, 0.5, 0.5}; // nodes with 3 weights for 3 inputs
    h1_1.func = tanh_calc; // activation function for this node
    h1_1.is_output_node = IS_HIDDEN_NODE; // this is a hidden node
    hidden_layer_node h1_2;
    h1_2.weights = {1.0, 0.6, 0.4};
    h1_2.func = tanh_calc; // activation function for this node
    h1_2.is_output_node = IS_HIDDEN_NODE; // this is a hidden node
    hidden_layer_node h1_3;
    h1_3.weights = {1.0, 0.2, 0.1};
    h1_3.func = tanh_calc; // activation function for this node
    h1_3.is_output_node = IS_HIDDEN_NODE; // this is a hidden node

    hidden_layer h1;
    h1.layer_weights = {h1_1, h1_2, h1_3}; // 3 nodes in this layer
    h1.is_output_layer = IS_HIDDEN_LAYER; // this is a hidden layer

    hidden_layer_node h3_1;
    h3_1.weights = {1.0, 0.02, 0.9}; 
    h3_1.func = linear_calc; // activation function for this node
    h3_1.is_output_node = IS_OUTPUT_NODE; // this is a hidden node
    hidden_layer_node h3_2;
    h3_2.weights = {1.0, 2.0, 2.0};
    h3_2.func = linear_calc; // activation function for this node
    h3_2.is_output_node = IS_OUTPUT_NODE; // this is a hidden node
    hidden_layer_node h3_3;
    h3_3.weights = {1.0, 2.0, 1.0};
    h3_3.func = linear_calc; // activation function for this node
    h3_3.is_output_node = IS_OUTPUT_NODE; // this is an output node

    hidden_layer h3;
    h3.layer_weights = {h3_1};//, h3_2}; // only 2 nodes this time
    h3.is_output_layer = IS_OUTPUT_LAYER; // this is a hidden layer
    
    std::cout << "Layers Created" << std::endl;


    double time = -1.0 * MPI_Wtime();

    int epochs = 1000;
    for(int i = 0; i < epochs; i++) {
        for(int k = 0; k < training_inputs.size(); k++) {
            //std::cout << "Epoch " << i << std::endl;
            //std::cout << "first layer" << std::endl;
            h0 = forward_pass_layer(h0, training_inputs.at(k));
            MPI_Barrier(MPI_COMM_WORLD);
            //std::cout << "second layer" << std::endl;
            h1 = forward_pass_layer(h1, h0.outputs);
            MPI_Barrier(MPI_COMM_WORLD);
            //std::cout << "third layer" << std::endl;
            h3 = forward_pass_layer(h3, h1.outputs);
            MPI_Barrier(MPI_COMM_WORLD);

            //std::cout << "backprop third layer" << std::endl;
            h3 = backward_prop_layer(h3, learning_rate, training_targets.at(k));
            //std::cout << "backprop second layer: getting error column" << std::endl;
            std::vector<float> error_column = std::vector<float>();
            for(int j = 0; j < h3.layer_weights.size(); j++) {
                error_column = get_column_error_terms(h3, j);
            }
            //std::cout << "backprop second layer: calculating" << std::endl;
            h1 = backward_prop_layer(h1, learning_rate, error_column, h3);

            //std::cout << "backprop first layer: getting error column" << std::endl;
            for(int j = 0; j < h1.layer_weights.size(); j++) {
                error_column = get_column_error_terms(h1, j);
            }
            h0 = backward_prop_layer(h0, learning_rate, error_column, h1);
        }
        if( (i % 250 == 0 || i == 0 ) && proc_id == 0) {
            std::cout << "Weights after " << i << " Epochs" << std::endl;
            for(auto i : h3.layer_weights) {
                std::cout << "Node " << std::endl;
                for(auto j : i.weights) {
                    std::cout << j << std::endl;

                }
            }
        }
    }

    time += MPI_Wtime();
    

    if(proc_id == 0) {
        std::cout << "Weights after 1000 Epochs" << std::endl;
        for(auto i : h3.layer_weights) {
            std::cout << "Node " << std::endl;
            for(auto j : i.weights) {
                std::cout << j << std::endl;

            }
        }
    

    }

    
    h0 = forward_pass_layer(h0, training_inputs.at(33));
    h1 = forward_pass_layer(h1, h0.outputs);
    h3 = forward_pass_layer(h3, h1.outputs);
        
    if(proc_id == 0) {  
        std::cout << "Accuracy Check: 33^2 = " << h3.outputs.at(0) * 100 * 100 << " (well not really)" << std::endl;
        std::cout << "Time to Complete: " << time << " (s)" << std::endl;
    }

    /*

    Testing

    h1 = forward_pass_layer(h1, inputs); // inputs to hidden layer 1
    h2 = forward_pass_layer(h2, h1.outputs); // outputs from previous layer go to hidden layer 2
    std::vector<float> outputs = h2.outputs;
    std::cout << "Outputs" << std::endl;
    for(auto o : outputs) {
        std::cout << o << std::endl;
    }

    std::cout << "Backpropagation" << std::endl;
    for(auto i : h3_2.weights) {
        std::cout << i << std::endl;
    }
    std::cout << "Inputs" << std::endl;
    for(auto i : h2.layer_weights.at(1).inputs) {
        std::cout << i << std::endl;

    }

    h2.layer_weights.at(1) = backward_prop_single_node(h2.layer_weights.at(1), learning_rate, h1.outputs);

    std::cout << "Done" << std::endl;
    for(auto i : h2.layer_weights.at(1).weights) {
        std::cout << i << std::endl;
    }


    std::cout << "Backpropagation Layer: H2" << std::endl;
    h2 = backward_prop_layer(h2, learning_rate, targets);
    h1 = backward_prop_layer(h1, learning_rate, targets, h2);

    std::cout << "Reults of Back Propagation" << std::endl;
    for(auto i : h2.layer_weights) {
        std::cout << "Node " << std::endl;
        for(auto j : i.weights) {
            std::cout << j << std::endl;

        }
    }

    int epochs = 1000;
    for(int i = 0; i < epochs; i++) {
        for(int k = 0; k < training_inputs.size(); k++) {
            h1 = forward_pass_layer(h1, training_inputs.at(k));
            h2 = forward_pass_layer(h2, h1.outputs);
            h2 = backward_prop_layer(h2, learning_rate, training_targets.at(k));
            h1 = backward_prop_layer(h1, learning_rate, get_column_error_terms(h2, k), h2);
        }
    }

    std::cout << "Reults after 1000 Epochs" << std::endl;
    for(auto i : h2.layer_weights) {
        std::cout << "Node " << std::endl;
        for(auto j : i.weights) {
            std::cout << j << std::endl;

        }
    }

    */

    MPI_Type_free(&node_scalars_t);

    MPI_Finalize();

    return 0;
}

/*

compilation on Said's machine
cl.exe /EHsc /MD "/IC:\Program Files (x86)\Microsoft SDKs\MPI\Include" ann_parallel_test.cpp /Fe:ann_parallel_test.exe /link "/LIBPATH:C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib

running on Said's machine
mpiexec -n 2 ann_parallel_test.exe


*/