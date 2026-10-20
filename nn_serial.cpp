#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>

const int IS_OUTPUT_NODE = 1;
const int IS_HIDDEN_NODE = 0;
const int IS_OUTPUT_LAYER = 1;
const int IS_HIDDEN_LAYER = 0;

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
    
    std::vector<float> outputs;
    
    for(int i = 0; i < hl.layer_weights.size(); i++) { // for all nodes
        //printf("Size of Layers: %d \n", hl.layer_weights.at(i).weights.size());
        float (*func)(float) = hl.layer_weights.at(i).func; // get the activation function for this node
        hl.layer_weights.at(i).output = forward_pass_single_node(hl.layer_weights.at(i).weights, inputs, func);
        hl.layer_weights.at(i).post_activation_output = hl.layer_weights.at(i).func(hl.layer_weights.at(i).output);
        hl.layer_weights.at(i).inputs = inputs; // store the inputs for backpropagation
        outputs.push_back(hl.layer_weights.at(i).post_activation_output); // get the node output thing
        //std::cout << outputs.back() << std::endl;
    }

    hl.outputs = outputs; // store the outputs for backpropagation
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

hidden_layer backward_prop_layer(hidden_layer hl, float learning_rate, std::vector<float> targets, hidden_layer prev_layer = hidden_layer()) {

    if(hl.is_output_layer == IS_OUTPUT_LAYER) {
        for(int i = 0; i < hl.layer_weights.size(); i++) {
            std::vector<float> single_target = {targets.at(i)};
            hl.layer_weights.at(i) = backward_prop_single_node(hl.layer_weights.at(i), learning_rate, single_target);

        }
    }
    else {
        //std::vector<float> prev_layer_weights = prev_layer.layer_weights ;
        for(int i = 0; i < hl.layer_weights.size(); i++) {
            //std::cout << "getting layer weights" << std::endl;
            std::vector<float> column_error_terms = find_column_error_terms(prev_layer, i);
            //std::cout << "backwarding" << std::endl;
            hl.layer_weights.at(i) = backward_prop_single_node(hl.layer_weights.at(i), learning_rate, column_error_terms);
            //std::cout << "finished backwarding";
        }
    }

    return hl;
}


int main() {
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

    int epochs = 1000;
    for(int i = 0; i < epochs; i++) {
        for(int k = 0; k < training_inputs.size(); k++) {
            
            //std::cout << "first layer" << std::endl;
            h0 = forward_pass_layer(h0, training_inputs.at(k));
            //std::cout << "second layer" << std::endl;
            h1 = forward_pass_layer(h1, h0.outputs);
            //std::cout << "third layer" << std::endl;
            h3 = forward_pass_layer(h3, h1.outputs);

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
        if(i % 250 == 0 || i == 0) {
            std::cout << "Weights after " << i << " Epochs" << std::endl;
            for(auto i : h3.layer_weights) {
                std::cout << "Node " << std::endl;
                for(auto j : i.weights) {
                    std::cout << j << std::endl;

                }
            }
        }
    }

    std::cout << "Weights after 1000 Epochs" << std::endl;
    for(auto i : h3.layer_weights) {
        std::cout << "Node " << std::endl;
        for(auto j : i.weights) {
            std::cout << j << std::endl;

        }
    }

    std::cout << "Accuracy Check" << std::endl;
    h0 = forward_pass_layer(h0, training_inputs.at(33));
    h1 = forward_pass_layer(h1, h0.outputs);
    h3 = forward_pass_layer(h3, h1.outputs);
    
    std::cout << h3.outputs.at(0) * 100 * 100 << std::endl;

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

    return 0;
}