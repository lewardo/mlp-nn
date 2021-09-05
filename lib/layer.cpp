//
//  layer.cpp
//  mlp-nn
//
//  Created by lewardo on 21/12/2020.
//  Copyright © 2020 lewardo. All rights reserved.
//

#include <cstdlib>
#include <cstdio>

#include "layer.h"

layer::layer(int num, int next_num, actfunc_t af) {
    nn = num;
    actfunc = af;
    
    for(int n = 0; n < nn; n++) {
        neuron node = neuron(next_num);
        neurons.push_back(node);
    }
};

void layer::assign(const std::vector<float>& list) {
    for (int n = 0; n < list.size(); n++)
        neurons[n].val = list[n];
}

void layer::set_nn(int num, layer* prev, layer* next, actfunc_t af) {
    int nnext = (next == nullptr) ? 0 : next->nn;
    neurons.resize(num, neuron(nnext));
    
    set_actfunc(af.identifier);
    
    if(prev) {
        for(int n = 0; n < prev->nn; n++) {
            prev->neurons[n].w.resize(num, utils::random());
            prev->neurons[n].dw.resize(num, utils::random());
        }
    }
}

void layer::update_next(int next_num) {
    for(int n = 0; n < nn; n++) {
        neurons[n].nw = next_num;
        
        neurons[n].w.resize(next_num, utils::random());
        neurons[n].dw.resize(next_num, utils::random());
    }
}

void layer::set_actfunc(actfunc_id identifier) {
    switch (identifier) {
        case SIGMOID:
            actfunc = actfunc::sigmoid;
            break;
            
        case TANH:
            actfunc = actfunc::tanh;
            break;
            
        case RELU:
            actfunc = actfunc::ReLU;
            break;
            
        case LINEAR:
            actfunc = actfunc::lin;
            break;
            
        default:
            throw;
    }
}

void layer::propagate(layer* next) {
    for(int n = 0; n < next->nn; n++) {
        float sum = 0.0f; // sum of weighted values
        
        for(int t = 0; t < nn; t++)
            sum += neurons[t].val * neurons[t].w[n]; // previous neuron value * connecting weight
        
        next->neurons[n].val = next->actfunc.f_x(sum + next->neurons[n].b); // set next value with bias and activation function
    }
};

void layer::backtrack(layer* prev, std::vector<float>& target, bool last, bool update, float lr, lossfunc_t lf) {
    for(int t = 0; t < prev->nn; t++) { // every neuron in prev layer
        float dn, dw, error = 0.0f; //dE/dz[Lk], dE/dw[Ljk]
        
        for(int n = 0; n < nn; n++) { // loop throught all neurons connected to previous neuron
            if(last) dn = lf.df_dx(neurons[n].val, target[n]) * actfunc.df_dx(neurons[n].val);
            else dn = neurons[n].err; // dE/dz[Lk]
            
            dw = dn * prev->neurons[t].val; // dE/dw[Ljk]
            
            prev->neurons[t].dw[n] += dw * lr;
            neurons[n].db += dn * lr;
            
            error += dn * prev->neurons[t].w[n]; // prev neuron error
            
            if(update) {
                prev->neurons[t].w[n] -= prev->neurons[t].dw[n]; // subrtact error prop. to learning rate
                neurons[n].b -= neurons[n].db; // same but w/o the prev neuron factor
                
                prev->neurons[t].w[n] = __constrain(prev->neurons[t].w[n], -w_max, w_max);
                neurons[n].b = __constrain(neurons[n].b, -b_max, b_max); // counstrain to stop exploding weigts
                
                prev->neurons[t].dw[n] = 0.0f;
                neurons[n].db = 0.0f; // reset the running totals for the batches
            }
        }
        
        prev->neurons[t].err = error * actfunc.df_dx(prev->neurons[t].val); // set prev neuron error
    }
};

