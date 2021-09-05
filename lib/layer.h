//
//  layer.h
//  mlp-nn
//
//  Created by lewardo on 19/06/2020.
//  Copyright © 2019 lewardo. All rights reserved.
//

#pragma once

#include <cstdlib>
#include <vector>

#include "actfunc.h"
#include "lossfunc.h"
#include "params.h"
#include "utils.h"

#define __constrain(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))

class layer {
private:
    float w_max = 16.0f, b_max = 16.0f; // weight range to avoid exploding gradients
    actfunc_t actfunc; // activation function
    
    
    struct __attribute__((packed)) neuron {
        float val,
        err;
        
        float b, db;
        
        int nw;
        std::vector<float> w, dw;
        
        neuron(int n) : val(0.0), err(0.0), b(utils::random()), db(0.0), nw(n) {
            for(int cw = 0; cw < n; cw++) {
                w.push_back((float) utils::random());
                dw.push_back(0.0f);
            }
        }
    };
    
public:
    int nn; // number of neurons
    std::vector<neuron> neurons;    // the neurons themselves
    
    /* constructor */
    layer(int num, int next_num = 0, actfunc_t af = actfunc::sigmoid); // constructor that initialises all the neurons, weights, biases and hyperparameters

    /* operators */
    void assign(const std::vector<float>& list); // a hard coded '=' but in function style
    
    /* topology */
    void set_nn(int num, layer* prev, layer* next, actfunc_t af); // to update the number of neurons
    void update_next(int next_num); // update number of neurons being fed into
    void set_actfunc(actfunc_id identifier); // update the activation function
    
    /* functionality */
    void propagate(layer* next); // propagate to pointer of next layer
    void backtrack(layer* prev, std::vector<float>& target, bool last, bool update, float lr, lossfunc_t lf); // backpropagate with repect to previous layer and update prev. layer error
    
    friend class mlp_core;
};

