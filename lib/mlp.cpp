//
//  mlp.h
//  mlp-nn
//
//  Created by lewardo on 19/6/2020.
//  Copyright © 2019 lewardo. All rights reserved.
//

#include <cstdlib>
#include <cstdbool>
#include <vector>

#include "mlp.h"
#include "layer.h"
#include "params.h"

mlp::mlp(mlp_param_t params) {
    init_core(params);
}

mlp::mlp(std::vector<int> init, actfunc_t actfunc) {
    init_core({init, actfunc});
};

int mlp::push_layer(int num, actfunc_t af) {
    layer new_layer = layer(num, 0, af);
    layers.push_back(new_layer);
    
    layers[nl - 1].update_next(num);
    
    npl[nl] = new_layer.nn;
    npl.push_back(0);
    
    nl++;
    
    return MLP_RETURN_OK;
};

int mlp::updt_layer(int num, int nn, actfunc_t af) {
    if(num > nl - 1 || num < 0) throw MLP_ARG_ERROR;
    
    layer* prev = (num == 0) ? nullptr : &layers[num - 1], * next = (num == nl - 1) ? nullptr : &layers[num + 1]; 
    
    layers[num].set_nn(nn, prev, next, af);
    
    return MLP_RETURN_OK;
};

int mlp::predict(std::vector<float>& input, std::vector<float>& output, bool softmax) {
    try {
        __predict(input, output);
    } catch(int n) {
        throw;
    };
    
    if(softmax) {
        float sum = 0.0;
        std::vector<float> exp_vec;
        
        for(int n = 0; n < layers[nl - 1].nn; n++) {
            float exp = expf(output[n]);
            
            sum += exp;
            exp_vec.push_back(exp);
        }
        
        for(int n = 0; n < layers[nl - 1].nn; n++)
            output[n] = exp_vec[n] / sum;
    }
    
    return layers[nl - 1].nn;
};

float mlp::regress(std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& target, int max_iters, int batch_size, float stop_err) {
    int epochs = 0, num_ins = (int) input.size();
    float err = 0;
    
    if(num_ins != target.size() || batch_size > num_ins) throw MLP_ERROR_SIZE_MISMATCH;
    if(max_iters < 0 || batch_size < 1 || num_ins < 1 || stop_err < 0) throw MLP_ARG_ERROR;
    
    if(max_iters == 0) {
        if(stop_err == 0) throw MLP_ARG_ERROR;
        else max_iters = MAX_EPOCHS;
    }
    
    for(int n = 0; n < num_ins; n++)
        if(input[n].size() != layers[0].nn || target[n].size() != layers[nl - 1].nn) throw MLP_ERROR_SIZE_MISMATCH;
    
    do {
        int n = 0, offset = (epochs * batch_size) % num_ins;
        
        while(n++ < batch_size) {
            int set = (offset + n) % num_ins;
            
            try {
                __train(input[set], target[set], false, lr);
            } catch(int n) {
                throw;
            };
        }
        
        try {
            err = __train(input[epochs % num_ins], target[epochs % num_ins], true, lr);
        } catch(int n) {
            throw;
        };
    } while(epochs++ < max_iters && err > stop_err);
    
    return err;
};

