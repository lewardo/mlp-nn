//
//  mlp.h
//  mlp-nn
//
//  Created by lewardo on 19/06/2020.
//  Copyright © 2020 lewardo. All rights reserved.
//

#pragma once

#include <cstdlib>
#include <vector>

#include "core.h"
#include "layer.h"
#include "actfunc.h"
#include "params.h"
#include "utils.h"

#define MAX_EPOCHS 10e16


enum {
    MLP_RETURN_OK = 0,
    MLP_ARG_ERROR,
    MLP_ERROR_SIZE_MISMATCH,
};

class mlp : public mlp_core {    
public:
    /* constructors */
    using mlp_core::mlp_core;
    mlp(mlp_param_t params);
    mlp(std::vector<int> npl, actfunc_t actfunc = actfunc::tanh); // declare and initialise nn
    
    /* topology stuff */
    int push_layer(int num, actfunc_t af = actfunc::tanh);
    int updt_layer(int layer, int nn, actfunc_t af = actfunc::tanh);
    
    /* functions */    
    float regress(std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& target, int max_iters = 10e4, int batch_size = 1, float stop_err = 0); // train nn on set of data
    int predict(std::vector<float>& input, std::vector<float>& output, bool softmax); // predict output from given input
};
