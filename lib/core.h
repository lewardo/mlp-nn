//
//  core.hpp
//  mlp-nn
//
//  Created by lewardo on 21/12/2020.
//  Copyright © 2020 lewardo. All rights reserved.
//

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdbool>
#include <string>
#include <vector>

#include "layer.h"
#include "actfunc.h"
#include "params.h"
#include "utils.h"

enum {
    CORE_RETURN_OK = 0,
    CORE_UNINITIALISED,
    CORE_ARG_ERROR,
    CORE_ERROR_SIZE_MISMATCH,
    CORE_FILE_FAILED_OPEN,
};

class mlp_core {
protected:
    /* layers */
    std::vector<layer> layers;
    std::vector<int> npl;
    int nl;
    
    /* hyperparameters */
    float lr;
    actfunc_t af;
    
    /* is initialised */
    bool initialised = false;

    /* error calculation stuffs */
    float running_err = 1.0;
    int avg_len = 16;
    
    /* internal functions */
    float __train(std::vector<float>& input, std::vector<float>& target, bool update, float rate);
    int __predict(std::vector<float>& input, std::vector<float>& output);
    
public:
    /* constructor and initialise mlp core */
    int init_core(mlp_param_t params);
        
    /* topology stuff */
    int updt_params(mlp_param_t params);
    virtual int updt_layer(int layer, int nn, actfunc_t af = actfunc::tanh) = 0;
    
    /* save and load topologies */
    int load(std::string dir),
        save(std::string dir);
};

