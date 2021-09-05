//
//  params.h
//  mlp-nn
//
//  Created by lewardo on 13/07/2020.
//  Copyright © 2020 lewardo. All rights reserved.
//

#pragma once

#include <cstdlib>
#include <cstdbool>
#include <vector>

using mlp_param_t = struct {
    /* neurons per layer */
    std::vector<int> npl;
    
    /* activation and loss functions */
    actfunc_t af = actfunc::tanh;
    
    /* hyperparameters */
    float lr = 0.03, mo = 0.75;
};


