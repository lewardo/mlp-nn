//
//  actfunc.h
//  mlp-nn
//
//  Created by lewardo on 19/06/2020.
//  Copyright © 2020 lewardo. All rights reserved.
//

#pragma once

#include <cstdlib>
#include <cmath>

#include "utils.h"

enum actfunc_id {
    UNSET,
    SIGMOID = 0,
    TANH,
    RELU,
    LINEAR,
};

using actfunc_t = struct {
    enum actfunc_id identifier;
    i_o_func_t f_x, df_dx;
};

namespace actfunc {
    static actfunc_t sigmoid = {
        (actfunc_id) SIGMOID,
        [](float x) -> float {
            return 1.0f / (1.0f + expf(-x));
        },
        
        [](float x) -> float {
            return x * (1.0f - x);
        }
    };
    
    static actfunc_t tanh = {
        (actfunc_id) TANH,
        [](float x) -> float {
            return tanhf(x);
        },
        
        [](float x) -> float {
            return 1 - (x * x);
        }
    };
    
    static actfunc_t ReLU = {
        (actfunc_id) RELU,
        [](float x) -> float {
            return (x < 0) ? 0 : x;
        },
        
        [](float x) -> float {
            return (x < 0) ? 0 : 1;
        }
    };
    
    static actfunc_t lin = {
        (actfunc_id) LINEAR,
        [](float x) -> float {
            return x;
        },
        
        [](float x) -> float {
            return 1;
        }
    };
    
    static actfunc_t null = {
        (actfunc_id) UNSET,
        nullptr,
        nullptr,
    };
}

