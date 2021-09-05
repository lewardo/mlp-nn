//
//  lossfunc.h
//  mlp-nn
//
//  Created by lewardo on 20/12/2020.
//  Copyright © 2020 lewardo. All rights reserved.
//

#pragma once

#include <cstdlib>
#include <cstdbool>
#include <cmath>

#include "utils.h"

using lossfunc_t = struct {
    ii_o_func_t f_x, df_dx;
};

namespace lossfunc {
    static lossfunc_t MSE = {
        [](float x, float t) -> float {
            return 0.5 * (t - x) * (t - x);
        },
        
        [](float x, float t) -> float {
            return (x - t);
        }
    };
    
    static lossfunc_t MAE = {
        [](float x, float t) -> float {
            return 0.5 * abs(t - x);
        },
        
        [](float x, float t) -> float {
            if(x < t) return -0.5;
            else if(x > t) return 0.5;
            
            return 0;
        }
    };
}
