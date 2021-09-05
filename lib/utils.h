//
//  utils.h
//  mlp-nn
//
//  Created by lewardo on 18/12/2020.
//  Copyright © 2020 lewardo. All rights reserved.
//

#pragma once

#include <cstdlib>


using tensor_t = std::vector<std::vector<float>>;

using i_o_func_t = float (*)(float);
using ii_o_func_t = float (*)(float, float);


namespace utils {
    inline float random() {
        return ((float) rand() / (float) RAND_MAX) - 0.5f;  // rand func between ±0.5
    }
}
