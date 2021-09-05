//
//  core.cpp
//  mlp-nn
//
//  Created by lewardo on 21/12/2020.
//  Copyright © 2020 lewardo. All rights reserved.
//

#include "core.h"

int mlp_core::init_core(mlp_param_t params) {
    nl = (int) params.npl.size();
    npl = params.npl;
    
    lr = params.lr;
    af = params.af;
    //    lf = lossfunc::MSE;
    
    initialised = true;
    
    npl.push_back(0); // just to simplify initialisation statement
    
    for(int n = 0; n < nl; n++) {
        layer level = layer(npl[n], npl[n + 1], af);
        layers.push_back(level);
    }
    
    return CORE_RETURN_OK;
}

int mlp_core::updt_params(mlp_param_t params) {
    if(!initialised) throw CORE_UNINITIALISED;
    
    lr = params.lr;
    af = params.af;
    //    lf = params.lf;
    
    return CORE_RETURN_OK;
};

int mlp_core::__predict(std::vector<float>& input, std::vector<float>& output) {
    if(!initialised) throw CORE_UNINITIALISED;
    if(layers[0].nn != input.size()) throw CORE_ERROR_SIZE_MISMATCH;
    
    output.clear();
    layers[0].assign(input);
    
    for (int n = 0; n < nl - 1; n++)
        layers[n].propagate(&layers[n + 1]);
    
    for (int n = 0; n < layers[nl - 1].nn; n++)
        output.push_back((float) layers[nl - 1].neurons[n].val);
    
    return CORE_RETURN_OK;
};

float mlp_core::__train(std::vector<float>& input, std::vector<float>& target, bool update, float rate) {
    if(!initialised) throw CORE_UNINITIALISED;
    
    if(layers[0].nn == input.size() && layers[nl - 1].nn == target.size()) layers[0].assign(input);
    else throw CORE_ERROR_SIZE_MISMATCH;
    
    float error = 0.0;
    
    for(int n = 0; n < nl - 1; ++n)
        layers[n].propagate(&layers[n + 1]);
    
    for(int n = nl - 1; n > 0; --n)
        layers[n].backtrack(&layers[n - 1], target, n == (nl - 1), update, rate, lossfunc::MSE);
    
    for(int n = 0; n < target.size(); n++)
        error += lossfunc::MSE.f_x(layers[nl - 1].neurons[n].val, target[n]);
    
    running_err -= running_err / avg_len;
    running_err += error / avg_len;
    
    return running_err;
};

int mlp_core::load(std::string dir) {
    FILE * src;
    std::vector<int> parsed;
    
    if((src = fopen(dir.c_str(), "rb")) == NULL) throw CORE_FILE_FAILED_OPEN;
    
    fscanf(src, "%d", &nl);
    for(int n = 0; n < nl; n++) {
        int nn;
        
        fscanf(src, "%d", &nn);
        parsed.push_back(nn);
    }
    
    fscanf(src, "%f", &lr);
    
    init_core((mlp_param_t) {parsed, actfunc::null, lr});
    
    for(int m = 1; m < nl; m++) {
        int af_id = 0;
        
        for(int n = 0; n < npl[m]; n++) {
            for(int p = 0; p < npl[m - 1]; p++) fscanf(src, "%f", &layers[m - 1].neurons[p].w[n]);
            
            fscanf(src, "%f", &layers[m].neurons[n].b);
        }
        
        fscanf(src, "%d", &af_id);
        layers[m].set_actfunc((actfunc_id) af_id);
    }
    
    fclose(src);
    
    return CORE_RETURN_OK;
}

int mlp_core::save(std::string dir) {
    FILE * src;
    
    if(!initialised) throw CORE_UNINITIALISED;
    if((src = fopen(dir.c_str(), "wb")) == NULL) throw CORE_FILE_FAILED_OPEN;
    
    fprintf(src, "%d\n", nl);
    for(int n = 0; n < nl; n++) fprintf(src, "%d\t", npl[n]);
    
    fprintf(src, "%f\n", lr);
    
    for(int m = 1; m < nl; m++) {
        for(int n = 0; n < npl[m]; n++) {
            for(int p = 0; p < npl[m - 1]; p++) fprintf(src, "%f\t", layers[m - 1].neurons[p].w[n]);
            fprintf(src, "%f\t", layers[m].neurons[n].b);
        }
        
        fprintf(src, "%d\n", layers[m].actfunc.identifier);
    }
    
    fclose(src);
    
    return CORE_RETURN_OK;
}
