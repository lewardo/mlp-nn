//
//  main.cpp
//  mlp-nn
//
//  Created by lewardo on 19/06/2020.
//  Copyright © 2020 lewardo. All rights reserved.
//

#include <cstdio>
#include <vector>
#include <ctime>

#include "lib/mlp.h"

#define NUM_EPOCHS (10e4)
#define LEARNING_RATE (0.03)
#define BATCH_SIZE (4)
#define MOMENTUM 0
#define SOFTMAX 0


#define TRAIN 1
#define PREDICT 0


tensor_t data = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1},
}, target = {
    {0},
    {1},
    {1},
    {0},
};

void draw_logic_output(mlp* netw, float inc) {
    std::vector<float> d00, d01, d10, d11;
    
    netw->predict(data[0], d00, SOFTMAX);
    netw->predict(data[1], d01, SOFTMAX);
    netw->predict(data[2], d10, SOFTMAX);
    netw->predict(data[3], d11, SOFTMAX);
    
    printf("%.2f\t\t  %.2f\n", d00[0], d01[0]);
    for(float i = 0.0; i <= 1; i += inc){
        printf("\t");
        for(float j = 0.0; j <= 1; j += inc) {
            std::vector<float> iv = {i, j}, ov = {0};
            
            netw->predict(iv, ov, SOFTMAX);
            
            if(ov[0] > 0.75) printf("█");
            else if(ov[0] > 0.5) printf("▒");
            else if(ov[0] > 0.25) printf("░");
            else printf(" ");
        }
        printf("\n");
    }
    printf("%.2f\t\t  %.2f\n", d10[0], d11[0]);
}

int main(void) {
    float err = 0.0;
    int mode = PREDICT;

    srand((unsigned int) time(NULL));

    try {
        if(mode) {
            mlp nn({
                {2, 2, 1},              // neurons per layer
                actfunc::tanh,          // activation function
                LEARNING_RATE,          //
                MOMENTUM,               //
            });
            
            err = nn.regress(data, target, NUM_EPOCHS, BATCH_SIZE);
            nn.save("save.data");
            
            printf("%.2e\n", err);
        } else {
            mlp nn;
            nn.load("save.data");
            
            draw_logic_output(&nn, 0.1);
        }
        
    } catch(int e) {
        switch (e) {
            case MLP_ARG_ERROR:
                fprintf(stderr, "argument error");
                break;

            case MLP_ERROR_SIZE_MISMATCH:
                fprintf(stderr, "vector size mismatch");
                break;

            default:
                fprintf(stderr, "exception somewhere");
                break;
        }
    }
}
