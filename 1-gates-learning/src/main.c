#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "neural.h"

int main() {
    srand((unsigned int) (time(NULL) * clock()));
    
    float w1 = rand_float() * 10.0f;
    float w2 = rand_float() * 10.0f;
    float b = rand_float() * 5.0f;

    float eps = 1e-3;
    float rate = 1e-2;

    printf("Initial: w1=%f, w2=%f, b=%f, cost=%f\n", w1, w2, b, cost_double(w1, w2, b));
    
    for(size_t i = 0; i < 1000000; i++) {
        float c = cost_double(w1, w2, b);
        float dw1 = (cost_double(w1 + eps, w2, b) - c) / eps;
        float dw2 = (cost_double(w1, w2 + eps, b) - c) / eps;
        float db = (cost_double(w1, w2, b + eps) - c) / eps;
        
        w1 -= rate * dw1;
        w2 -= rate * dw2;
        b -= rate * db;
        
        if(i % 100000 == 0 && i > 0) {
            printf("Iteration %zu: cost=%f\n", i, c);
        }
    }
    
    printf("Final: w1=%f, w2=%f, b=%f, cost=%f\n", w1, w2, b, cost_double(w1, w2, b));
    
    print_results_double(w1, w2, b);

    return 0;
}