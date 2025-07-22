#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "neural.h"

int main() {
    srand((unsigned int) (time(NULL) * clock()));
    float w = rand_float() * 10.0f;
    float b = rand_float() * 5.0f;

    float eps = 1e-3;
    float rate = 1e-3;

    printf("w=%f, cost=%f\n", w, cost(w,b));
    for(size_t i = 0; i < 1000*1000; i = i + 1){
        float dw = (cost(w + eps, b) - cost(w, b))/eps;
        float db = (cost(w, b + eps) - cost(w, b))/eps;
        w -= rate * dw;
        b -= rate * db;
    }
    
    printf("w=%f, b=%f cost=%f\n", w,b, cost(w,b));
    printf("\nFinal w: %f\n", w + b);
    return 0;
}