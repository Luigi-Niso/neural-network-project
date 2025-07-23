#include <stdio.h>
#include <stdlib.h>
#include "neural.h"
#include <time.h>

int main() {
    srand((unsigned int) (time(NULL) * clock()));
    float w1 = rand_float() * 10.0f;
    float w2 = rand_float() * 10.0f;
    float b = rand_float() * 5.0f;

    float eps = 1e-3;
    float rate = 1e-2;

    printf("w1=%f, w2=%f, b=%f, cost=%f\n", w1, w2, b, cost(w1, w2 ,b));
    for(size_t i = 0; i < 1e7; i = i + 1){
        float dw1 = (cost(w1 + eps, w2, b) - cost(w1, w2, b))/eps;
        float dw2 = (cost(w1, w2 + eps, b) - cost(w1, w2, b))/eps;
        float db = (cost(w1, w2, b + eps) - cost(w1, w2, b))/eps;
        w1 -= rate * dw1;
        w2 -= rate * dw2;
        b -= rate * db; 
    }
    
    printf("w1=%f, w2=%f, b=%f, cost=%f\n", w1, w2, b, cost(w1, w2 ,b));

    for (size_t i = 0; i < 4; i++)
    {
        float y = sigmoidf((train[i][0] * w1 + train[i][1] * w2 + b));
        printf("Expected: %f - Actual: %f\n", train[i][2], y);
    }
    

    return 0;
}