#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "neural.h"

// OR GATE
float train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

#define train_count (sizeof(train)/sizeof(train[0]))

float sigmoidf(float x){
    return 1.0f / (1.0f + expf(-x));
}

// Restituisce un float casuale tra 0.0f e 1.0f
float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

float cost(float w1, float w2, float b){
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i = i + 1)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoidf(x1 * w1 + x2 * w2 + b);
        float d = y - train[i][2];
        result += d*d;
    }

    result /= train_count;
    return result;
}
