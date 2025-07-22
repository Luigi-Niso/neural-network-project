#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "neural.h"

float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8}
};

#define train_count (sizeof(train)/sizeof(train[0]))

// Restituisce un float casuale tra 0.0f e 1.0f
float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

float cost(float w, float b){
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i = i + 1)
    {
        float x = train[i][0];
        float y = x * w + b;
        float d = y - train[i][1];
        result += d*d;
    }

    result /= train_count;
    return result;
}
