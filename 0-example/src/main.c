#include <stdlib.h>
#include <stdio.h>
#include <time.h>

float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8}
};

#define train_count (sizeof(train) / sizeof(train[0]))

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