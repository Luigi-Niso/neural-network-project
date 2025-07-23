#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "neural.h"

// Dataset di training predefinito (può essere overridden)
float train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

size_t train_count = sizeof(train)/sizeof(train[0]);

// Funzione sigmoid
float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Generatore di numeri casuali
float rand_float(void) {
    return (float) rand() / (float) RAND_MAX;
}

// Funzione di costo per modello semplice (1 input)
float cost_simple(float w, float b) {
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x = train[i][0]; // Solo primo input
        float y = sigmoidf(x * w + b);
        float d = y - train[i][2];
        result += d * d;
    }
    result /= train_count;
    return result;
}

// Funzione di costo per modello con 2 input
float cost_double(float w1, float w2, float b) {
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoidf(x1 * w1 + x2 * w2 + b);
        float d = y - train[i][2];
        result += d * d;
    }
    result /= train_count;
    return result;
}

// Forward pass per XOR (rete complessa)
float forward_xor(Xor m, float x1, float x2) {
    float a = sigmoidf(x1 * m.or_w1 + x2 * m.or_w2 + m.or_b);
    float b = sigmoidf(x1 * m.nand_w1 + x2 * m.nand_w2 + m.nand_b);
    return sigmoidf(a * m.and_w1 + b * m.and_w2 + m.and_b);
}

// Inizializzazione casuale per XOR
Xor rand_xor(void) {
    Xor m;
    m.or_w1 = rand_float();
    m.or_w2 = rand_float();
    m.or_b = rand_float();
    
    m.nand_w1 = rand_float();
    m.nand_w2 = rand_float();
    m.nand_b = rand_float();
    
    m.and_w1 = rand_float();
    m.and_w2 = rand_float();
    m.and_b = rand_float();
    
    return m;
}

// Funzione di costo per XOR
float cost_xor(Xor m) {
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward_xor(m, x1, x2);
        float d = y - train[i][2];
        result += d * d;
    }
    result /= train_count;
    return result;
}

// Stampa risultati per modello semplice
void print_results_simple(float w, float b) {
    printf("\n=== Results ===\n");
    for (size_t i = 0; i < train_count; i++) {
        float x = train[i][0];
        float prediction = sigmoidf(x * w + b);
        printf("Input: %.0f | Expected: %.0f | Actual: %.6f\n", 
               x, train[i][2], prediction);
    }
}

// Stampa risultati per modello con 2 input
void print_results_double(float w1, float w2, float b) {
    printf("\n=== Results ===\n");
    for (size_t i = 0; i < train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float prediction = sigmoidf(x1 * w1 + x2 * w2 + b);
        printf("Input: (%.0f,%.0f) | Expected: %.0f | Actual: %.6f\n", 
               x1, x2, train[i][2], prediction);
    }
}
