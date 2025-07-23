#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "neural.h"

// Dataset XOR locale (sovrascrive quello globale)
static float xor_train[][3] = {
    {0, 0, 0},  // 0 XOR 0 = 0
    {0, 1, 1},  // 0 XOR 1 = 1
    {1, 0, 1},  // 1 XOR 0 = 1
    {1, 1, 0},  // 1 XOR 1 = 0
};

static size_t xor_train_count = sizeof(xor_train)/sizeof(xor_train[0]);

// Funzione di costo locale per XOR
float cost_xor_local(Xor m) {
    float result = 0.0f;
    for (size_t i = 0; i < xor_train_count; i++) {
        float x1 = xor_train[i][0];
        float x2 = xor_train[i][1];
        float y = forward_xor(m, x1, x2);
        float d = y - xor_train[i][2];
        result += d * d;
    }
    result /= xor_train_count;
    return result;
}

int main() {
    srand((unsigned int) (time(NULL) * clock()));
    
    // Inizializza la rete XOR con pesi casuali
    Xor xor_model = rand_xor();
    
    float eps = 1e-3;
    float rate = 1e-1;  // Learning rate più alto per reti complesse
    
    printf("Initial XOR model cost: %f\n", cost_xor_local(xor_model));
    
    // Training per 100,000 iterazioni
    for(size_t i = 0; i < 100000; i++) {
        float c = cost_xor_local(xor_model);
        
        // Calcola gradienti per tutti i pesi (finite difference)
        Xor grad;
        
        // Gradienti per OR gate
        grad.or_w1 = (cost_xor_local((Xor){xor_model.or_w1 + eps, xor_model.or_w2, xor_model.or_b,
                                      xor_model.nand_w1, xor_model.nand_w2, xor_model.nand_b,
                                      xor_model.and_w1, xor_model.and_w2, xor_model.and_b}) - c) / eps;
        
        grad.or_w2 = (cost_xor_local((Xor){xor_model.or_w1, xor_model.or_w2 + eps, xor_model.or_b,
                                      xor_model.nand_w1, xor_model.nand_w2, xor_model.nand_b,
                                      xor_model.and_w1, xor_model.and_w2, xor_model.and_b}) - c) / eps;
        
        grad.or_b = (cost_xor_local((Xor){xor_model.or_w1, xor_model.or_w2, xor_model.or_b + eps,
                                     xor_model.nand_w1, xor_model.nand_w2, xor_model.nand_b,
                                     xor_model.and_w1, xor_model.and_w2, xor_model.and_b}) - c) / eps;
        
        // Gradienti per NAND gate
        grad.nand_w1 = (cost_xor_local((Xor){xor_model.or_w1, xor_model.or_w2, xor_model.or_b,
                                       xor_model.nand_w1 + eps, xor_model.nand_w2, xor_model.nand_b,
                                       xor_model.and_w1, xor_model.and_w2, xor_model.and_b}) - c) / eps;
        
        grad.nand_w2 = (cost_xor_local((Xor){xor_model.or_w1, xor_model.or_w2, xor_model.or_b,
                                       xor_model.nand_w1, xor_model.nand_w2 + eps, xor_model.nand_b,
                                       xor_model.and_w1, xor_model.and_w2, xor_model.and_b}) - c) / eps;
        
        grad.nand_b = (cost_xor_local((Xor){xor_model.or_w1, xor_model.or_w2, xor_model.or_b,
                                      xor_model.nand_w1, xor_model.nand_w2, xor_model.nand_b + eps,
                                      xor_model.and_w1, xor_model.and_w2, xor_model.and_b}) - c) / eps;
        
        // Gradienti per AND gate
        grad.and_w1 = (cost_xor_local((Xor){xor_model.or_w1, xor_model.or_w2, xor_model.or_b,
                                      xor_model.nand_w1, xor_model.nand_w2, xor_model.nand_b,
                                      xor_model.and_w1 + eps, xor_model.and_w2, xor_model.and_b}) - c) / eps;
        
        grad.and_w2 = (cost_xor_local((Xor){xor_model.or_w1, xor_model.or_w2, xor_model.or_b,
                                      xor_model.nand_w1, xor_model.nand_w2, xor_model.nand_b,
                                      xor_model.and_w1, xor_model.and_w2 + eps, xor_model.and_b}) - c) / eps;
        
        grad.and_b = (cost_xor_local((Xor){xor_model.or_w1, xor_model.or_w2, xor_model.or_b,
                                     xor_model.nand_w1, xor_model.nand_w2, xor_model.nand_b,
                                     xor_model.and_w1, xor_model.and_w2, xor_model.and_b + eps}) - c) / eps;
        
        // Applica i gradienti
        xor_model.or_w1 -= rate * grad.or_w1;
        xor_model.or_w2 -= rate * grad.or_w2;
        xor_model.or_b -= rate * grad.or_b;
        
        xor_model.nand_w1 -= rate * grad.nand_w1;
        xor_model.nand_w2 -= rate * grad.nand_w2;
        xor_model.nand_b -= rate * grad.nand_b;
        
        xor_model.and_w1 -= rate * grad.and_w1;
        xor_model.and_w2 -= rate * grad.and_w2;
        xor_model.and_b -= rate * grad.and_b;
        
        // Progress report
        if(i % 10000 == 0 && i > 0) {
            printf("Iteration %zu: cost=%f\n", i, c);
        }
    }
    
    printf("Final XOR model cost: %f\n", cost_xor_local(xor_model));
    
    // Testa il modello XOR
    printf("\n=== XOR Gate Results ===\n");
    for (size_t i = 0; i < xor_train_count; i++) {
        float x1 = xor_train[i][0];
        float x2 = xor_train[i][1];
        float expected = xor_train[i][2];
        float actual = forward_xor(xor_model, x1, x2);
        
        printf("Input: (%.0f,%.0f) | Expected: %.0f | Actual: %.6f | Error: %.6f\n", 
               x1, x2, expected, actual, fabsf(expected - actual));
    }
    
    return 0;
}