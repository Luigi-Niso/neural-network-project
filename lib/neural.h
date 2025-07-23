#ifndef NEURAL_H
#define NEURAL_H

#include <stddef.h>

// Strutture per reti neurali complesse
typedef struct {
    float or_w1;
    float or_w2;
    float or_b;

    float nand_w1;
    float nand_w2;
    float nand_b;

    float and_w1;
    float and_w2;
    float and_b;
} Xor;

// Dataset di training (dichiarato come extern)
extern float train[][3];
extern size_t train_count;

// Funzioni di base
float sigmoidf(float x);
float rand_float(void);

// Funzioni di costo - versione semplice (1 input)
float cost_simple(float w, float b);

// Funzioni di costo - versione con 2 input
float cost_double(float w1, float w2, float b);

// Funzioni per reti complesse
float forward_xor(Xor m, float x1, float x2);
Xor rand_xor(void);
float cost_xor(Xor m);

// Funzioni di utilità
void print_results_simple(float w, float b);
void print_results_double(float w1, float w2, float b);

#endif // NEURAL_H
