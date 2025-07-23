#ifndef NEURAL_H
#define NEURAL_H

// Dataset di training
extern float train[][3];


// Funzioni del neural network
float sigmoidf(float x);
float rand_float(void);
float cost(float w1, float w2, float b);

#endif // NEURAL_H
