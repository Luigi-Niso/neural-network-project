#include "unity.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "../../src/neural.h"

#define train_count 5
#define EPSILON 1e-3

void setUp(void) {
    // Setup per ogni test
    srand(12345); // Seed fisso per test riproducibili
}

void tearDown(void) {
    // Cleanup dopo ogni test
}

// Test della funzione rand_float
void test_rand_float_range(void) {
    for(int i = 0; i < 100; i++) {
        float val = rand_float();
        TEST_ASSERT_TRUE_MESSAGE(val >= 0.0f, "rand_float should be >= 0.0");
        TEST_ASSERT_TRUE_MESSAGE(val <= 1.0f, "rand_float should be <= 1.0");
    }
}

// Test del dataset di training
void test_training_data_correctness(void) {
    TEST_ASSERT_EQUAL_FLOAT(0.0f, train[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, train[0][1]);
    
    TEST_ASSERT_EQUAL_FLOAT(1.0f, train[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, train[1][1]);
    
    TEST_ASSERT_EQUAL_FLOAT(4.0f, train[4][0]);
    TEST_ASSERT_EQUAL_FLOAT(8.0f, train[4][1]);
}

// Test della relazione lineare y = 2x
void test_linear_relationship(void) {
    for(int i = 0; i < train_count; i++) {
        float expected_y = 2.0f * train[i][0];
        TEST_ASSERT_EQUAL_FLOAT_MESSAGE(expected_y, train[i][1], 
            "Training data should follow y = 2x relationship");
    }
}

// Test della funzione cost con valori perfetti
void test_cost_function_perfect_weights(void) {
    // Con w=2, b=0 il costo dovrebbe essere ~0
    float perfect_cost = cost(2.0f, 0.0f);
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(EPSILON, 0.0f, perfect_cost,
        "Cost should be near 0 with perfect weights w=2, b=0");
}

// Test della funzione cost con pesi sbagliati
void test_cost_function_wrong_weights(void) {
    // Con w=0, b=0 il costo dovrebbe essere alto
    float bad_cost = cost(0.0f, 0.0f);
    TEST_ASSERT_TRUE_MESSAGE(bad_cost > 10.0f, 
        "Cost should be high with wrong weights");
    
    // Con w=10, b=5 il costo dovrebbe essere molto alto
    float worse_cost = cost(10.0f, 5.0f);
    TEST_ASSERT_TRUE_MESSAGE(worse_cost > bad_cost,
        "Cost should increase with worse weights");
}

// Test che il costo diminuisca durante il training
void test_gradient_descent_convergence(void) {
    float w = 5.0f;  // Peso iniziale sbagliato
    float b = 3.0f;  // Bias iniziale sbagliato
    float eps = 1e-3;
    float rate = 1e-2; // Learning rate più alto per test veloce
    
    float initial_cost = cost(w, b);
    
    // Fai più step di gradient descent per una migliore convergenza
    for(int i = 0; i < 500; i++) {
        float dw = (cost(w + eps, b) - cost(w, b))/eps;
        float db = (cost(w, b + eps) - cost(w, b))/eps;
        w -= rate * dw;
        b -= rate * db;
    }
    
    float final_cost = cost(w, b);
    
    TEST_ASSERT_TRUE_MESSAGE(final_cost < initial_cost,
        "Cost should decrease after training");
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(0.5f, 2.0f, w,
        "Weight should converge near 2.0");
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1.0f, 0.0f, b,
        "Bias should converge near 0.0");
}

int main(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_rand_float_range);
    RUN_TEST(test_training_data_correctness);
    RUN_TEST(test_linear_relationship);
    RUN_TEST(test_cost_function_perfect_weights);
    RUN_TEST(test_cost_function_wrong_weights);
    RUN_TEST(test_gradient_descent_convergence);
    
    return UNITY_END();
}
