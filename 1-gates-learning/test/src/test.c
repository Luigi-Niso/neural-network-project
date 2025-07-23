#include "unity.h"
#include "../../src/neural.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

void setUp(void) {
    // Setup per ogni test
    srand(42); // Seed fisso per test riproducibili
}

void tearDown(void) {
    // Cleanup dopo ogni test
}

// Test della funzione sigmoidf
void test_sigmoidf_returns_correct_values(void) {
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.5f, sigmoidf(0.0f));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.731f, sigmoidf(1.0f));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.269f, sigmoidf(-1.0f));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.999f, sigmoidf(10.0f));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.001f, sigmoidf(-10.0f));
}

// Test della funzione sigmoidf boundaries
void test_sigmoidf_boundaries(void) {
    float result_positive = sigmoidf(100.0f);
    float result_negative = sigmoidf(-100.0f);
    
    TEST_ASSERT_TRUE(result_positive > 0.99f);
    TEST_ASSERT_TRUE(result_negative < 0.01f);
    TEST_ASSERT_TRUE(result_positive <= 1.0f);
    TEST_ASSERT_TRUE(result_negative >= 0.0f);
}

// Test della funzione rand_float
void test_rand_float_returns_valid_range(void) {
    for(int i = 0; i < 100; i++) {
        float value = rand_float();
        TEST_ASSERT_TRUE(value >= 0.0f);
        TEST_ASSERT_TRUE(value <= 1.0f);
    }
}

// Test della funzione cost con valori perfetti
void test_cost_perfect_weights(void) {
    // Per OR gate: w1=20, w2=20, b=-10 dovrebbe dare cost molto basso
    float perfect_cost = cost(20.0f, 20.0f, -10.0f);
    TEST_ASSERT_TRUE(perfect_cost < 0.1f);
}

// Test della funzione cost con valori casuali
void test_cost_random_weights(void) {
    float random_cost = cost(1.0f, 1.0f, 1.0f);
    TEST_ASSERT_TRUE(random_cost >= 0.0f);
    TEST_ASSERT_TRUE(random_cost <= 4.0f); // Massimo teorico per 4 campioni
}

// Test gradient descent convergence
void test_gradient_descent_convergence(void) {
    float w1 = 1.0f;
    float w2 = 1.0f;
    float b = 1.0f;
    
    float eps = 1e-3;
    float rate = 1e-1;
    
    float initial_cost = cost(w1, w2, b);
    
    // Esegui alcune iterazioni di gradient descent
    for(int i = 0; i < 1000; i++) {
        float dw1 = (cost(w1 + eps, w2, b) - cost(w1, w2, b))/eps;
        float dw2 = (cost(w1, w2 + eps, b) - cost(w1, w2, b))/eps;
        float db = (cost(w1, w2, b + eps) - cost(w1, w2, b))/eps;
        
        w1 -= rate * dw1;
        w2 -= rate * dw2;
        b -= rate * db;
    }
    
    float final_cost = cost(w1, w2, b);
    
    // Il costo finale dovrebbe essere minore di quello iniziale
    TEST_ASSERT_TRUE(final_cost < initial_cost);
}

// Test OR gate logic
void test_or_gate_logic(void) {
    // Testa che i dati di training rappresentino correttamente OR gate
    TEST_ASSERT_EQUAL_FLOAT(0.0f, train[0][2]); // 0 OR 0 = 0
    TEST_ASSERT_EQUAL_FLOAT(1.0f, train[1][2]); // 0 OR 1 = 1
    TEST_ASSERT_EQUAL_FLOAT(1.0f, train[2][2]); // 1 OR 0 = 1
    TEST_ASSERT_EQUAL_FLOAT(1.0f, train[3][2]); // 1 OR 1 = 1
}

// Test trained model predictions
void test_trained_model_predictions(void) {
    // Pesi pre-addestrati per OR gate
    float w1 = 8.5f;
    float w2 = 8.5f;
    float b = -4.0f;
    
    // Test predizioni
    float pred1 = sigmoidf(0.0f * w1 + 0.0f * w2 + b); // 0,0 -> dovrebbe essere ~0
    float pred2 = sigmoidf(0.0f * w1 + 1.0f * w2 + b); // 0,1 -> dovrebbe essere ~1
    float pred3 = sigmoidf(1.0f * w1 + 0.0f * w2 + b); // 1,0 -> dovrebbe essere ~1
    float pred4 = sigmoidf(1.0f * w1 + 1.0f * w2 + b); // 1,1 -> dovrebbe essere ~1
    
    TEST_ASSERT_TRUE(pred1 < 0.1f);  // Vicino a 0
    TEST_ASSERT_TRUE(pred2 > 0.9f);  // Vicino a 1
    TEST_ASSERT_TRUE(pred3 > 0.9f);  // Vicino a 1
    TEST_ASSERT_TRUE(pred4 > 0.9f);  // Vicino a 1
}

// Test numerical stability
void test_numerical_stability(void) {
    // Test con valori estremi
    float very_large = sigmoidf(1000.0f);
    float very_small = sigmoidf(-1000.0f);
    
    // Non dovrebbero essere NaN o infinito
    TEST_ASSERT_FALSE(isnan(very_large));
    TEST_ASSERT_FALSE(isinf(very_large));
    TEST_ASSERT_FALSE(isnan(very_small));
    TEST_ASSERT_FALSE(isinf(very_small));
}

int main(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_sigmoidf_returns_correct_values);
    RUN_TEST(test_sigmoidf_boundaries);
    RUN_TEST(test_rand_float_returns_valid_range);
    RUN_TEST(test_cost_perfect_weights);
    RUN_TEST(test_cost_random_weights);
    RUN_TEST(test_gradient_descent_convergence);
    RUN_TEST(test_or_gate_logic);
    RUN_TEST(test_trained_model_predictions);
    RUN_TEST(test_numerical_stability);
    
    return UNITY_END();
}
