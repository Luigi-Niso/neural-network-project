#include "unity.h"
#include "neural.h"
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

// Test della funzione cost_double con valori perfetti
void test_cost_double_perfect_weights(void) {
    // Per OR gate: w1=20, w2=20, b=-10 dovrebbe dare cost molto basso
    float perfect_cost = cost_double(20.0f, 20.0f, -10.0f);
    TEST_ASSERT_TRUE(perfect_cost < 0.1f);
}

// Test della funzione cost_double con valori casuali
void test_cost_double_random_weights(void) {
    float random_cost = cost_double(1.0f, 1.0f, 1.0f);
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
    
    float initial_cost = cost_double(w1, w2, b);
    
    // Esegui alcune iterazioni di gradient descent
    for(int i = 0; i < 1000; i++) {
        float c = cost_double(w1, w2, b);
        float dw1 = (cost_double(w1 + eps, w2, b) - c)/eps;
        float dw2 = (cost_double(w1, w2 + eps, b) - c)/eps;
        float db = (cost_double(w1, w2, b + eps) - c)/eps;
        
        w1 -= rate * dw1;
        w2 -= rate * dw2;
        b -= rate * db;
    }
    
    float final_cost = cost_double(w1, w2, b);
    
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

// Test XOR model initialization
void test_xor_model_initialization(void) {
    Xor model = rand_xor();
    
    // Tutti i pesi dovrebbero essere tra 0 e 1
    TEST_ASSERT_TRUE(model.or_w1 >= 0.0f && model.or_w1 <= 1.0f);
    TEST_ASSERT_TRUE(model.or_w2 >= 0.0f && model.or_w2 <= 1.0f);
    TEST_ASSERT_TRUE(model.and_w1 >= 0.0f && model.and_w1 <= 1.0f);
}

// Test cost functions exist
void test_cost_functions_exist(void) {
    float simple_cost = cost_simple(1.0f, 0.0f);
    float double_cost = cost_double(1.0f, 1.0f, 0.0f);
    
    TEST_ASSERT_TRUE(simple_cost >= 0.0f);
    TEST_ASSERT_TRUE(double_cost >= 0.0f);
}

int main(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_sigmoidf_returns_correct_values);
    RUN_TEST(test_sigmoidf_boundaries);
    RUN_TEST(test_rand_float_returns_valid_range);
    RUN_TEST(test_cost_double_perfect_weights);
    RUN_TEST(test_cost_double_random_weights);
    RUN_TEST(test_gradient_descent_convergence);
    RUN_TEST(test_or_gate_logic);
    RUN_TEST(test_xor_model_initialization);
    RUN_TEST(test_cost_functions_exist);
    
    return UNITY_END();
}
