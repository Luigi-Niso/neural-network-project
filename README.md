# Neural Network Project

Progetto di neural networks in C con architettura modulare e funzioni di costo multiple.

## 📁 Struttura del Progetto

```
neural-network-project/
├── lib/                          # Libreria condivisa
│   ├── neural.h                  # Header con tutte le dichiarazioni
│   └── neural.c                  # Implementazioni delle funzioni
├── 1-gates-learning/             # Apprendimento porte logiche semplici
│   ├── Makefile
│   ├── src/
│   │   └── main.c               # OR/AND gate con 2 input
│   └── test/
│       └── src/
│           └── test.c           # Test unitari
├── 2-complex-gates/              # Porte logiche complesse
│   ├── Makefile
│   ├── src/
│   │   └── main.c               # XOR gate con multi-layer perceptron
│   └── test/
│       └── src/
│           └── test.c
└── unity/                        # Framework di testing
    └── src/
```

## 🚀 Funzionalità

### **Libreria Condivisa (`lib/`)**

#### **Funzioni di Costo:**
- `cost_simple(w, b)` - Per modelli con 1 input
- `cost_double(w1, w2, b)` - Per modelli con 2 input  
- `cost_xor(model)` - Per reti multi-layer (XOR)

#### **Funzioni di Attivazione:**
- `sigmoidf(x)` - Funzione sigmoid: 1/(1+e^(-x))

#### **Utilità:**
- `rand_float()` - Generatore numeri casuali [0,1]
- `print_results_simple(w, b)` - Stampa risultati modello semplice
- `print_results_double(w1, w2, b)` - Stampa risultati modello a 2 input

#### **Reti Complesse:**
- `Xor` struct - Multi-layer perceptron per XOR
- `forward_xor(model, x1, x2)` - Forward pass della rete XOR
- `rand_xor()` - Inizializzazione casuale dei pesi

## 📊 Progetti

### **1. Gates Learning (`1-gates-learning/`)**
Implementa l'apprendimento di porte logiche lineari (OR, AND) usando:
- **Algoritmo:** Gradient Descent con finite differences
- **Architettura:** Singolo perceptron (2 input → 1 output)
- **Dataset:** OR gate (modificabile)

```bash
cd 1-gates-learning
make run    # Esegue il training
make test   # Esegue i test unitari
```

### **2. Complex Gates (`2-complex-gates/`)**
Implementa XOR gate usando Multi-Layer Perceptron:
- **Architettura:** OR + NAND → AND (3 layer)
- **Algoritmo:** Gradient descent su tutti i 9 pesi
- **Dataset:** XOR gate

```bash
cd 2-complex-gates
make run    # Addestra XOR gate
```

## 🔬 Testing

Ogni progetto include test unitari con Unity framework:

```bash
# Test per 1-gates-learning
cd 1-gates-learning && make test

# Include test per:
# - Funzione sigmoid
# - Generazione numeri casuali  
# - Funzioni di costo
# - Convergenza gradient descent
# - Logica delle porte
# - Inizializzazione modelli
```

## 📈 Risultati Attesi

### **OR Gate (1-gates-learning):**
```
Input: (0,0) | Expected: 0 | Actual: 0.022315
Input: (0,1) | Expected: 1 | Actual: 0.985732
Input: (1,0) | Expected: 1 | Actual: 0.986234
Input: (1,1) | Expected: 1 | Actual: 0.999995
Final cost: 0.000223
```

### **XOR Gate (2-complex-gates):**
```
Input: (0,0) | Expected: 0 | Actual: 0.018852 | Error: 0.018852
Input: (0,1) | Expected: 1 | Actual: 0.983725 | Error: 0.016275
Input: (1,0) | Expected: 1 | Actual: 0.983720 | Error: 0.016280
Input: (1,1) | Expected: 0 | Actual: 0.016814 | Error: 0.016814
Final cost: 0.000292
```

## ⚙️ Compilazione

Ogni Makefile utilizza:
- **Compiler:** gcc con flags `-Wall -Wextra -std=c99 -lm`
- **Include paths:** Automatici per `lib/` e `unity/`
- **Linking:** Matematica (`-lm`) per `expf()`, `fabsf()`

## 🧮 Algoritmi Implementati

1. **Finite Difference Gradient:** `(f(x+ε) - f(x))/ε`
2. **Gradient Descent:** `w = w - α∇f(w)`
3. **Mean Squared Error:** `MSE = Σ(y_pred - y_true)²/n`
4. **Forward Propagation:** Multi-layer con sigmoid

## 🔄 Estensibilità

La struttura modulare permette di aggiungere facilmente:
- Nuove funzioni di attivazione
- Algoritmi di ottimizzazione diversi  
- Architetture di rete più complesse
- Nuovi dataset di training

## 📝 Note Tecniche

- **Linearmente Separabile:** OR, AND funzionano con singolo perceptron
- **Non-linearly Separabile:** XOR richiede multi-layer perceptron
- **Numerical Stability:** Gestione overflow/underflow in sigmoid
- **Cross-platform:** Compatibile macOS/Linux/Windows
