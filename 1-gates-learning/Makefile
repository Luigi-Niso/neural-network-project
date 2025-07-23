# Compilatore e flag
CC = gcc
CFLAGS = -Wall -Wextra -std=c99

# Directory
SRC_DIR = src
BIN_DIR = bin
TEST_SRC_DIR = test/src
TEST_BIN_DIR = test/bin
UNITY_DIR = ../unity/src

# File sorgente e eseguibili
MAIN_SRC = $(SRC_DIR)/main.c $(SRC_DIR)/neural.c
MAIN_TARGET = $(BIN_DIR)/main
TEST_SRC = $(TEST_SRC_DIR)/test.c $(SRC_DIR)/neural.c
TEST_TARGET = $(TEST_BIN_DIR)/test
UNITY_SRC = $(UNITY_DIR)/unity.c

# Target di default
all: $(MAIN_TARGET)

# Compila il programma principale
$(MAIN_TARGET): $(MAIN_SRC) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(MAIN_TARGET) $(MAIN_SRC)

# Compila i test (con Unity)
$(TEST_TARGET): $(TEST_SRC) $(UNITY_SRC) | $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -I$(UNITY_DIR) -I$(SRC_DIR) -o $(TEST_TARGET) $(TEST_SRC) $(UNITY_SRC)

# Crea le directory se non esistono
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TEST_BIN_DIR):
	mkdir -p $(TEST_BIN_DIR)

# Esegue il programma principale
run: $(MAIN_TARGET)
	clear
	./$(MAIN_TARGET)

# Esegue i test
test: $(TEST_TARGET)
	clear
	./$(TEST_TARGET)

# Pulisce i file generati
clean:
	rm -rf $(BIN_DIR) $(TEST_BIN_DIR)

# Ricompila tutto da zero
rebuild: clean all

.PHONY: all run test clean rebuild