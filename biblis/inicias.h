// biblis/inicias.h
#pragma once
#include <math.h>
#include <stdlib.h>

static const float RAND_MAXf = (float)RAND_MAX;

static inline float _uniforme() {
    return (float)rand() / RAND_MAXf;
}

// uniforme entre -limite e +limite
static inline void iniUniforme(float* pesos, int n, float limite) {
    for(int i = 0; i < n; i++) {
        pesos[i] = _uniforme() * 2.0f * limite - limite;
    }
}

// xavier uniforme: sqrt(6 / (entradas + saidas))
static inline void iniXavier(float* pesos, int n, int entradas, int saidas) {
    float limite = sqrtf(6.0f / (entradas + saidas));
    iniUniforme(pesos, n, limite);
}

// he uniforme: sqrt(6 / entradas)
static inline void iniHe(float* pesos, int n, int entradas) {
    float limite = sqrtf(6.0f / entradas);
    iniUniforme(pesos, n, limite);
}

// zeros
static inline void iniZeros(float* pesos, int n) {
    memset(pesos, 0, n * sizeof(float));
}

// constante
static inline void iniConstante(float* pesos, int n, float valor) {
    for(int i = 0; i < n; i++) {
        pesos[i] = valor;
    }
}