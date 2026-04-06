// biblis/ativas.h
#pragma once
#include <math.h>

inline int degrau(float x) {
    return x > 0 ? 1 : 0;
}

inline float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}
inline float derivadaSigmoid(float x) {
    const float s = sigmoid(x);
    return s * (1.0f - s);
}

inline float hardSigmoid(float x) {
    return fmax(0, fmin(1, 0.2 * x + 0.5));
}
inline float derivadaHardSigmoid(float y) {
    return (y > -2.5 && y < 2.5) ? 0.2 : 0;
}

inline float tanhF(float x) {
    return tanhf(x);
}
inline float derivadaTanh(float x) {
    const float t = tanhf(x);
    return 1.0f - t * t;
}

inline float ReLU(float x) {
    return fmax(0, x);
}

inline float derivadaReLU(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

inline float leakyReLU(float x) {
    return x > 0 ? x : 0.01 * x;
}
inline float derivadaLeakyReLU(float y) {
    return y > 0 ? 1 : 0.01;
}

inline float softsign(float x) {
    return x / (1 + abs(x));
}
inline float derivadaSoftsign(float x) {
    const float denom = 1 + fabsf(x);
    return 1.0f / (denom * denom);
}

inline float softplus(float x) {
    return log(1 + exp(x));
}

inline float swish(float x) {
    return x * sigmoid(x);
}
inline float derivadaSwish(float y){
    const float sigmoidX = sigmoid(y);
    return sigmoidX + y * sigmoidX * (1 - sigmoidX);
}

inline float hardSwish(float x) {
    return x * fmax(0, fmin(1, (x + 3) / 6));
}
inline float derivadaHardSwish(float y) {
    return y <= -3 ? 0 : y >= 3 ? 1 : (y + 3) / 6 + y / 6;
}

inline float GELU(float x) {
    return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}
inline float derivadaGELU(float x) {
    const float c = 0.7978845608f; // sqrt(2/pi)
    const float u = c * (x + 0.044715f * x * x * x);
    const float th = tanhf(u);
    return 0.5f*(1.0f+th) + 0.5f*x*(1.0f-th*th)*c*(1.0f+3.0f*0.044715f*x*x);
}

inline float ELU(float x) {
    return x >= 0 ? x : 1.0 * (exp(x) - 1);
}
inline float derivadaELU(float x) {
    return x >= 0 ? 1.0f : expf(x);
}

inline float SELU(float x) {
    return 1.0507 * (x >= 0 ? x : 1.67326 * (exp(x) - 1));
}
inline float derivadaSELU(float x) {
    return x >= 0 ? 1.0507f : 1.0507f * 1.67326f * expf(x);
}

inline float SiLU(float x) {
    return x * sigmoid(x);
}

inline float mish(float x) {
    return x * tanh(log(1 + exp(x)));
}
inline float derivadaMish(float y) {
    const float omega = 4 * (y + 1) + 4 * exp(2 * y) + exp(3 * y) + exp(y) * (4 * y + 6);
    const float delta = 2 * exp(y) + exp(2 * y) + 2;
    return exp(y) * omega / (delta * delta);
}

inline float bentIdentity(float x){
    return (sqrt(x * x + 1) - 1) / 2 + x;
}
inline float derivadaBentIdentity(float x) {
    return x / (2.0f * sqrtf(x * x + 1.0f)) + 1.0f;
}

inline float gaussian(float x) {
    return exp(-x * x);
}
inline float derivadaGaussian(float y) {
    return -2 * y * exp(-y * y);
}