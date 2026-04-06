// biblis/ativas.h
#pragma once
#include <math.h>

int degrau(float x) {
    return x > 0 ? 1: 0;
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}
float derivadaSigmoid(float y) {
    return y * (1 - y);
}

float hardSigmoid(float x) {
    return fmax(0, fmin(1,0.2 * x + 0.5));
}
float derivadaHardSigmoid(float y) {
    return (y > -2.5 && y < 2.5) ? 0.2 : 0;
}

float tanhF(float x) {
    return (float)tanh(x);
}
float derivadaTanh(float y) {
    return 1 - y * y;
}

float ReLU(float x) {
    return fmax(0,x);
}

float leakyReLU(float x) {
    return x > 0 ? x : 0.01 * x;
}
float derivadaLeakyReLU(float y) {
    return y > 0 ? 1 : 0.01;
}

float softsign(float x) {
    return x / (1 + abs(x));
}
float derivadaSoftsign(float y) {
    const float denom = 1 + abs(y);
    return 1 / (denom * denom);
}

float softplus(float x) {
    return log(1 + exp(x));
}

float swish(float x) {
    return x * sigmoid(x);
}
float derivadaSwish(float y){
    const float sigmoidX = sigmoid(y);
    return sigmoidX + y * sigmoidX * (1 - sigmoidX);
}

float hardSwish(float x) {
    return x * fmax(0, fmin(1, (x + 3) / 6));
}
float derivadaHardSwish(float y) {
    return y <= -3 ? 0 : y >= 3 ? 1 : (y + 3) / 6 + y / 6;
}

float GELU(float x) {
    return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}
float derivadaGELU(float x){
   const float cdf = 0.5 * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * x * x * x)));
   return cdf + x * (1 - cdf * cdf) * (0.5 * sqrt(2 / M_PI) * (1 + 3 * 0.044715 * x * x));
}

float ELU(float x, float alfa = 1.0) {
    return x >= 0 ? x : alfa * (exp(x) - 1);
}
float derivadaELU(float y, float alfa = 1.0) {
    return y >= 0 ? 1 : ELU(y, alfa) + alfa;
}

float SELU(float x, float alfa = 1.67326, float escala = 1.0507) {
    return escala * (x >= 0 ? x : alfa * (exp(x) - 1));
}
float derivadaSELU(float y, float alfa = 1.67326, float escala = 1.0507) {
    return escala * (y >= 0 ? 1 : alfa * exp(y));
}

float SiLU(float x) {
    return x * sigmoid(x);
}

float mish(float x) {
    return x * tanh(log(1 + exp(x)));
}
float derivadaMish(float y) {
    const float omega = 4 * (y + 1) + 4 * exp(2 * y) + exp(3 * y) + exp(y) * (4 * y + 6);
    const float delta = 2 * exp(y) + exp(2 * y) + 2;
    return exp(y) * omega / (delta * delta);
}

float bentIdentity(float x){
    return (sqrt(x * x + 1) - 1) / 2 + x;
}
float derivadaBentIdentity(float y) {
    return y / (2 * sqrt(y * y + 1)) + 1;
}

float gaussian(float x) {
    return exp(-x * x);
}
float derivadaGaussian(float y) {
    return -2 * y * exp(-y * y);
}