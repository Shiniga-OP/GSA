// biblioteca de utilitários %100 portugues
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef RAND_MAX_F
#define RAND_MAX_F 2147483647.0f
#endif

int degrau(float x) {
  return x >= 0 ? 1 : 0;
}

float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}
float derivadaSigmoid(float y) {
  return y * (1 - y);
}

float tanhA(float x) {
  return tanhf(x);
}
float derivadaTanh(float y) {
  return 1.0f - y * y;
}

float ReLU(float x) {
  return fmaxf(0.0f, x);
}
float derivadaReLU(float x) {
  return x > 0 ? 1.0f : 0.0f;
}

float leakyReLU(float x, float alpha) {
  return x > 0 ? x : alpha * x;
}
float derivadaLeakyReLU(float x, float alpha) {
  return x > 0 ? 1.0f : alpha;
}

float softsign(float x) {
  return x / (1.0f + fabsf(x));
}
float derivadaSoftsign(float x) {
  float denom = 1.0f + fabsf(x);
  return 1.0f / (denom * denom);
}

float softplus(float x) {
  if(x > 0) return x + log1pf(expf(-x));
  return log1pf(expf(x));
}

float swish(float x) {
  return x * sigmoid(x);
}
float derivadaSwish(float x) {
  float sig_x = sigmoid(x);
  return sig_x + x * sig_x * (1.0f - sig_x);
}

float GELU(float x) {
  return 0.5f * x * 
  (1.0f + tanhf(sqrtf(2.0f / (float)M_PI) * 
  (x + 0.044715f * x * x * x)));
}

float ELU(float x, float alpha) {
  return x >= 0 ? x : alpha * (expf(x) - 1);
}
float derivadaELU(float x, float alpha) {
  return x >= 0 ? 1.0f : ELU(x, alpha) + alpha;
}

float SELU(float x, float alpha, float escala) {
  return escala * (x >= 0 ? x : alpha * (expf(x) - 1));
}
float derivadaSELU(float x, float alpha, float escala) {
  return escala * (x >= 0 ? 1.0f : alpha * expf(x));
}

float mish(float x) {
  return x * tanhf(log1pf(expf(x)));
}
float derivadaMish(float x) {
  float sp = softplus(x);
  float t = tanhf(sp);
  float sig = sigmoid(x);
  return t + x * sig * (1 - t * t);
}
// funcoes de saida:
void softmax(float *entrada, float *saida, int tamanho, float temperatura) {
  if(temperatura <= 0.0f || !isfinite(temperatura)) temperatura = 1e-8f;
  float max = -FLT_MAX;
  for(int i = 0; i < tamanho; i++) {
    float v = isfinite(entrada[i]) ? entrada[i] : 0.0f;
    if(v > max) max = v;
  }
  float *exps = (float *)malloc(sizeof(float) * tamanho);
  for(int i = 0; i < tamanho; i++) {
    float t = (entrada[i] - max) / temperatura;
    exps[i] = isfinite(t) ? expf(t) : 0.0f;
  }
  float soma = 0.0f;
  for(int i = 0; i < tamanho; i++) soma += exps[i];
  if(soma == 0.0f) soma = 1e-8f;
  for(int i = 0; i < tamanho; i++) {
    saida[i] = exps[i] / soma;
  }
  free(exps);
}

void derivada_softmax(const float *saida_softmax, const float *grad_saida, float *grad_entrada, int tamanho) {
  for(int i = 0; i < tamanho; i++) {
    float soma = 0.0f;
    for(int j = 0; j < tamanho; j++) {
      float delta = (i == j) ? 1.0f : 0.0f;
      soma += grad_saida[j] * saida_softmax[i] * (delta - saida_softmax[j]);
    }
    grad_entrada[i] = soma;
  }
}

void softmax_lote(float **matriz, float **saida, int linhas, int colunas, float temperatura) {
  for(int i = 0; i < linhas; i++) {
    softmax(matriz[i], saida[i], colunas, temperatura);
  }
}

int argmax(const float *v, int tamanho) {
  int indice = 0;
  float max = v[0];
  for(int i = 1; i < tamanho; i++) {
    if(v[i] > max) {
      max = v[i];
      indice = i;
    }
  }
  return indice;
}

void add_ruido(float *v, int tamanho, float intensidade) {
  for(int i = 0; i < tamanho; i++) {
    float ruido = (rand() / RAND_MAX_F) * 2.0f - 1.0f;
    v[i] += ruido * intensidade;
  }
}
// funcoes de erro:
double erroAbsolutoMedio(double *saida, double *esperado, int tamanho) {
  double soma = .0;
  for(int i = 0; i < tamanho; i++)
  soma += fabs(saida[i] - esperado[i]);
  return soma / tamanho;
}

double erroQuadradoEsperado(double *saida, double *esperado, int tamanho) {
  double soma = .0;
  for(int i = 0; i < tamanho; i++) {
    double diff = saida[i] - esperado[i];
    soma += 0.5 * diff * diff;
  }
  return soma;
}
double* derivadaErro(double *saida, double *esperado, int tamanho) {
  double *derivadas = (double*)malloc(tamanho * sizeof(double));
  for(int i = 0; i < tamanho; i++)
  derivadas[i] = saida[i] - esperado[i];
  return derivadas; // libere memória depois
}

double entropiaCruzada(double *y, double *yChapeu, int tamanho) {
  const double epsilon = 1e-12;
  double soma = 0.0;
  for(int i = 0; i < tamanho; i++) {
    double p = yChapeu[i];
    if(p < epsilon) p = epsilon;
    else if(p > 1.0 - epsilon) p = 1.0 - epsilon;
    soma += y[i] * log(p);
  }
  return -soma;
}
double* derivadaEntropiaCruzada(double *y, double *yChapeu, int tamanho) {
  double *derivadas = (double*)malloc(tamanho * sizeof(double));
  for(int i = 0; i < tamanho; i++) {
    if(!isfinite(y[i]) || !isfinite(yChapeu[i])) {
      fprintf(stderr, "[derivadaEntropiaCruzada]: valor inválido em i=%d: y=%f, y^=%f\n", i, y[i], yChapeu[i]);
      derivadas[i] = 0;
    } else {
      derivadas[i] = yChapeu[i] - y[i];
    }
  }
  return derivadas; // livre memória depois
}

double huberPerda(double *saida, double *esperado, int tamanho, double delta) {
  double soma = .0;
  for(int i = 0; i < tamanho; i++) {
    double diff = saida[i] - esperado[i];
    double abs_diff = fabs(diff);
    soma += (abs_diff <= delta) 
    ? 0.5 * diff * diff 
    : delta * (abs_diff - 0.5 * delta);
  }
  return soma / tamanho;
}
double* derivadaHuber(double *saida, double *esperado, int tamanho, double delta) {
  double *derivadas = (double*)malloc(tamanho * sizeof(double));
  for(int i = 0; i < tamanho; i++) {
    double diff = saida[i] - esperado[i];
    double abs_diff = fabs(diff);
    derivadas[i] = (abs_diff <= delta) 
    ? diff 
    : delta * (diff > 0 ? 1 : -1);
  }
  return derivadas; // libere memória depois
}

double perdaTripleto(double *ancora, double *positiva, double *negativa, int tamanho, double margem) {
  double distPos = 0, distNeg = 0;
  for(int i = 0; i < tamanho; i++) {
    distPos += pow(ancora[i] - positiva[i], 2);
    distNeg += pow(ancora[i] - negativa[i], 2);
  }
  double perda = distPos - distNeg + margem;
  return (perda > 0) ? perda : 0;
}

double contrastivaPerda(double *saida1, double *saida2, int rotulo, int tamanho, double margem) {
  double distancia = 0;
  for(int i = 0; i < tamanho; i++) distancia += pow(saida1[i] - saida2[i], 2);
  
  if(rotulo == 1) return distancia;
  double dist_euclid = sqrt(distancia);
  return (margem > dist_euclid) ? margem - dist_euclid : 0;
}

int main() {

    return 0;
}
