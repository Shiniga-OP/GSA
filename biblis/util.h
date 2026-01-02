// biblis/util.h
#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <math.h>

using namespace std;

// funções de pesos
vector<vector<float>> iniPesosXavier(int l, int c) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    float limite = sqrt(6.0f / (l + c));
    vector<vector<float>> pesos(l, vector<float>(c));
    
    for(int i = 0; i < l; ++i) {
        for(int j = 0; j < c; ++j) pesos[i][j] = dis(gen) * limite;
    }
    return pesos;
}

vector<vector<float>> iniPesosHe(int l, int c) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    float limite = sqrt(2.0f / l);
    vector<vector<float>> pesos(l, vector<float>(c));
    
    for(int i = 0; i < l; ++i) {
        for(int j = 0; j < c; ++j) pesos[i][j] = dis(gen) * limite;
    }
    return pesos;
}

// att de pesos
vector<vector<float>> attPesos(const vector<vector<float>>& pesos, const vector<vector<float>>& grad, float taxa, float lambda = 1e-3f) {
    vector<vector<float>> nova(pesos.size(), vector<float>(pesos[0].size()));
    
    for(size_t i = 0; i < pesos.size(); ++i) {
        for(size_t j = 0; j < pesos[i].size(); ++j) {
            nova[i][j] = pesos[i][j] - taxa * grad[i][j] - lambda * pesos[i][j];
        }
    }
    return nova;
}

vector<vector<float>> attPesosMomentum(const vector<vector<float>>& pesos, const vector<vector<float>>& grad, float taxa, float momento, vector<vector<float>>& velocidade) {
    vector<vector<float>> nova(pesos.size(), vector<float>(pesos[0].size()));
    
    for(size_t i = 0; i < pesos.size(); ++i) {
        for(size_t j = 0; j < pesos[i].size(); ++j) {
            velocidade[i][j] = momento * velocidade[i][j] + grad[i][j];
            nova[i][j] = pesos[i][j] - taxa * velocidade[i][j];
        }
    }
    return nova;
}

vector<vector<float>> attPesosAdam(const vector<vector<float>>& pesos, const vector<vector<float>>& grad, vector<vector<float>>& m, vector<vector<float>>& v, float taxa, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, int iteracao = 1, float lambda = 0.001f) {
    vector<vector<float>> nova(pesos.size(), vector<float>(pesos[0].size()));
    
    float fator1 = 1.0f - pow(beta1, iteracao);
    float fator2 = 1.0f - pow(beta2, iteracao);
    float umMenosBeta1 = 1.0f - beta1;
    float umMenosBeta2 = 1.0f - beta2;
    
    for(size_t i = 0; i < pesos.size(); ++i) {
        for(size_t j = 0; j < pesos[i].size(); ++j) {
            float g = grad[i][j] + lambda * pesos[i][j];
            m[i][j] = beta1 * m[i][j] + umMenosBeta1 * g;
            v[i][j] = beta2 * v[i][j] + umMenosBeta2 * g * g;
            float mChapeu = m[i][j] / fator1;
            float vChapeu = v[i][j] / fator2;
            nova[i][j] = pesos[i][j] - taxa * mChapeu / (sqrt(vChapeu) + eps);
        }
    }
    return nova;
}

vector<float> attPesosAdam1D(vector<float>& p, const vector<float>& grad, vector<float>& m, vector<float>& v, float taxa, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, int t = 1, float lambda = 0.001f) {
    float umMenosBeta1 = 1.0f - beta1;
    float umMenosBeta2 = 1.0f - beta2;
    float fator1 = 1.0f - pow(beta1, t);
    float fator2 = 1.0f - pow(beta2, t);
    
    for(size_t i = 0; i < p.size(); ++i) {
        float g = grad[i] + lambda * p[i];
        m[i] = beta1 * m[i] + umMenosBeta1 * g;
        v[i] = beta2 * v[i] + umMenosBeta2 * g * g;
        float mChapeu = m[i] / fator1;
        float vChapeu = v[i] / fator2;
        p[i] -= taxa * mChapeu / (sqrt(vChapeu) + eps);
    }
    return p;
}

// funções de regularização:
vector<vector<float>> regularL1(const vector<vector<float>>& pesos, float lambda) {
    vector<vector<float>> res;
    for(const auto& linha : pesos) {
        vector<float> novaLinha;
        for(float p : linha) novaLinha.push_back(lambda * (p > 0 ? 1.0f : (p < 0 ? -1.0f : 0.0f)));
        res.push_back(novaLinha);
    }
    return res;
}

vector<vector<float>> regularL2(const vector<vector<float>>& pesos, float lambda) {
    vector<vector<float>> res;
    for(const auto& linha : pesos) {
        vector<float> novaLinha;
        for(float p : linha) novaLinha.push_back(lambda * p);
        res.push_back(novaLinha);
    }
    return res;
}

vector<vector<float>> dropout(vector<vector<float>> tensor, float taxa) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for(auto& linha : tensor) {
        for(float& valor : linha) {
            if(dis(gen) < taxa) valor = 0.0f;
            else valor /= (1.0f - taxa);
        }
    }
    return tensor;
}

vector<float> normEntrada(const vector<float>& vetor) {
    float max = *max_element(vetor.begin(), vetor.end());
    float min = *min_element(vetor.begin(), vetor.end());
    float amplitude = (max - min) > 1e-8f ? (max - min) : 1e-8f;
    
    vector<float> res;
    for(float x : vetor) res.push_back((x - min) / amplitude);
    return res;
}

vector<float> normZPonto(const vector<float>& v) {
    float soma = 0.0f;
    for(float x : v) soma += x;
    float media = soma / v.size();
    
    float variancia = 0.0f;
    for (float x : v) variancia += pow(x - media, 2);
    variancia /= v.size();
    
    float desvio = sqrt(variancia + 1e-8f);
    
    vector<float> res;
    for(float x : v) res.push_back((x - media) / desvio);
    return res;
}

// funções de metricas
float acuracia(const vector<vector<float>>& saida, const vector<vector<float>>& esperado) {
    int corretos = 0;
    for(size_t i = 0; i < saida.size(); i++) {
        int pred = distance(saida[i].begin(), max_element(saida[i].begin(), saida[i].end()));
        int real = distance(esperado[i].begin(), max_element(esperado[i].begin(), esperado[i].end()));
        if(pred == real) corretos++;
    }
    return static_cast<float>(corretos) / saida.size();
}

float precisao(const vector<vector<int>>& confusao) {
    int tp = confusao[0][0];
    int fp = 0;
    for (size_t i = 1; i < confusao[0].size(); i++) {
        fp += confusao[0][i];
    }
    return static_cast<float>(tp) / (tp + fp + 1e-8f);
}

float recall(const vector<vector<int>>& confusao) {
    int tp = confusao[0][0];
    int fn = 0;
    for(size_t i = 1; i < confusao.size(); i++) fn += confusao[i][0];
    return static_cast<float>(tp) / (tp + fn + 1e-8f);
}

float f1Ponto(const vector<vector<int>>& confusao) {
    float p = precisao(confusao);
    float r = recall(confusao);
    return 2.0f * (p * r) / (p + r + 1e-8f);
}

float mse(const vector<float>& saida, const vector<float>& esperado) {
    float soma = 0.0f;
    for(size_t i = 0; i < saida.size(); i++) soma += pow(saida[i] - esperado[i], 2);
    return soma / saida.size();
}

float klDivergencia(const vector<float>& p, const vector<float>& q) {
    float soma = 0.0f;
    for(size_t i = 0; i < p.size(); i++) soma += p[i] * log((p[i] + 1e-12f) / (q[i] + 1e-12f));
    return soma;
}

float rocAuc(const vector<float>& pontos, const vector<int>& rotulos) {
    // cria pares [pontuação, rótulo] e ordenar por pontuação (decrescente)
    vector<pair<float, int>> pares;
    for(size_t i = 0; i < pontos.size(); i++) pares.push_back({pontos[i], rotulos[i]});
    
    sort(pares.begin(), pares.end(), 
        [](const auto& a, const auto& b) {
            return a.first > b.first;
    });
    
    float auc = 0.0f;
    int fp = 0, tp = 0, fpPrev = 0, tpPrev = 0;
    
    for(const auto& par : pares) {
        if(par.second == 1) tp++;
        else fp++;
        
        auc += (fp - fpPrev) * (tp + tpPrev) / 2.0f;
        fpPrev = fp;
        tpPrev = tp;
    }
    return auc / (tp * fp);
}

// funções de erro:
float erroAbsolutoMedio(const vector<float>& saida, const vector<float>& esperado) {
    float soma = 0.0f;
    for(size_t i = 0; i < saida.size(); i++) soma += abs(saida[i] - esperado[i]);
    return soma / saida.size();
}

float erroQuadradoEsperado(const vector<float>& saida, const vector<float>& esperado) {
    float soma = 0.0f;
    for(size_t i = 0; i < saida.size(); i++) {
        float diff = saida[i] - esperado[i];
        soma += 0.5f * diff * diff;
    }
    return soma;
}

vector<float> derivadaErro(const vector<float>& saida, const vector<float>& esperado) {
    vector<float> deriv(saida.size());
    for(size_t i = 0; i < saida.size(); i++) deriv[i] = saida[i] - esperado[i];
    return deriv;
}

float entropiaCruzada(const vector<float>& y, const vector<float>& yChapeu) {
    float soma = 0.0f;
    for(size_t i = 0; i < y.size(); i++) soma += y[i] * log(yChapeu[i] + 1e-12f);
    return -soma;
}

vector<float> derivadaEntropiaCruzada(const vector<float>& y, const vector<float>& yChapeu) {
    vector<float> deriv(yChapeu.size());
    for(size_t i = 0; i < yChapeu.size(); i++) deriv[i] = yChapeu[i] - y[i];
    return deriv;
}

float huberPerda(const vector<float>& saida, const vector<float>& esperado, float delta = 1.0f) {
    float soma = 0.0f;
    for(size_t i = 0; i < saida.size(); i++) {
        float diff = saida[i] - esperado[i];
        if(abs(diff) <= delta) soma += 0.5f * diff * diff;
        else soma += delta * (abs(diff) - 0.5f * delta);
    }
    return soma / saida.size();
}

vector<float> derivadaHuber(const vector<float>& saida, const vector<float>& esperado, float delta = 1.0f) {
    vector<float> deriv(saida.size());
    for(size_t i = 0; i < saida.size(); i++) {
        float diff = saida[i] - esperado[i];
        
        if(abs(diff) <= delta) deriv[i] = diff;
        else deriv[i] = delta * (diff > 0 ? 1.0f : -1.0f);
    }
    return deriv;
}

float perdaTripleto(const vector<float>& ancora, const vector<float>& positiva, const vector<float>& negativa, float margem = 1.0f) {
    float distPos = 0.0f;
    float distNeg = 0.0f;
    for(size_t i = 0; i < ancora.size(); i++) {
        distPos += pow(ancora[i] - positiva[i], 2);
        distNeg += pow(ancora[i] - negativa[i], 2);
    }
    return max(0.0f, distPos - distNeg + margem);
}

float contrastivaPerda(const vector<float>& saida1, const vector<float>& saida2, int rotulo, float margem = 1.0f) {
    float distancia = 0.0f;
    for(size_t i = 0; i < saida1.size(); i++) distancia += pow(saida1[i] - saida2[i], 2);
    
    if(rotulo == 1) return distancia;
    else return max(0.0f, margem - sqrt(distancia));
}

// funções de saida:
vector<float> softmax(const vector<float>& arr, float temp = 1.0f) {
    // encontra o maior valor para evitar overflow
    float max = *max_element(arr.begin(), arr.end());
    
    // calcula exponenciais
    vector<float> exps(arr.size());
    float soma = 0.0f;
    for(size_t i = 0; i < arr.size(); ++i) {
        exps[i] = exp((arr[i] - max) / temp);
        soma += exps[i];
    }
    // evita divisão por zero
    if(soma < 1e-6f) soma = 1e-6f;
    // normalizar
    for(size_t i = 0; i < exps.size(); ++i) exps[i] /= soma;
    
    return exps;
}

vector<float> derivadaSoftmax(const vector<float>& arr, const vector<float>& gradSaida) {
    float soma = 0.0f;
    
    // calcula soma de gradSaida[i] * arr[i]
    for(size_t j = 0; j < gradSaida.size(); ++j) soma += gradSaida[j] * arr[j];
    
    // calcula derivada
    vector<float> res(arr.size());
    for(size_t i = 0; i < arr.size(); ++i) res[i] = arr[i] * (gradSaida[i] - soma);
    
    return res;
}

vector<vector<float>> softmaxLote(const vector<vector<float>>& m, float temp = 1.0f) {
    vector<vector<float>> res(m.size());
    for(size_t i = 0; i < m.size(); ++i) res[i] = softmax(m[i], temp);
    return res;
}

int argmax(const vector<float>& v) {
    if(v.empty()) return -1; // caso o vetor esteja vazio
    return distance(v.begin(), max_element(v.begin(), v.end()));
}

vector<float> addRuido(const vector<float>& v, float intenso = 0.01f) {
    // configura gerador de números aleatórios
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-intenso, intenso);
    
    vector<float> res(v.size());
    for(size_t i = 0; i < v.size(); ++i) res[i] = v[i] + dis(gen);
    return res;
}

// funções tensores 3D
vector<vector<vector<float>>> tensor3D(int p, int l, int c, float escala = 0.1f) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-escala, escala);

    vector<vector<vector<float>>> t(p, vector<vector<float>>(l, vector<float>(c)));
    for(int i = 0; i < p; ++i) {
        for(int j = 0; j < l; ++j) {
            for(int k = 0; k < c; ++k) t[i][j][k] = dis(gen);
        }
    }
    return t;
}
vector<vector<vector<float>>> zeros3D(int p, int l, int c) {
    return vector<vector<vector<float>>>(p, vector<vector<float>>(l, vector<float>(c, 0.0f)));
}
vector<vector<vector<float>>> mapear3D(const vector<vector<vector<float>>>& t, function<float(float)> fn) {
    vector<vector<vector<float>>> res(t.size(), vector<vector<float>>(t[0].size(), vector<float>(t[0][0].size())));
    for(size_t i = 0; i < t.size(); ++i) {
        for(size_t j = 0; j < t[i].size(); ++j) {
            for(size_t k = 0; k < t[i][j].size(); ++k) res[i][j][k] = fn(t[i][j][k]);
        }
    }
    return res;
}
vector<vector<vector<float>>> somar3D(const vector<vector<vector<float>>>& a, const vector<vector<vector<float>>>& b) {
    if(a.size() != b.size() || a[0].size() != b[0].size() || a[0][0].size() != b[0][0].size()) {
        throw invalid_argument("Dimensões dos tensores incompatíveis em somar3D");
    }
    vector<vector<vector<float>>> res(a.size(), vector<vector<float>>(a[0].size(), vector<float>(a[0][0].size())));
    for(size_t i = 0; i < a.size(); ++i) {
        for(size_t j = 0; j < a[i].size(); ++j) {
            for(size_t k = 0; k < a[i][j].size(); ++k) res[i][j][k] = a[i][j][k] + b[i][j][k];
        }
    }
    return res;
}
vector<vector<vector<float>>> mult3DporEscalar(const vector<vector<vector<float>>>& t, float escalar) {
    vector<vector<vector<float>>> res(t.size(), vector<vector<float>>(t[0].size(), vector<float>(t[0][0].size())));
    for(size_t i = 0; i < t.size(); ++i) {
        for(size_t j = 0; j < t[i].size(); ++j) {
            for(size_t k = 0; k < t[i][j].size(); ++k) res[i][j][k] = t[i][j][k] * escalar;
        }
    }
    return res;
}
vector<vector<vector<float>>> tensorZeros3D(int l, int c, int p) {
    return vector<vector<vector<float>>>(l, vector<vector<float>>(c, vector<float>(p, 0.0f)));
}
vector<vector<float>> aplicarMatrizLote(const vector<vector<float>>& m, const vector<vector<float>>& v) {
    if(m[0].size() != v[0].size()) {
        throw invalid_argument("Dimensões incompatíveis em aplicarMatrizLote");
    }
    vector<vector<float>> res(v.size(), vector<float>(m.size(), 0.0f));
    
    for(size_t i = 0; i < v.size(); ++i) {
        for(size_t j = 0; j < m.size(); ++j) {
            for(size_t k = 0; k < m[0].size(); ++k) {
                res[i][j] += m[j][k] * v[i][k];
            }
        }
    }
    return res;
}
vector<vector<float>> somarVetorMatriz(const vector<vector<float>>& m, const vector<float>& v) {
    if(m.size() != v.size()) {
        throw invalid_argument("Dimensões incompatíveis em somarVetorMatriz");
    }
    vector<vector<float>> resultado = m;
    for(size_t i = 0; i < m.size(); ++i) {
        for(size_t j = 0; j < m[i].size(); ++j) {
            resultado[i][j] += v[i];
        }
    }
    return resultado;
}
// funções matriz 2D
vector<vector<float>> matriz(int l, int c, float escala = 0.1f) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-escala, escala);

    vector<vector<float>> m(l, vector<float>(c));
    for(int i = 0; i < l; ++i) {
        for(int j = 0; j < c; ++j) m[i][j] = dis(gen);
    }
    return m;
}
vector<vector<float>> matrizZeros(size_t linhas, size_t colunas) {
    return vector<vector<float>>(linhas, vector<float>(colunas, 0.0f));
}
vector<vector<float>> exterior(const vector<float>& a, const vector<float>& b) {
    vector<vector<float>> res(a.size(), vector<float>(b.size()));
    for(size_t i = 0; i < a.size(); ++i) {
        for(size_t j = 0; j < b.size(); ++j) res[i][j] = a[i] * b[j];
    }
    return res;
}
vector<vector<float>> somarMatriz(const vector<vector<float>>& a, const vector<vector<float>>& b) {
    if(a.size() != b.size() || a[0].size() != b[0].size()) throw invalid_argument("Dimensões das matrizes incompatíveis em somarMatriz");
    vector<vector<float>> res(a.size(), vector<float>(a[0].size()));
    for(size_t i = 0; i < a.size(); ++i) {
        for(size_t j = 0; j < a[i].size(); ++j) res[i][j] = a[i][j] + b[i][j];
    }
    return res;
}
vector<vector<float>> subMatriz(const vector<vector<float>>& a, const vector<vector<float>>& b) {
    if(a.size() != b.size() || a[0].size() != b[0].size()) throw invalid_argument("Dimensões das matrizes incompatíveis em subMatriz");
    vector<vector<float>> res(a.size(), vector<float>(a[0].size()));
    for(size_t i = 0; i < a.size(); ++i) {
        for(size_t j = 0; j < a[i].size(); ++j) res[i][j] = a[i][j] - b[i][j];
    }
    return res;
}
vector<vector<float>> multMatriz(const vector<vector<float>>& m, float s) {
    vector<vector<float>> res(m.size(), vector<float>(m[0].size()));
    for(size_t i = 0; i < m.size(); ++i) {
        for(size_t j = 0; j < m[i].size(); ++j) res[i][j] = m[i][j] * s;
    }
    return res;
}
vector<vector<float>> multMatrizes(const vector<vector<float>>& a, const vector<vector<float>>& b) {
    if(a[0].size() != b.size()) throw invalid_argument("Dimensões incompatíveis para multiplicação de matrizes");
    vector<vector<float>> res(a.size(), vector<float>(b[0].size(), 0.0f));
    for(size_t i = 0; i < a.size(); ++i) {
        for(size_t j = 0; j < b[0].size(); ++j) {
            for(size_t k = 0; k < a[0].size(); ++k) res[i][j] += a[i][k] * b[k][j];
        }
    }
    return res;
}
vector<float> aplicarMatriz(const vector<vector<float>>& m, const vector<float>& v) {
    if(m[0].size() != v.size()) throw invalid_argument("Dimensões incompatíveis em aplicarMatriz");
    vector<float> res(m.size(), 0.0f);
    for(size_t i = 0; i < m.size(); ++i) {
        for(size_t j = 0; j < v.size(); ++j) res[i] += m[i][j] * v[j];
    }
    return res;
}
vector<vector<float>> transpor(const vector<vector<float>>& m) {
    vector<vector<float>> res(m[0].size(), vector<float>(m.size()));
    for(size_t i = 0; i < m.size(); ++i) {
        for(size_t j = 0; j < m[0].size(); ++j) res[j][i] = m[i][j];
    }
    return res;
}
vector<vector<float>> matrizZeros(int l, int c) {
    return vector<vector<float>>(l, vector<float>(c, 0.0f));
}
vector<vector<float>> identidade(int n) {
    vector<vector<float>> res(n, vector<float>(n, 0.0f));
    for(int i = 0; i < n; ++i) res[i][i] = 1.0f;
    return res;
}
vector<float> matrizVetor(const vector<vector<float>>& m, const vector<float>& v) {
    if(m[0].size() != v.size()) throw invalid_argument("Dimensões incompatíveis em matrizVetor");
    vector<float> res(m.size(), 0.0f);
    for(size_t i = 0; i < m.size(); ++i) {
        for(size_t j = 0; j < v.size(); ++j) res[i] += m[i][j] * v[j];
    }
    return res;
}
vector<float> multVetorMatriz(const vector<float>& v, const vector<float>& c) {
    if(v.size() != c.size()) throw invalid_argument("Dimensões dos vetores incompatíveis em multVetorMatriz");
    vector<float> res(v.size());
    for(size_t i = 0; i < v.size(); ++i) res[i] = v[i] * c[i];
    return res;
}
vector<float> multMatrizVetor(const vector<vector<float>>& m, const vector<float>& v) {
    if(m[0].size() != v.size()) throw invalid_argument("Dimensões incompatíveis em multMatrizVetor");
    vector<float> res(m.size(), 0.0f);
    for(size_t i = 0; i < m.size(); ++i) {
        for(size_t j = 0; j < v.size(); ++j) res[i] += m[i][j] * v[j];
    }
    return res;
}
// vetores:
vector<float> vetor(int c, float escala = 0.1f) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-escala, escala);
    vector<float> v(c);
    for(int i = 0; i < c; ++i) v[i] = dis(gen);
    return v;
}
float escalarDot(const vector<float>& a, const vector<float>& b) {
    if(a.size() != b.size()) throw invalid_argument("Dimensões incompatíveis em escalarDot");
    float soma = 0.0f;
    for(size_t i = 0; i < a.size(); ++i) soma += a[i] * b[i];
    return soma;
}
vector<float> zeros(int n) {
    return vector<float>(n, 0.0f);
}
vector<float> somarVetores(const vector<float>& a, const vector<float>& b) {
    if (a.size() != b.size()) {
        throw invalid_argument("Vetores com dimensões incompatíveis para soma.");
    }
    vector<float> resultado(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        resultado[i] = a[i] + b[i];
    }
    return resultado;
}