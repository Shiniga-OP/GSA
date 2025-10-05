#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <algorithm>
#include <stdexcept>

void _testeMatrizes();

// funções de pesos
std::vector<std::vector<float>> iniPesosXavier(int l, int c) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    float limite = std::sqrt(6.0f / (l + c));
    std::vector<std::vector<float>> pesos(l, std::vector<float>(c));
    
    for(int i = 0; i < l; ++i) {
        for(int j = 0; j < c; ++j) pesos[i][j] = dis(gen) * limite;
    }
    return pesos;
}

std::vector<std::vector<float>> iniPesosHe(int l, int c) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    float limite = std::sqrt(2.0f / l);
    std::vector<std::vector<float>> pesos(l, std::vector<float>(c));
    
    for(int i = 0; i < l; ++i) {
        for(int j = 0; j < c; ++j) pesos[i][j] = dis(gen) * limite;
    }
    return pesos;
}

// att de pesos
std::vector<std::vector<float>> attPesos(const std::vector<std::vector<float>>& pesos, const std::vector<std::vector<float>>& grad, float taxa, float lambda = 1e-3f) {
    std::vector<std::vector<float>> nova(pesos.size(), std::vector<float>(pesos[0].size()));
    
    for(size_t i = 0; i < pesos.size(); ++i) {
        for(size_t j = 0; j < pesos[i].size(); ++j) {
            nova[i][j] = pesos[i][j] - taxa * grad[i][j] - lambda * pesos[i][j];
        }
    }
    return nova;
}

std::vector<std::vector<float>> attPesosMomentum(const std::vector<std::vector<float>>& pesos, const std::vector<std::vector<float>>& grad, float taxa, float momento, std::vector<std::vector<float>>& velocidade) {
    std::vector<std::vector<float>> nova(pesos.size(), std::vector<float>(pesos[0].size()));
    
    for(size_t i = 0; i < pesos.size(); ++i) {
        for(size_t j = 0; j < pesos[i].size(); ++j) {
            velocidade[i][j] = momento * velocidade[i][j] + grad[i][j];
            nova[i][j] = pesos[i][j] - taxa * velocidade[i][j];
        }
    }
    return nova;
}

std::vector<std::vector<float>> attPesosAdam(const std::vector<std::vector<float>>& pesos, const std::vector<std::vector<float>>& grad, std::vector<std::vector<float>>& m, std::vector<std::vector<float>>& v, float taxa, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, int iteracao = 1, float lambda = 0.001f) {
    std::vector<std::vector<float>> nova(pesos.size(), std::vector<float>(pesos[0].size()));
    
    float fator1 = 1.0f - std::pow(beta1, iteracao);
    float fator2 = 1.0f - std::pow(beta2, iteracao);
    float umMenosBeta1 = 1.0f - beta1;
    float umMenosBeta2 = 1.0f - beta2;
    
    for(size_t i = 0; i < pesos.size(); ++i) {
        for(size_t j = 0; j < pesos[i].size(); ++j) {
            float g = grad[i][j] + lambda * pesos[i][j];
            m[i][j] = beta1 * m[i][j] + umMenosBeta1 * g;
            v[i][j] = beta2 * v[i][j] + umMenosBeta2 * g * g;
            float mChapeu = m[i][j] / fator1;
            float vChapeu = v[i][j] / fator2;
            nova[i][j] = pesos[i][j] - taxa * mChapeu / (std::sqrt(vChapeu) + eps);
        }
    }
    return nova;
}

std::vector<float> attPesosAdam1D(std::vector<float>& p, const std::vector<float>& grad, std::vector<float>& m, std::vector<float>& v, float taxa, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, int t = 1, float lambda = 0.001f) {
    float umMenosBeta1 = 1.0f - beta1;
    float umMenosBeta2 = 1.0f - beta2;
    float fator1 = 1.0f - std::pow(beta1, t);
    float fator2 = 1.0f - std::pow(beta2, t);
    
    for(size_t i = 0; i < p.size(); ++i) {
        float g = grad[i] + lambda * p[i];
        m[i] = beta1 * m[i] + umMenosBeta1 * g;
        v[i] = beta2 * v[i] + umMenosBeta2 * g * g;
        float mChapeu = m[i] / fator1;
        float vChapeu = v[i] / fator2;
        p[i] -= taxa * mChapeu / (std::sqrt(vChapeu) + eps);
    }
    return p;
}

// funções de regularização:
std::vector<std::vector<float>> regularL1(const std::vector<std::vector<float>>& pesos, float lambda) {
    std::vector<std::vector<float>> res;
    for(const auto& linha : pesos) {
        std::vector<float> novaLinha;
        for(float p : linha) novaLinha.push_back(lambda * (p > 0 ? 1.0f : (p < 0 ? -1.0f : 0.0f)));
        res.push_back(novaLinha);
    }
    return res;
}

std::vector<std::vector<float>> regularL2(const std::vector<std::vector<float>>& pesos, float lambda) {
    std::vector<std::vector<float>> res;
    for(const auto& linha : pesos) {
        std::vector<float> novaLinha;
        for(float p : linha) novaLinha.push_back(lambda * p);
        res.push_back(novaLinha);
    }
    return res;
}

std::vector<std::vector<float>> dropout(std::vector<std::vector<float>> tensor, float taxa) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for(auto& linha : tensor) {
        for(float& valor : linha) {
            if(dis(gen) < taxa) valor = 0.0f;
            else valor /= (1.0f - taxa);
        }
    }
    return tensor;
}

std::vector<float> normEntrada(const std::vector<float>& vetor) {
    float max = *std::max_element(vetor.begin(), vetor.end());
    float min = *std::min_element(vetor.begin(), vetor.end());
    float amplitude = (max - min) > 1e-8f ? (max - min) : 1e-8f;
    
    std::vector<float> res;
    for(float x : vetor) res.push_back((x - min) / amplitude);
    return res;
}

std::vector<float> normZPonto(const std::vector<float>& v) {
    float soma = 0.0f;
    for(float x : v) soma += x;
    float media = soma / v.size();
    
    float variancia = 0.0f;
    for (float x : v) variancia += std::pow(x - media, 2);
    variancia /= v.size();
    
    float desvio = std::sqrt(variancia + 1e-8f);
    
    std::vector<float> res;
    for(float x : v) res.push_back((x - media) / desvio);
    return res;
}

// funções de metricas
float acuracia(const std::vector<std::vector<float>>& saida, const std::vector<std::vector<float>>& esperado) {
    int corretos = 0;
    for(size_t i = 0; i < saida.size(); i++) {
        int pred = std::distance(saida[i].begin(), std::max_element(saida[i].begin(), saida[i].end()));
        int real = std::distance(esperado[i].begin(), std::max_element(esperado[i].begin(), esperado[i].end()));
        if(pred == real) corretos++;
    }
    return static_cast<float>(corretos) / saida.size();
}

float precisao(const std::vector<std::vector<int>>& confusao) {
    int tp = confusao[0][0];
    int fp = 0;
    for (size_t i = 1; i < confusao[0].size(); i++) {
        fp += confusao[0][i];
    }
    return static_cast<float>(tp) / (tp + fp + 1e-8f);
}

float recall(const std::vector<std::vector<int>>& confusao) {
    int tp = confusao[0][0];
    int fn = 0;
    for(size_t i = 1; i < confusao.size(); i++) fn += confusao[i][0];
    return static_cast<float>(tp) / (tp + fn + 1e-8f);
}

float f1Ponto(const std::vector<std::vector<int>>& confusao) {
    float p = precisao(confusao);
    float r = recall(confusao);
    return 2.0f * (p * r) / (p + r + 1e-8f);
}

float mse(const std::vector<float>& saida, const std::vector<float>& esperado) {
    float soma = 0.0f;
    for(size_t i = 0; i < saida.size(); i++) soma += std::pow(saida[i] - esperado[i], 2);
    return soma / saida.size();
}

float klDivergencia(const std::vector<float>& p, const std::vector<float>& q) {
    float soma = 0.0f;
    for(size_t i = 0; i < p.size(); i++) soma += p[i] * std::log((p[i] + 1e-12f) / (q[i] + 1e-12f));
    return soma;
}

float rocAuc(const std::vector<float>& pontos, const std::vector<int>& rotulos) {
    // cria pares [pontuação, rótulo] e ordenar por pontuação (decrescente)
    std::vector<std::pair<float, int>> pares;
    for(size_t i = 0; i < pontos.size(); i++) pares.push_back({pontos[i], rotulos[i]});
    
    std::sort(pares.begin(), pares.end(), 
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
float erroAbsolutoMedio(const std::vector<float>& saida, const std::vector<float>& esperado) {
    float soma = 0.0f;
    for(size_t i = 0; i < saida.size(); i++) soma += std::abs(saida[i] - esperado[i]);
    return soma / saida.size();
}

float erroQuadradoEsperado(const std::vector<float>& saida, const std::vector<float>& esperado) {
    float soma = 0.0f;
    for(size_t i = 0; i < saida.size(); i++) {
        float diff = saida[i] - esperado[i];
        soma += 0.5f * diff * diff;
    }
    return soma;
}

std::vector<float> derivadaErro(const std::vector<float>& saida, const std::vector<float>& esperado) {
    std::vector<float> deriv(saida.size());
    for(size_t i = 0; i < saida.size(); i++) deriv[i] = saida[i] - esperado[i];
    return deriv;
}

float entropiaCruzada(const std::vector<float>& y, const std::vector<float>& yChapeu) {
    float soma = 0.0f;
    for(size_t i = 0; i < y.size(); i++) soma += y[i] * std::log(yChapeu[i] + 1e-12f);
    return -soma;
}

std::vector<float> derivadaEntropiaCruzada(const std::vector<float>& y, const std::vector<float>& yChapeu) {
    std::vector<float> deriv(yChapeu.size());
    for(size_t i = 0; i < yChapeu.size(); i++) deriv[i] = yChapeu[i] - y[i];
    return deriv;
}

float huberPerda(const std::vector<float>& saida, const std::vector<float>& esperado, float delta = 1.0f) {
    float soma = 0.0f;
    for(size_t i = 0; i < saida.size(); i++) {
        float diff = saida[i] - esperado[i];
        if(std::abs(diff) <= delta) soma += 0.5f * diff * diff;
        else soma += delta * (std::abs(diff) - 0.5f * delta);
    }
    return soma / saida.size();
}

std::vector<float> derivadaHuber(const std::vector<float>& saida, const std::vector<float>& esperado, float delta = 1.0f) {
    std::vector<float> deriv(saida.size());
    for(size_t i = 0; i < saida.size(); i++) {
        float diff = saida[i] - esperado[i];
        
        if(std::abs(diff) <= delta) deriv[i] = diff;
        else deriv[i] = delta * (diff > 0 ? 1.0f : -1.0f);
    }
    return deriv;
}

float perdaTripleto(const std::vector<float>& ancora, const std::vector<float>& positiva, const std::vector<float>& negativa, float margem = 1.0f) {
    float distPos = 0.0f;
    float distNeg = 0.0f;
    for(size_t i = 0; i < ancora.size(); i++) {
        distPos += std::pow(ancora[i] - positiva[i], 2);
        distNeg += std::pow(ancora[i] - negativa[i], 2);
    }
    return std::max(0.0f, distPos - distNeg + margem);
}

float contrastivaPerda(const std::vector<float>& saida1, const std::vector<float>& saida2, int rotulo, float margem = 1.0f) {
    float distancia = 0.0f;
    for(size_t i = 0; i < saida1.size(); i++) distancia += std::pow(saida1[i] - saida2[i], 2);
    
    if(rotulo == 1) return distancia;
    else return std::max(0.0f, margem - std::sqrt(distancia));
}

// funções de saida:
std::vector<float> softmax(const std::vector<float>& arr, float temp = 1.0f) {
    // encontra o maior valor para evitar overflow
    float max = *std::max_element(arr.begin(), arr.end());
    
    // calcula exponenciais
    std::vector<float> exps(arr.size());
    float soma = 0.0f;
    for(size_t i = 0; i < arr.size(); ++i) {
        exps[i] = std::exp((arr[i] - max) / temp);
        soma += exps[i];
    }
    // evita divisão por zero
    if(soma < 1e-6f) soma = 1e-6f;
    // normalizar
    for(size_t i = 0; i < exps.size(); ++i) exps[i] /= soma;
    
    return exps;
}

std::vector<float> derivadaSoftmax(const std::vector<float>& arr, const std::vector<float>& gradSaida) {
    float soma = 0.0f;
    
    // calcula soma de gradSaida[i] * arr[i]
    for(size_t j = 0; j < gradSaida.size(); ++j) soma += gradSaida[j] * arr[j];
    
    // calcula derivada
    std::vector<float> res(arr.size());
    for(size_t i = 0; i < arr.size(); ++i) res[i] = arr[i] * (gradSaida[i] - soma);
    
    return res;
}

std::vector<std::vector<float>> softmaxLote(const std::vector<std::vector<float>>& m, float temp = 1.0f) {
    std::vector<std::vector<float>> res(m.size());
    for(size_t i = 0; i < m.size(); ++i) res[i] = softmax(m[i], temp);
    return res;
}

int argmax(const std::vector<float>& v) {
    if(v.empty()) return -1; // caso o vetor esteja vazio
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

std::vector<float> addRuido(const std::vector<float>& v, float intenso = 0.01f) {
    // configura gerador de números aleatórios
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-intenso, intenso);
    
    std::vector<float> res(v.size());
    for(size_t i = 0; i < v.size(); ++i) res[i] = v[i] + dis(gen);
    return res;
}

// funções matriz 3D
std::vector<std::vector<std::vector<float>>> tensor3D(int p, int l, int c, float escala = 0.1f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-escala, escala);

    std::vector<std::vector<std::vector<float>>> t(p, std::vector<std::vector<float>>(l, std::vector<float>(c)));
    for(int i = 0; i < p; ++i) {
        for(int j = 0; j < l; ++j) {
            for(int k = 0; k < c; ++k) t[i][j][k] = dis(gen);
        }
    }
    return t;
}
std::vector<std::vector<std::vector<float>>> zeros3D(int p, int l, int c) {
    return std::vector<std::vector<std::vector<float>>>(p, std::vector<std::vector<float>>(l, std::vector<float>(c, 0.0f)));
}
std::vector<std::vector<std::vector<float>>> mapear3D(const std::vector<std::vector<std::vector<float>>>& t, std::function<float(float)> fn) {
    std::vector<std::vector<std::vector<float>>> res(t.size(), std::vector<std::vector<float>>(t[0].size(), std::vector<float>(t[0][0].size())));
    for(size_t i = 0; i < t.size(); ++i) {
        for(size_t j = 0; j < t[i].size(); ++j) {
            for(size_t k = 0; k < t[i][j].size(); ++k) res[i][j][k] = fn(t[i][j][k]);
        }
    }
    return res;
}
std::vector<std::vector<std::vector<float>>> somar3D(const std::vector<std::vector<std::vector<float>>>& a, const std::vector<std::vector<std::vector<float>>>& b) {
    if(a.size() != b.size() || a[0].size() != b[0].size() || a[0][0].size() != b[0][0].size()) {
        throw std::invalid_argument("Dimensões dos tensores incompatíveis em somar3D");
    }
    std::vector<std::vector<std::vector<float>>> res(a.size(), std::vector<std::vector<float>>(a[0].size(), std::vector<float>(a[0][0].size())));
    for(size_t i = 0; i < a.size(); ++i) {
        for(size_t j = 0; j < a[i].size(); ++j) {
            for(size_t k = 0; k < a[i][j].size(); ++k) res[i][j][k] = a[i][j][k] + b[i][j][k];
        }
    }
    return res;
}
std::vector<std::vector<std::vector<float>>> mult3DporEscalar(const std::vector<std::vector<std::vector<float>>>& t, float escalar) {
    std::vector<std::vector<std::vector<float>>> res(t.size(), std::vector<std::vector<float>>(t[0].size(), std::vector<float>(t[0][0].size())));
    for(size_t i = 0; i < t.size(); ++i) {
        for(size_t j = 0; j < t[i].size(); ++j) {
            for(size_t k = 0; k < t[i][j].size(); ++k) res[i][j][k] = t[i][j][k] * escalar;
        }
    }
    return res;
}

// funções matriz 2D
std::vector<std::vector<float>> matriz(int l, int c, float escala = 0.1f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-escala, escala);

    std::vector<std::vector<float>> m(l, std::vector<float>(c));
    for(int i = 0; i < l; ++i) {
        for(int j = 0; j < c; ++j) m[i][j] = dis(gen);
    }
    return m;
}
std::vector<std::vector<float>> exterior(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<std::vector<float>> res(a.size(), std::vector<float>(b.size()));
    for(size_t i = 0; i < a.size(); ++i) {
        for(size_t j = 0; j < b.size(); ++j) res[i][j] = a[i] * b[j];
    }
    return res;
}
std::vector<std::vector<float>> somarMatriz(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    if(a.size() != b.size() || a[0].size() != b[0].size()) throw std::invalid_argument("Dimensões das matrizes incompatíveis em somarMatriz");
    std::vector<std::vector<float>> res(a.size(), std::vector<float>(a[0].size()));
    for(size_t i = 0; i < a.size(); ++i) {
        for(size_t j = 0; j < a[i].size(); ++j) res[i][j] = a[i][j] + b[i][j];
    }
    return res;
}
std::vector<std::vector<float>> subMatriz(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    if(a.size() != b.size() || a[0].size() != b[0].size()) throw std::invalid_argument("Dimensões das matrizes incompatíveis em subMatriz");
    std::vector<std::vector<float>> res(a.size(), std::vector<float>(a[0].size()));
    for(size_t i = 0; i < a.size(); ++i) {
        for(size_t j = 0; j < a[i].size(); ++j) res[i][j] = a[i][j] - b[i][j];
    }
    return res;
}
std::vector<std::vector<float>> multMatriz(const std::vector<std::vector<float>>& m, float s) {
    std::vector<std::vector<float>> res(m.size(), std::vector<float>(m[0].size()));
    for(size_t i = 0; i < m.size(); ++i) {
        for(size_t j = 0; j < m[i].size(); ++j) res[i][j] = m[i][j] * s;
    }
    return res;
}
std::vector<std::vector<float>> multMatrizes(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    if(a[0].size() != b.size()) throw std::invalid_argument("Dimensões incompatíveis para multiplicação de matrizes");
    std::vector<std::vector<float>> res(a.size(), std::vector<float>(b[0].size(), 0.0f));
    for(size_t i = 0; i < a.size(); ++i) {
        for(size_t j = 0; j < b[0].size(); ++j) {
            for(size_t k = 0; k < a[0].size(); ++k) res[i][j] += a[i][k] * b[k][j];
        }
    }
    return res;
}
std::vector<float> aplicarMatriz(const std::vector<std::vector<float>>& m, const std::vector<float>& v) {
    if(m[0].size() != v.size()) throw std::invalid_argument("Dimensões incompatíveis em aplicarMatriz");
    std::vector<float> res(m.size(), 0.0f);
    for(size_t i = 0; i < m.size(); ++i) {
        for(size_t j = 0; j < v.size(); ++j) res[i] += m[i][j] * v[j];
    }
    return res;
}
std::vector<std::vector<float>> transpor(const std::vector<std::vector<float>>& m) {
    std::vector<std::vector<float>> res(m[0].size(), std::vector<float>(m.size()));
    for(size_t i = 0; i < m.size(); ++i) {
        for(size_t j = 0; j < m[0].size(); ++j) res[j][i] = m[i][j];
    }
    return res;
}
std::vector<std::vector<float>> matrizZeros(int l, int c) {
    return std::vector<std::vector<float>>(l, std::vector<float>(c, 0.0f));
}
std::vector<std::vector<float>> identidade(int n) {
    std::vector<std::vector<float>> res(n, std::vector<float>(n, 0.0f));
    for(int i = 0; i < n; ++i) res[i][i] = 1.0f;
    return res;
}
std::vector<float> matrizVetor(const std::vector<std::vector<float>>& m, const std::vector<float>& v) {
    if(m[0].size() != v.size()) throw std::invalid_argument("Dimensões incompatíveis em matrizVetor");
    std::vector<float> res(m.size(), 0.0f);
    for(size_t i = 0; i < m.size(); ++i) {
        for(size_t j = 0; j < v.size(); ++j) res[i] += m[i][j] * v[j];
    }
    return res;
}
std::vector<float> multVetorMatriz(const std::vector<float>& v, const std::vector<float>& c) {
    if(v.size() != c.size()) throw std::invalid_argument("Dimensões dos vetores incompatíveis em multVetorMatriz");
    std::vector<float> res(v.size());
    for(size_t i = 0; i < v.size(); ++i) res[i] = v[i] * c[i];
    return res;
}
std::vector<float> multMatrizVetor(const std::vector<std::vector<float>>& m, const std::vector<float>& v) {
    if(m[0].size() != v.size()) throw std::invalid_argument("Dimensões incompatíveis em multMatrizVetor");
    std::vector<float> res(m.size(), 0.0f);
    for(size_t i = 0; i < m.size(); ++i) {
        for(size_t j = 0; j < v.size(); ++j) res[i] += m[i][j] * v[j];
    }
    return res;
}
// vetores:
std::vector<float> vetor(int c, float escala = 0.1f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-escala, escala);
    std::vector<float> v(c);
    for(int i = 0; i < c; ++i) v[i] = dis(gen);
    return v;
}
float escalarDot(const std::vector<float>& a, const std::vector<float>& b) {
    if(a.size() != b.size()) throw std::invalid_argument("Dimensões incompatíveis em escalarDot");
    float soma = 0.0f;
    for(size_t i = 0; i < a.size(); ++i) soma += a[i] * b[i];
    return soma;
}
std::vector<float> zeros(int n) {
    return std::vector<float>(n, 0.0f);
}
std::vector<float> somarVetores(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vetores com dimensões incompatíveis para soma.");
    }
    std::vector<float> resultado(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        resultado[i] = a[i] + b[i];
    }
    return resultado;
}

void testeU() {
    std::cout << "\n=== TESTES SAIDA ===\n\n";
    std::vector<float> entrada = {1.0f, 2.0f, 3.0f};
    std::vector<float> saida_softmax = softmax(entrada, 1.0f);
    std::cout << "Softmax: ";
    for(float x : saida_softmax) std::cout << x << " ";
    std::cout << std::endl;
    std::vector<float> grad = {0.1f, 0.2f, 0.3f};
    std::vector<float> saida_derivada = derivadaSoftmax(saida_softmax, grad);
    std::cout << "Derivada Softmax: ";
    for(float x : saida_derivada) std::cout << x << " ";
    std::cout << std::endl;
    std::vector<std::vector<float>> matriz = {{1.0f, 2.0f, 3.0f}, {1.0f, 3.0f, 2.0f}};
    std::vector<std::vector<float>> saida_lote = softmaxLote(matriz, 1.0f);
    std::cout << "Softmax Lote:\n";
    for(const auto& linha : saida_lote) {
        for(float x : linha) std::cout << x << " ";
        std::cout << std::endl;
    }
    std::cout << "Argmax: " << argmax(entrada) << std::endl;
    std::vector<float> saida_ruido = addRuido(entrada, 0.01f);
    std::cout << "Add Ruido: ";
    for(float x : saida_ruido) std::cout << x << " ";
    std::cout << std::endl;
    std::cout << "\n=== TESTES ERRO ===\n\n";
    std::vector<float> saida = {0.8f, 0.2f, 0.5f};
    std::vector<float> esperado = {1.0f, 0.0f, 0.5f};
    std::vector<float> y = {1.0f, 0.0f, 0.0f};
    std::vector<float> yChapeu = {0.7f, 0.2f, 0.1f};
    std::vector<float> ancora = {0.5f, 0.5f};
    std::vector<float> positiva = {0.6f, 0.6f};
    std::vector<float> negativa = {0.4f, 0.4f};
    std::vector<float> saida1 = {0.1f, 0.9f};
    std::vector<float> saida2 = {0.2f, 0.8f};
    std::cout << "erroAbsolutoMedio: " << erroAbsolutoMedio(saida, esperado) << std::endl;
    std::cout << "erroQuadradoEsperado: " << erroQuadradoEsperado(saida, esperado) << std::endl;
    auto derivErro = derivadaErro(saida, esperado);
    std::cout << "derivadaErro: [";
    for(float val : derivErro) std::cout << val << " ";
    std::cout << "]\n";
    std::cout << "entropiaCruzada: " << entropiaCruzada(y, yChapeu) << std::endl;
    auto derivEntropia = derivadaEntropiaCruzada(y, yChapeu);
    std::cout << "derivadaEntropiaCruzada: [";
    for(float val : derivEntropia) std::cout << val << " ";
    std::cout << "]\n";
    std::cout << "huberPerda: " << huberPerda(saida, esperado) << std::endl;
    auto derivHuber = derivadaHuber(saida, esperado);
    std::cout << "derivadaHuber: [";
    for(float val : derivHuber) std::cout << val << " ";
    std::cout << "]\n";
    std::cout << "perdaTripleto: " << perdaTripleto(ancora, positiva, negativa) << std::endl;
    std::cout << "contrastivaPerda (rotulo=1): " << contrastivaPerda(saida1, saida2, 1) << std::endl;
    std::cout << "contrastivaPerda (rotulo=0): " << contrastivaPerda(saida1, saida2, 0) << std::endl;
    std::cout << "\n=== TESTES REGULARIZAÇÃO E MÉTRICAS ===\n\n";
    std::vector<std::vector<float>> pesos = {{1.0f, -2.0f, 0.0f}, {3.0f, -4.0f, 5.0f}};
    std::vector<std::vector<float>> tensor = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    std::vector<float> vetor = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<std::vector<float>> saidas = {{0.8f, 0.1f, 0.1f}, {0.2f, 0.7f, 0.1f}};
    std::vector<std::vector<float>> esperados = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    std::vector<std::vector<int>> matrizConfusao = {{5, 2}, {1, 8}};
    std::vector<float> p = {0.4f, 0.6f};
    std::vector<float> q = {0.5f, 0.5f};
    std::vector<float> pontos = {0.8f, 0.6f, 0.4f, 0.2f};
    std::vector<int> rotulos = {1, 0, 1, 0};
    auto l1 = regularL1(pesos, 0.1f);
    std::cout << "L1 Regularization: [";
    for(const auto& linha : l1) {
        for(float val : linha) std::cout << val << " ";
    }
    std::cout << "]\n";
    auto l2 = regularL2(pesos, 0.1f);
    std::cout << "L2 Regularization: [";
    for(const auto& linha : l2) {
        for(float val : linha) std::cout << val << " ";
    }
    std::cout << "]\n";
    auto dropoutRes = dropout(tensor, 0.3f);
    std::cout << "Dropout: [";
    for(const auto& linha : dropoutRes) {
        for(float val : linha) std::cout << val << " ";
    }
    std::cout << "]\n";
    auto normEnt = normEntrada(vetor);
    std::cout << "Normalização Entrada: [";
    for(float val : normEnt) std::cout << val << " ";
    std::cout << "]\n";
    auto normZ = normZPonto(vetor);
    std::cout << "Normalização Z: [";
    for(float val : normZ) std::cout << val << " ";
    std::cout << "]\n";
    std::cout << "Acurácia: " << acuracia(saidas, esperados) << "\n";
    std::cout << "Precisão: " << precisao(matrizConfusao) << "\n";
    std::cout << "Recall: " << recall(matrizConfusao) << "\n";
    std::cout << "F1-ponto: " << f1Ponto(matrizConfusao) << "\n";
    std::cout << "MSE: " << mse({1.0f, 2.0f}, {1.5f, 1.8f}) << "\n";
    std::cout << "KL Divergence: " << klDivergencia(p, q) << "\n";
    std::cout << "ROC AUC: " << rocAuc(pontos, rotulos) << "\n";
    std::cout << "\n=== TESTES PESOS ===\n\n";
    std::vector<std::vector<float>> pesos2 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    std::vector<std::vector<float>> grad2 = {{0.1f, 0.2f}, {0.3f, 0.4f}};
    std::vector<std::vector<float>> velocidade = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    std::vector<std::vector<float>> m = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    std::vector<std::vector<float>> v = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    auto xavier = iniPesosXavier(2, 3);
    std::cout << "Xavier: ";
    for(const auto& linha : xavier) {
        for(float val : linha) std::cout << val << " ";
    }
    std::cout << "\n";
    auto novosPesos = attPesos(pesos2, grad2, 0.01f);
    std::cout << "attPesos: ";
    for(const auto& linha : novosPesos) {
        for(float val : linha) std::cout << val << " ";
    }
    std::cout << "\n";
    _testeMatrizes();
}

void _testeMatrizes() {
    std::cout << "\n=== TESTES MATRIZES ===\n\n";
    auto t3d = tensor3D(2, 2, 2, 0.1f);
    std::cout << "Tensor 3D:\n";
    for(const auto& m : t3d) {
        for(const auto& l : m) {
            for (float v : l) std::cout << v << " ";
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    auto z3d = zeros3D(2, 2, 2);
    std::cout << "Zeros 3D:\n";
    for(const auto& m : z3d) {
        for(const auto& l : m) {
            for(float v : l) std::cout << v << " ";
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    auto mapeado = mapear3D(t3d, [](float x) { return x * 2.0f; });
    std::cout << "Mapear 3D (x2):\n";
    for(const auto& m : mapeado) {
        for(const auto& l : m) {
            for(float v : l) std::cout << v << " ";
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    auto soma3d = somar3D(t3d, z3d);
    std::cout << "Soma 3D:\n";
    for(const auto& m : soma3d) {
        for(const auto& l : m) {
            for (float v : l) std::cout << v << " ";
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    auto m = matriz(2, 3, 0.1f);
    std::cout << "Matriz 2x3:\n";
    for(const auto& l : m) {
        for(float v : l) std::cout << v << " ";
        std::cout << "\n";
    }
    std::vector<float> v1 = {1.0f, 2.0f};
    std::vector<float> v2 = {3.0f, 4.0f};
    auto ext = exterior(v1, v2);
    std::cout << "Produto Externo:\n";
    for(const auto& l : ext) {
        for(float v : l) std::cout << v << " ";
        std::cout << "\n";
    }
    auto m2 = matriz(2, 3, 0.1f);
    auto soma_m = somarMatriz(m, m2);
    std::cout << "Soma Matriz:\n";
    for(const auto& l : soma_m) {
        for(float v : l) std::cout << v << " ";
        std::cout << "\n";
    }
    auto m3 = matriz(3, 2, 0.1f);
    auto prod_m = multMatrizes(m, m3);
    std::cout << "Multiplicação Matrizes:\n";
    for(const auto& l : prod_m) {
        for(float v : l) std::cout << v << " ";
        std::cout << "\n";
    }
}