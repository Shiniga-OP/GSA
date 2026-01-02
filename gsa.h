#pragma once
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <math.h>
#include <chrono>
#include <numeric>
#include <iomanip>

// tokenizadores:
class TokenizadorBPE {
public:
    explicit TokenizadorBPE(std::vector<std::pair<std::string,std::string>> merges = std::vector<std::pair<std::string,std::string>>()) {
        // inicia merges
        for(size_t i = 0; i < merges.size(); ++i) {
            std::string chave = merges[i].first + " " + merges[i].second;
            bpeRanks[chave] = i;
        }
        // tokens especiais
        tokenPraId["<ALMO>"] = 0;
        tokenPraId["<DES>"] = 1;
        tokenPraId["<FIM>"] = 2;

        idPraToken[0] = "<ALMO>";
        idPraToken[1] = "<DES>";
        idPraToken[2] = "<FIM>";

        proximoId = 3;
    }

    void construirVocab(const std::vector<std::string>& textos) {
        std::unordered_set<std::string> todosTokens;
        std::unordered_set<char> todosCaracteres;
        // tokens especiais
        todosTokens.insert("<ALMO>");
        todosTokens.insert("<DES>");
        todosTokens.insert("<FIM>");
        // coleta caracteres únicos
        for(const std::string& texto : textos) for(char c : texto) if(!isspace(c)) todosCaracteres.insert(c);

        for(char c : todosCaracteres) todosTokens.insert(std::string(1, c));
        // processa textos para BPE
        for(const std::string& texto : textos) {
            std::vector<std::string> tokens = encode(texto);
            for(const std::string& token : tokens) todosTokens.insert(token);
        }
        // mapeia tokens para IDs
        int id = 3;
        for(const std::string& token : todosTokens) {
            if(tokenPraId.find(token) == tokenPraId.end()) {
                tokenPraId[token] = id;
                idPraToken[id] = token;
                id++;
            }
        }
        proximoId = id;
        printf("Vocabulário construído: %d tokens\n", proximoId);
    }

    std::vector<int> codificar(const std::string& texto) {
        std::vector<std::string> tokensBPE = encode(texto);
        std::vector<int> resultado;
        for(const std::string& token : tokensBPE) {
            auto it = tokenPraId.find(token);
            if(it != tokenPraId.end()) resultado.push_back(it->second);
            else {
                for(char c : token) {
                    auto cit = tokenPraId.find(std::string(1,c));
                    if(cit != tokenPraId.end()) resultado.push_back(cit->second);
                    else resultado.push_back(1); // <DES>
                }
            }
        }
        return resultado;
    }

    std::string decodificar(const std::vector<int>& ids) {
        std::vector<std::string> tokens;
        for(int id : ids) {
            if(id == 2) continue; // <FIM>
            auto it = idPraToken.find(id);
            if(it != idPraToken.end()) tokens.push_back(it->second);
            else tokens.push_back("<DES>");
        }
        return decode(tokens);
    }

    int vocabTam() {
        return proximoId;
    }

public:
    std::unordered_map<std::string,int> tokenPraId;
    std::unordered_map<int, std::string> idPraToken;
    std::unordered_map<std::string,int> bpeRanks;
    std::unordered_map<std::string,std::vector<std::string>> cache;
    int proximoId;

    std::unordered_set<std::string> obterPares(const std::vector<std::string>& palavra) {
        std::unordered_set<std::string> pares;
        for(size_t i = 0; i < palavra.size() - 1; ++i) pares.insert(palavra[i] + " " + palavra[i+1]);
        return pares;
    }

    std::vector<std::string> bpe(const std::string& token) {
        if(cache.find(token) != cache.end()) return cache[token];

        std::vector<std::string> palavra;
        for(char c : token) palavra.push_back(std::string(1,c));

        std::unordered_set<std::string> pares = obterPares(palavra);
        if(pares.empty()) return { token };

        while(true) {
            int minRank = INT32_MAX;
            std::string melhorPar;
            for(const std::string& par : pares) {
                auto it = bpeRanks.find(par);
                if(it != bpeRanks.end() && it->second < minRank) {
                    minRank = it->second;
                    melhorPar = par;
                }
            }
            if(melhorPar.empty()) break;

            std::string primeiro = melhorPar.substr(0, melhorPar.find(' '));
            std::string segundo = melhorPar.substr(melhorPar.find(' ')+1);

            std::vector<std::string> novaPalavra;
            size_t i = 0;
            while(i < palavra.size()) {
                auto it = std::find(palavra.begin()+i, palavra.end(), primeiro);
                if(it == palavra.end()) {
                    novaPalavra.insert(novaPalavra.end(), palavra.begin()+i, palavra.end());
                    break;
                }
                size_t j = it - palavra.begin();
                novaPalavra.insert(novaPalavra.end(), palavra.begin()+i, palavra.begin()+j);
                if(j < palavra.size()-1 && palavra[j+1] == segundo) {
                    novaPalavra.push_back(primeiro+segundo);
                    i = j+2;
                } else {
                    novaPalavra.push_back(primeiro);
                    i = j+1;
                }
            }
            palavra = novaPalavra;
            pares = obterPares(palavra);
        }
        cache[token] = palavra;
        return palavra;
    }

    std::vector<std::string> encode(const std::string& texto) {
        std::vector<std::string> tokens;
        std::istringstream iss(texto);
        std::string palavra;
        while(iss >> palavra) {
            std::vector<std::string> bpeTokens = bpe(palavra);
            if(bpeTokens.size()==1 && bpeTokens[0]==palavra && tokenPraId.find(palavra)==tokenPraId.end())
                for(char c : palavra) tokens.push_back(std::string(1,c));
            else tokens.insert(tokens.end(), bpeTokens.begin(), bpeTokens.end());
            tokens.push_back("Ġ"); // espaço entre palavras
        }
        if(!tokens.empty() && tokens.back()=="Ġ") tokens.pop_back();
        return tokens;
    }

    std::string decode(const std::vector<std::string>& tokens) {
        std::string texto;
        for(const std::string& token : tokens) {
            if(token == "Ġ") texto += ' ';
            else if(token.size() > 1 && token.substr(0,2) == "Ġ") texto += ' ' + token.substr(2);
            else texto += token;
        }
        return texto;
    }
};

void testeT() {
    std::cout << "\n=== TESTES TOKENIZADOR BPE ===\n\n";
    TokenizadorBPE t({});
    std::vector<std::string> textos = { "olá mundo", "teste de tokenização" };
    t.construirVocab(textos);

    std::string frase = "olá mundo";
    std::vector<int> cod = t.codificar(frase);
    std::string dec = t.decodificar(cod);
    std::cout << "Texto: " << frase << std::endl;
    printf("Codificado: ");
    for(int id : cod) printf("%i ", id);
    printf("\nDecodificado: %s\n", dec.c_str());
}

// utilitarios:

void _testeMatrizes();

// funções de pesos
std::vector<std::vector<float>> iniPesosXavier(int l, int c) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    float limite = sqrt(6.0f / (l + c));
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
    
    float limite = sqrt(2.0f / l);
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

std::vector<float> attPesosAdam1D(std::vector<float>& p, const std::vector<float>& grad, std::vector<float>& m, std::vector<float>& v, float taxa, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, int t = 1, float lambda = 0.001f) {
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
    for (float x : v) variancia += pow(x - media, 2);
    variancia /= v.size();
    
    float desvio = sqrt(variancia + 1e-8f);
    
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
    for(size_t i = 0; i < saida.size(); i++) soma += pow(saida[i] - esperado[i], 2);
    return soma / saida.size();
}

float klDivergencia(const std::vector<float>& p, const std::vector<float>& q) {
    float soma = 0.0f;
    for(size_t i = 0; i < p.size(); i++) soma += p[i] * log((p[i] + 1e-12f) / (q[i] + 1e-12f));
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
    for(size_t i = 0; i < saida.size(); i++) soma += abs(saida[i] - esperado[i]);
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
    for(size_t i = 0; i < y.size(); i++) soma += y[i] * log(yChapeu[i] + 1e-12f);
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
        if(abs(diff) <= delta) soma += 0.5f * diff * diff;
        else soma += delta * (abs(diff) - 0.5f * delta);
    }
    return soma / saida.size();
}

std::vector<float> derivadaHuber(const std::vector<float>& saida, const std::vector<float>& esperado, float delta = 1.0f) {
    std::vector<float> deriv(saida.size());
    for(size_t i = 0; i < saida.size(); i++) {
        float diff = saida[i] - esperado[i];
        
        if(abs(diff) <= delta) deriv[i] = diff;
        else deriv[i] = delta * (diff > 0 ? 1.0f : -1.0f);
    }
    return deriv;
}

float perdaTripleto(const std::vector<float>& ancora, const std::vector<float>& positiva, const std::vector<float>& negativa, float margem = 1.0f) {
    float distPos = 0.0f;
    float distNeg = 0.0f;
    for(size_t i = 0; i < ancora.size(); i++) {
        distPos += pow(ancora[i] - positiva[i], 2);
        distNeg += pow(ancora[i] - negativa[i], 2);
    }
    return std::max(0.0f, distPos - distNeg + margem);
}

float contrastivaPerda(const std::vector<float>& saida1, const std::vector<float>& saida2, int rotulo, float margem = 1.0f) {
    float distancia = 0.0f;
    for(size_t i = 0; i < saida1.size(); i++) distancia += pow(saida1[i] - saida2[i], 2);
    
    if(rotulo == 1) return distancia;
    else return std::max(0.0f, margem - sqrt(distancia));
}

// funções de saida:
std::vector<float> softmax(const std::vector<float>& arr, float temp = 1.0f) {
    // encontra o maior valor para evitar overflow
    float max = *std::max_element(arr.begin(), arr.end());
    
    // calcula exponenciais
    std::vector<float> exps(arr.size());
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

// funções tensores 3D
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
std::vector<std::vector<std::vector<float>>> tensorZeros3D(int l, int c, int p) {
    return std::vector<std::vector<std::vector<float>>>(l, std::vector<std::vector<float>>(c, std::vector<float>(p, 0.0f)));
}
std::vector<std::vector<float>> aplicarMatrizLote(const std::vector<std::vector<float>>& m, const std::vector<std::vector<float>>& v) {
    if(m[0].size() != v[0].size()) {
        throw std::invalid_argument("Dimensões incompatíveis em aplicarMatrizLote");
    }
    std::vector<std::vector<float>> res(v.size(), std::vector<float>(m.size(), 0.0f));
    
    for(size_t i = 0; i < v.size(); ++i) {
        for(size_t j = 0; j < m.size(); ++j) {
            for(size_t k = 0; k < m[0].size(); ++k) {
                res[i][j] += m[j][k] * v[i][k];
            }
        }
    }
    return res;
}
std::vector<std::vector<float>> somarVetorMatriz(const std::vector<std::vector<float>>& m, const std::vector<float>& v) {
    if(m.size() != v.size()) {
        throw std::invalid_argument("Dimensões incompatíveis em somarVetorMatriz");
    }
    std::vector<std::vector<float>> resultado = m;
    for(size_t i = 0; i < m.size(); ++i) {
        for(size_t j = 0; j < m[i].size(); ++j) {
            resultado[i][j] += v[i];
        }
    }
    return resultado;
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

// ativacoes:
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

// camadas:
// DENSA:
class CamadaDensa {
public:
    CamadaDensa(int dimEntrada, int dimSaida, float taxaDropout = 0.0f);
    std::vector<std::vector<float>> propagar(const std::vector<std::vector<float>>& x, bool treino = true);
    std::vector<std::vector<float>> retropropagar(const std::vector<std::vector<float>>& dY, float taxa, float lambda = 0.001f);
    std::vector<std::vector<float>> p1, p2;
    std::vector<float> b1, b2;

    std::vector<std::vector<float>> m1, v1, m2, v2;
    std::vector<float> mB1, vB1, mB2, vB2;

    int iteracao;
    float taxaDrop;
    int dimEntrada_, dimSaida_;
    float ultimoErro;

    std::vector<std::vector<float>> cache_x;
    std::vector<std::vector<float>> cache_z1;
    std::vector<std::vector<float>> cache_ativ1;
    std::vector<std::vector<bool>> cache_mascara_dropout;
};

inline CamadaDensa::CamadaDensa(int dimEntrada, int dimSaida, float taxaDropout) 
    : dimEntrada_(dimEntrada), dimSaida_(dimSaida), taxaDrop(taxaDropout), iteracao(1), ultimoErro(0.0f) {
    
    this->p1 = iniPesosXavier(dimSaida, dimEntrada);
    this->b1 = zeros(dimSaida);
    this->p2 = iniPesosXavier(dimEntrada, dimSaida);
    this->b2 = zeros(dimEntrada);

    this->m1 = matrizZeros(dimSaida, dimEntrada);
    this->v1 = matrizZeros(dimSaida, dimEntrada);
    this->m2 = matrizZeros(dimEntrada, dimSaida);
    this->v2 = matrizZeros(dimEntrada, dimSaida);
    
    this->mB1 = zeros(dimSaida);
    this->vB1 = zeros(dimSaida);
    this->mB2 = zeros(dimEntrada);
    this->vB2 = zeros(dimEntrada);
}

inline std::vector<std::vector<float>> CamadaDensa::propagar(const std::vector<std::vector<float>>& x, bool treino) {
    int lote = x.size();
    
    auto z1 = matrizZeros(lote, dimSaida_);
    auto ativ1 = matrizZeros(lote, dimSaida_);
    auto saida = matrizZeros(lote, dimEntrada_);

    this->cache_x = x;
    // camada 1
    // acesso sequencial
    for(int i = 0; i < lote; ++i) {
        const auto& x_i = x[i];
        auto& z1_i = z1[i];
        
        for (int j = 0; j < dimSaida_; ++j) {
            float soma = b1[j];
            const auto& p1_j = p1[j];
            for (int k = 0; k < dimEntrada_; ++k) {
                soma += x_i[k] * p1_j[k];
            }
            z1_i[j] = soma;
        }
    }
    this->cache_z1 = z1;

    for(int i = 0; i < lote; ++i) {
        for(int j = 0; j < dimSaida_; ++j) {
            ativ1[i][j] = tanh(z1[i][j]);
        }
    }
    this->cache_ativ1 = ativ1;

    auto ativ1_dropout = ativ1;
    this->cache_mascara_dropout.clear();
    
    if(treino && taxaDrop > 0.0f) {
        this->cache_mascara_dropout.resize(lote, std::vector<bool>(dimSaida_, false));
        float escala = 1.0f / (1.0f - taxaDrop);
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        for(int i = 0; i < lote; ++i) {
            for(int j = 0; j < dimSaida_; ++j) {
                if(dist(gen) < taxaDrop) {
                    ativ1_dropout[i][j] = 0.0f;
                    cache_mascara_dropout[i][j] = false;
                } else {
                    ativ1_dropout[i][j] *= escala;
                    cache_mascara_dropout[i][j] = true;
                }
            }
        }
    }

    // camada 2
    for(int i = 0; i < lote; ++i) {
        const auto& ativ_i = ativ1_dropout[i];
        auto& saida_i = saida[i];
        
        for(int j = 0; j < dimEntrada_; ++j) {
            float soma = b2[j];
            const auto& p2_j = p2[j];
            for(int k = 0; k < dimSaida_; ++k) {
                soma += ativ_i[k] * p2_j[k];
            }
            saida_i[j] = soma;
        }
    }
    return saida;
}

inline std::vector<std::vector<float>> CamadaDensa::retropropagar(const std::vector<std::vector<float>>& dY, float taxa, float lambda) {
    int lote = dY.size();
    
    auto dP2 = matrizZeros(dimEntrada_, dimSaida_);
    auto dB2 = zeros(dimEntrada_);
    auto dP1 = matrizZeros(dimSaida_, dimEntrada_);
    auto dB1 = zeros(dimSaida_);
    auto dO = matrizZeros(lote, dimSaida_);
    auto dX = matrizZeros(lote, dimEntrada_);

    // gradiente da camada 2
    for(int i = 0; i < lote; ++i) {
        const auto& dY_i = dY[i];
        for(int j = 0; j < dimEntrada_; ++j) {
            dB2[j] += dY_i[j];
            
            const auto& ativ1_i = cache_ativ1[i];
            for(int k = 0; k < dimSaida_; ++k) {
                bool manter = cache_mascara_dropout.empty() || cache_mascara_dropout[i][k];
                if(manter) {
                    float escala = cache_mascara_dropout.empty() ? 1.0f : (1.0f / (1.0f - taxaDrop));
                    dP2[j][k] += dY_i[j] * ativ1_i[k] * escala;
                }
            }
        }
    }
    // camada 2
    for(int i = 0; i < lote; ++i) {
        const auto& dY_i = dY[i];
        auto& dO_i = dO[i];
        
        for(int j = 0; j < dimSaida_; ++j) {
            float soma = 0.0f;
            for(int k = 0; k < dimEntrada_; ++k) {
                soma += dY_i[k] * p2[k][j];
            }
            dO_i[j] = soma;
        }
    }
    // dropout e derivada da Tanh
    for(int i = 0; i < lote; ++i) {
        auto& dO_i = dO[i];
        const auto& z1_i = cache_z1[i];
        
        for(int j = 0; j < dimSaida_; ++j) {
            bool manter = cache_mascara_dropout.empty() || cache_mascara_dropout[i][j];
            if(!manter) {
                dO_i[j] = 0.0f;
            } else {
                // derivada da tanh: 1 - tanh^2(x)
                float tanh_x = tanh(z1_i[j]);
                dO_i[j] *= (1.0f - tanh_x * tanh_x);
            }
        }
    }
    // gradiente da camada 1
    for(int i = 0; i < lote; ++i) {
        const auto& dO_i = dO[i];
        const auto& x_i = cache_x[i];
        
        for(int j = 0; j < dimSaida_; ++j) {
            dB1[j] += dO_i[j];
            
            for(int k = 0; k < dimEntrada_; ++k) {
                dP1[j][k] += dO_i[j] * x_i[k];
            }
        }
    }

    // gradiente pra entrada anterior
    for(int i = 0; i < lote; ++i) {
        const auto& dO_i = dO[i];
        auto& dX_i = dX[i];
        
        for(int j = 0; j < dimEntrada_; ++j) {
            float soma = 0.0f;
            for(int k = 0; k < dimSaida_; ++k) {
                soma += dO_i[k] * p1[k][j];
            }
            dX_i[j] = soma;
        }
    }
    // normalização
    float invLote = 1.0f / lote;
    for(auto& linha : dP1) for (auto& val : linha) val *= invLote;
    for(auto& linha : dP2) for (auto& val : linha) val *= invLote;
    for(auto& val : dB1) val *= invLote;
    for(auto& val : dB2) val *= invLote;
    // ATUALIZAÇÃO DOS PESOS
    this->p1 = attPesosAdam(this->p1, dP1, this->m1, this->v1, taxa, 0.9f, 0.999f, 1e-8f, this->iteracao, lambda);
    this->p2 = attPesosAdam(this->p2, dP2, this->m2, this->v2, taxa, 0.9f, 0.999f, 1e-8f, this->iteracao, lambda);
    this->b1 = attPesosAdam1D(this->b1, dB1, this->mB1, this->vB1, taxa, 0.9f, 0.999f, 1e-8f, this->iteracao, lambda);
    this->b2 = attPesosAdam1D(this->b2, dB2, this->mB2, this->vB2, taxa, 0.9f, 0.999f, 1e-8f, this->iteracao, lambda);
    this->iteracao++;
    return dX;
}

void testeCD() {
    std::cout << "\n=== TESTE CAMADA DENSA ===\n\n";
    int dimEntrada = 10;
    int dimOculta = 50;
    int dimSaida = 10;
    int tamanhoLote = 32;
    int epocas = 300;
    // sem dropout pra tarefa de identidade
    CamadaDensa camada(dimEntrada, dimOculta, 0.0f);
    
    auto entrada = std::vector<std::vector<float>>(tamanhoLote);
    auto saidaEsperada = std::vector<std::vector<float>>(tamanhoLote);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for(int i = 0; i < tamanhoLote; ++i) {
        entrada[i] = std::vector<float>(dimEntrada);
        for(int j = 0; j < dimEntrada; ++j) {
            entrada[i][j] = dist(gen);
        }
        saidaEsperada[i] = entrada[i];
    }
    // taxa de aprendizado adaptativa
    float taxaBase = 0.005f;
    std::vector<float> historicoErro;
    
    std::cout << "Treinando função identidade...\n";
    std::cout << "Época\tErro Médio\n";
    std::cout << "------\t----------\n";
    
    auto inicio = std::chrono::high_resolution_clock::now();
    
    for(int epoca = 0; epoca < epocas; ++epoca) {
        // decaimento
        float taxaAtual = taxaBase * (1.0f / (1.0f + 0.01f * epoca));
        
        auto saida = camada.propagar(entrada, true);
        // erro quadrático médio
        float erroTotal = 0.0f;
        for(int i = 0; i < tamanhoLote; ++i) {
            for(int j = 0; j < dimSaida; ++j) {
                float diff = saida[i][j] - saidaEsperada[i][j];
                erroTotal += diff * diff;
            }
        }
        float erroMedio = erroTotal / (tamanhoLote * dimSaida);
        historicoErro.push_back(erroMedio);
        
        if(epoca % 50 == 0) {
            std::cout << epoca << "\t" << erroMedio << "\n";
        }
        // gradiente do MSE
        auto gradSaida = saida;
        for(int i = 0; i < tamanhoLote; ++i) {
            for(int j = 0; j < dimSaida; ++j) {
                gradSaida[i][j] = 2.0f * (saida[i][j] - saidaEsperada[i][j]) / tamanhoLote;
            }
        }
        camada.retropropagar(gradSaida, taxaAtual, 0.0001f);
    }
    auto fim = std::chrono::high_resolution_clock::now();
    auto duracao = std::chrono::duration_cast<std::chrono::milliseconds>(fim - inicio);
    // avaliação
    auto saidaFinal = camada.propagar(entrada, false);
    float erroFinal = 0.0f;
    for(int i = 0; i < tamanhoLote; ++i) {
        for(int j = 0; j < dimSaida; ++j) {
            float diff = saidaFinal[i][j] - saidaEsperada[i][j];
            erroFinal += abs(diff);
        }
    }
    erroFinal /= (tamanhoLote * dimSaida);
    
    std::cout << "\n--- RESULTADOS ---\n";
    std::cout << "Erro final: " << erroFinal << "\n";
    std::cout << "Tempo: " << duracao.count() << " ms\n";
    std::cout << "Redução: " << ((historicoErro[0] - historicoErro.back()) / historicoErro[0] * 100) << "%\n";
    
    if(erroFinal < 0.02f) {
        std::cout << "🎉 EXCELENTE: Aprendizado quase perfeito!\n";
    } else if(erroFinal < 0.05f) {
        std::cout << "✅ MUITO BOM: Aprendizado sólido!\n";
    } else if(erroFinal < 0.1f) {
        std::cout << "👍 BOM: Aprendizado funcional\n";
    } else if(erroFinal < 0.2f) {
        std::cout << "⚠️  RAZOÁVEL: Aprendizado parcial\n";
    } else {
        std::cout << "❌ INSUFICIENTE\n";
    }
    
    std::cout << "\n📊 Comparação (primeira amostra):\n";
    int exemplos = 3;
    for(int i = 0; i < exemplos && i < dimEntrada; ++i) {
        printf("Entrada[%d]: %7.3f → Saída: %7.3f (Esperado: %7.3f) %s\n", 
               i, entrada[0][i], saidaFinal[0][i], saidaEsperada[0][i],
               std::abs(saidaFinal[0][i] - saidaEsperada[0][i]) < 0.1f ? "✓" : "✗");
    }
    std::cout << "\n=== TESTE DE DESEMPENHO ===\n\n";
    const int lote = 16;
    dimEntrada = 64;
    dimOculta = 128;
    const int iteracoes = 10;

    camada = CamadaDensa(dimEntrada, dimOculta, 0.1f);
    
    auto x = std::vector<std::vector<float>>(lote);
    for(int i = 0; i < lote; ++i) {
        x[i] = vetor(dimEntrada, 1.0f);
    }
    inicio = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < iteracoes; ++i) {
        auto saida = camada.propagar(x, true);
        auto grad = saida;
        for(auto& linha : grad) for(auto& val : linha) val = (val - 1.0f) / lote;
        auto dX = camada.retropropagar(grad, 1e-4f);
    }
    fim = std::chrono::high_resolution_clock::now();
    duracao = std::chrono::duration_cast<std::chrono::milliseconds>(fim - inicio);

    std::cout << "Config: Lote=" << lote << ", Entrada=" << dimEntrada << ", Oculto=" << dimOculta << "\n";
    std::cout << "Tempo total: " << duracao.count() << " ms\n";
    std::cout << "Tempo/iteração: " << duracao.count() / iteracoes << " ms\n";
    
    double operacoes = (double)iteracoes * (double)lote * ((double)dimEntrada * dimOculta * 2 + (double)dimOculta * dimEntrada * 2);
    double segundos = duracao.count() / 1000.0;
    std::cout << "Performance: " << operacoes / segundos / 1e6 << " MFLOPS\n";
    
    double tempo_por_iteracao = (double)duracao.count() / iteracoes;
    if(tempo_por_iteracao > 0) {
        std::cout << "Velocidade atual: ~" << 800.0 / tempo_por_iteracao << "x\n";
    }
}
// CONVOLUÇÃO:
class CamadaConv {
public:
    CamadaConv(int entradaAltura, int entradaLargura, int entradaCanais, int numFiltros, int filtroTamanho, int passo = 1, int prenchimento = 0, float taxaDropout = 0.0f);
    
    std::vector<std::vector<std::vector<float>>> propagar(
        const std::vector<std::vector<std::vector<float>>>& entrada, 
        bool treino = true);
    
    std::vector<std::vector<std::vector<float>>> retropropagar(
        const std::vector<std::vector<std::vector<float>>>& gradSaida,
        float taxaAprendizado, float lambda = 0.001f);
    
    void imprimirInfo() const;
    void defFiltro(int filtroIdc, int canalIdc, const std::vector<std::vector<float>>& filtro);
    void defBias(int filtroIdc, float bias);
    int entradaAltura_, entradaLargura_, entradaCanais_;
    int numFiltros_, filtroTam_, passo_, prenchimento_;
    float taxaDropout_;
    
    std::vector<std::vector<std::vector<std::vector<float>>>> filtros_; // [filtro][canal][altura][largura]
    std::vector<float> biases_;
    
    // cache pra retropropagação
    std::vector<std::vector<std::vector<float>>> cacheEntrada_;
    std::vector<std::vector<std::vector<float>>> cacheSaida_;
    std::vector<std::vector<std::vector<bool>>> cacheMascaraDropout_;
    
    std::vector<std::vector<std::vector<std::vector<float>>>> mFiltros_, vFiltros_;
    std::vector<float> mBiases_, vBiases_;
    int iteracao_;
    
    int calcularSaidaAltura() const;
    int calcularSaidalargura() const;
    void aplicarprenchimento(const std::vector<std::vector<std::vector<float>>>& entrada,
    std::vector<std::vector<std::vector<float>>>& saidaComprenchimento) const;
    
    std::vector<std::vector<std::vector<std::vector<float>>>> criarGradFiltros() const;
};

inline CamadaConv::CamadaConv(int entradaAltura, int entradaLargura, int entradaCanais, int numFiltros, int filtroTamanho, int passo, int prenchimento, float taxaDropout)
    : entradaAltura_(entradaAltura), entradaLargura_(entradaLargura), 
      entradaCanais_(entradaCanais), numFiltros_(numFiltros),
      filtroTam_(filtroTamanho), passo_(passo), prenchimento_(prenchimento),
      taxaDropout_(taxaDropout), iteracao_(1) {
    
    float escala = sqrt(2.0f / (filtroTamanho * filtroTamanho * entradaCanais));
    
    filtros_.resize(numFiltros);
    mFiltros_.resize(numFiltros);
    vFiltros_.resize(numFiltros);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, escala);
    
    for(int f = 0; f < numFiltros; ++f) {
        filtros_[f].resize(entradaCanais);
        mFiltros_[f].resize(entradaCanais);
        vFiltros_[f].resize(entradaCanais);
        
        for(int c = 0; c < entradaCanais; ++c) {
            filtros_[f][c].resize(filtroTamanho, std::vector<float>(filtroTamanho));
            mFiltros_[f][c].resize(filtroTamanho, std::vector<float>(filtroTamanho, 0.0f));
            vFiltros_[f][c].resize(filtroTamanho, std::vector<float>(filtroTamanho, 0.0f));
            
            for(int i = 0; i < filtroTamanho; ++i) {
                for(int j = 0; j < filtroTamanho; ++j) {
                    filtros_[f][c][i][j] = dist(gen);
                }
            }
        }
    }
    biases_.resize(numFiltros, 0.1f);
    mBiases_.resize(numFiltros, 0.0f);
    vBiases_.resize(numFiltros, 0.0f);
}

inline std::vector<std::vector<std::vector<std::vector<float>>>> CamadaConv::criarGradFiltros() const {
    std::vector<std::vector<std::vector<std::vector<float>>>> gradFiltros(
        numFiltros_,
        std::vector<std::vector<std::vector<float>>>(
            entradaCanais_,
            std::vector<std::vector<float>>(
                filtroTam_,
                std::vector<float>(filtroTam_, 0.0f))));
    return gradFiltros;
}

inline int CamadaConv::calcularSaidaAltura() const {
    return (entradaAltura_ + 2 * prenchimento_ - filtroTam_) / passo_ + 1;
}

inline int CamadaConv::calcularSaidalargura() const {
    return (entradaLargura_ + 2 * prenchimento_ - filtroTam_) / passo_ + 1;
}

inline void CamadaConv::aplicarprenchimento(const std::vector<std::vector<std::vector<float>>>& entrada, std::vector<std::vector<std::vector<float>>>& saidaComprenchimento) const {
    int novaAltura = entradaAltura_ + 2 * prenchimento_;
    int novaLargura = entradaLargura_ + 2 * prenchimento_;
    
    saidaComprenchimento.resize(entradaCanais_, 
    std::vector<std::vector<float>>(novaAltura, 
    std::vector<float>(novaLargura, 0.0f)));
    
    for(int c = 0; c < entradaCanais_; ++c) {
        for(int i = 0; i < entradaAltura_; ++i) {
            for(int j = 0; j < entradaLargura_; ++j) {
                saidaComprenchimento[c][i + prenchimento_][j + prenchimento_] = entrada[c][i][j];
            }
        }
    }
}

inline std::vector<std::vector<std::vector<float>>> CamadaConv::propagar(
    const std::vector<std::vector<std::vector<float>>>& entrada, bool treino) {
    
    cacheEntrada_ = entrada;
    
    int saidaAltura = calcularSaidaAltura();
    int saidaLargura = calcularSaidalargura();
    
    std::vector<std::vector<std::vector<float>>> entradaComprenchimento;
    if(prenchimento_ > 0) {
        aplicarprenchimento(entrada, entradaComprenchimento);
    } else {
        entradaComprenchimento = entrada;
    }
    std::vector<std::vector<std::vector<float>>> saida(
        numFiltros_, 
        std::vector<std::vector<float>>(saidaAltura, 
        std::vector<float>(saidaLargura, 0.0f)));
    // convolução
    for(int filtro = 0; filtro < numFiltros_; ++filtro) {
        for(int i = 0; i < saidaAltura; ++i) {
            for(int j = 0; j < saidaLargura; ++j) {
                float soma = biases_[filtro];
                int inicioAltura = i * passo_;
                int inicioLargura = j * passo_;
                // aplica filtro
                for(int canal = 0; canal < entradaCanais_; ++canal) {
                    for(int fi = 0; fi < filtroTam_; ++fi) {
                        for(int fj = 0; fj < filtroTam_; ++fj) {
                            int posAltura = inicioAltura + fi;
                            int posLargura = inicioLargura + fj;
                            
                            soma += entradaComprenchimento[canal][posAltura][posLargura] * 
                                   filtros_[filtro][canal][fi][fj];
                        }
                    }
                }
                // aplica ReLU
                saida[filtro][i][j] = std::max(0.0f, soma);
            }
        }
    }
    cacheSaida_ = saida;
    // dropout
    if(treino && taxaDropout_ > 0.0f) {
        cacheMascaraDropout_.resize(numFiltros_);
        float escala = 1.0f / (1.0f - taxaDropout_);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        for(int filtro = 0; filtro < numFiltros_; ++filtro) {
            cacheMascaraDropout_[filtro].resize(saidaAltura, 
            std::vector<bool>(saidaLargura, false));
            
            for(int i = 0; i < saidaAltura; ++i) {
                for(int j = 0; j < saidaLargura; ++j) {
                    if(dist(gen) < taxaDropout_) {
                        saida[filtro][i][j] = 0.0f;
                        cacheMascaraDropout_[filtro][i][j] = false;
                    } else {
                        saida[filtro][i][j] *= escala;
                        cacheMascaraDropout_[filtro][i][j] = true;
                    }
                }
            }
        }
    }
    return saida;
}

inline std::vector<std::vector<std::vector<float>>> CamadaConv::retropropagar(
    const std::vector<std::vector<std::vector<float>>>& gradSaida,
    float taxaAprendizado, float lambda) {
    
    int saidaAltura = calcularSaidaAltura();
    int saidaLargura = calcularSaidalargura();
    
    // gradientes pra filtros e biases
    auto gradFiltros = criarGradFiltros();
    std::vector<float> gradBiases(numFiltros_, 0.0f);
    // gradiente pra a entrada anterior
    std::vector<std::vector<std::vector<float>>> gradEntrada(
        entradaCanais_,
        std::vector<std::vector<float>>(entradaAltura_ + 2 * prenchimento_,
        std::vector<float>(entradaLargura_ + 2 * prenchimento_, 0.0f)));
        
    std::vector<std::vector<std::vector<float>>> entradaComprenchimento;
    aplicarprenchimento(cacheEntrada_, entradaComprenchimento);
    // calculo dos gradientes
    for(int filtro = 0; filtro < numFiltros_; ++filtro) {
        for(int i = 0; i < saidaAltura; ++i) {
            for(int j = 0; j < saidaLargura; ++j) {
                // pula se dropout aplicado
                if(!cacheMascaraDropout_.empty() && 
                   !cacheMascaraDropout_[filtro][i][j]) {
                    continue;
                }
                float grad = gradSaida[filtro][i][j];
                // aplica gradiente da ReLU
                if(cacheSaida_[filtro][i][j] <= 0) {
                    grad = 0.0f;
                }
                gradBiases[filtro] += grad;
                
                int inicioAltura = i * passo_;
                int inicioLargura = j * passo_;
                // att gradientes dos filtros
                for(int canal = 0; canal < entradaCanais_; ++canal) {
                    for(int fi = 0; fi < filtroTam_; ++fi) {
                        for(int fj = 0; fj < filtroTam_; ++fj) {
                            int posAltura = inicioAltura + fi;
                            int posLargura = inicioLargura + fj;
                            
                            gradFiltros[filtro][canal][fi][fj] += 
                                grad * entradaComprenchimento[canal][posAltura][posLargura];
                        }
                    }
                }
                // calcula gradiente pra entrada anterior
                for(int canal = 0; canal < entradaCanais_; ++canal) {
                    for(int fi = 0; fi < filtroTam_; ++fi) {
                        for(int fj = 0; fj < filtroTam_; ++fj) {
                            int posAltura = inicioAltura + fi;
                            int posLargura = inicioLargura + fj;
                            
                            gradEntrada[canal][posAltura][posLargura] += 
                                grad * filtros_[filtro][canal][fi][fj];
                        }
                    }
                }
            }
        }
    }
    std::vector<std::vector<std::vector<float>>> gradEntradaFinal(
        entradaCanais_,
        std::vector<std::vector<float>>(entradaAltura_,
        std::vector<float>(entradaLargura_, 0.0f)));
    
    for(int canal = 0; canal < entradaCanais_; ++canal) {
        for(int i = 0; i < entradaAltura_; ++i) {
            for(int j = 0; j < entradaLargura_; ++j) {
                gradEntradaFinal[canal][i][j] = 
                    gradEntrada[canal][i + prenchimento_][j + prenchimento_];
            }
        }
    }
    for(int filtro = 0; filtro < numFiltros_; ++filtro) {
        for(int canal = 0; canal < entradaCanais_; ++canal) {
            for(int fi = 0; fi < filtroTam_; ++fi) {
                for(int fj = 0; fj < filtroTam_; ++fj) {
                    float g = gradFiltros[filtro][canal][fi][fj] + 
                             lambda * filtros_[filtro][canal][fi][fj];
                    
                    mFiltros_[filtro][canal][fi][fj] = 0.9f * mFiltros_[filtro][canal][fi][fj] + 
                                                       0.1f * g;
                    vFiltros_[filtro][canal][fi][fj] = 0.999f * vFiltros_[filtro][canal][fi][fj] + 
                                                       0.001f * g * g;
                    
                    float mCorrigido = mFiltros_[filtro][canal][fi][fj] / 
                                     (1 - pow(0.9f, iteracao_));
                    float vCorrigido = vFiltros_[filtro][canal][fi][fj] / 
                                     (1 - pow(0.999f, iteracao_));
                    
                    filtros_[filtro][canal][fi][fj] -= 
                        taxaAprendizado * mCorrigido / (sqrt(vCorrigido) + 1e-8f);
                }
            }
        }
        float gBias = gradBiases[filtro] + lambda * biases_[filtro];
        mBiases_[filtro] = 0.9f * mBiases_[filtro] + 0.1f * gBias;
        vBiases_[filtro] = 0.999f * vBiases_[filtro] + 0.001f * gBias * gBias;
        
        float mBiasCorrigido = mBiases_[filtro] / (1 - pow(0.9f, iteracao_));
        float vBiasCorrigido = vBiases_[filtro] / (1 - pow(0.999f, iteracao_));
        
        biases_[filtro] -= taxaAprendizado * mBiasCorrigido / (sqrt(vBiasCorrigido) + 1e-8f);
    }
    iteracao_++;
    return gradEntradaFinal;
}

inline void CamadaConv::imprimirInfo() const {
    std::cout << "Camada Conv - Entrada: " << entradaAltura_ << "x" << entradaLargura_ 
              << "x" << entradaCanais_ << " | Filtros: " << numFiltros_ 
              << " (" << filtroTam_ << "x" << filtroTam_ << ") | Saída: " 
              << calcularSaidaAltura() << "x" << calcularSaidalargura() 
              << "x" << numFiltros_ << std::endl;
}
inline void CamadaConv::defFiltro(int filtroIdc, int canalIdc, const std::vector<std::vector<float>>& filtro) {
    if(filtroIdc >= 0 && filtroIdc < numFiltros_ && 
       canalIdc >= 0 && canalIdc < entradaCanais_ &&
       filtro.size() == filtroTam_ && 
       filtro[0].size() == filtroTam_) {
        filtros_[filtroIdc][canalIdc] = filtro;
    }
}

inline void CamadaConv::defBias(int filtroIdc, float bias) {
    if(filtroIdc >= 0 && filtroIdc < numFiltros_) {
        biases_[filtroIdc] = bias;
    }
}

void testeCC() {
    std::cout << "\n=== TESTE CAMADA CONVOLUCIONAL ===\n\n";
    
    std::cout << "🔍 TESTE 1: Detecção de Bordas\n";
    {
        CamadaConv conv(5, 5, 1, 1, 3, 1, 1, 0.0f);
        
        // filtro de detecção de bordas horizontal manual
        std::vector<std::vector<float>> filtroBorda = {
            {-1, -1, -1},
            {0, 0, 0},
            {1, 1, 1}
        };
        conv.defFiltro(0, 0, filtroBorda);
        conv.defBias(0, 0.0f);
        
        // imagem de teste(borda horizontal no meio)
        std::vector<std::vector<std::vector<float>>> imagem = {{
            {0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0},
            {1, 1, 1, 1, 1},
            {0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0}
        }};
        auto saida = conv.propagar(imagem, false);
        
        std::cout << "Entrada (borda horizontal):\n";
        for(const auto& linha : imagem[0]) {
            for(float val : linha) std::cout << val << " ";
            std::cout << "\n";
        }
        std::cout << "Saída (detecção de bordas):\n";
        for(const auto& linha : saida[0]) {
            for(float val : linha) std::cout << val << " ";
            std::cout << "\n";
        }
        // verifica se detectou a borda
        float maxVal = 0.0f;
        for(const auto& linha : saida[0]) {
            for(float val : linha) maxVal = std::max(maxVal, val);
        }
        if(maxVal > 2.0f) {
            std::cout << "✅ DETECÇÃO DE BORDAS FUNCIONANDO\n";
        } else {
            std::cout << "❌ FALHA NA DETECÇÃO\n";
        }
    }
    std::cout << "\n🎯 TESTE 2: Aprendizado de Padrões Convolucionais\n";
    {
        const int epocas = 300;
        const float taxaAprendizado = 0.01f;
        
        CamadaConv conv(3, 3, 1, 1, 2, 1, 0, 0.0f);
        
        std::vector<std::vector<std::vector<float>>> entrada = {{
            {1, 0, 0},
            {0, 1, 0}, 
            {0, 0, 1}
        }};
        auto saidaEsperada = zeros3D(1, 2, 2);
        saidaEsperada[0][0][0] = 0.8f;  // canto superior esquerdo
        saidaEsperada[0][1][1] = 0.8f;  // canto inferior direito
        
        std::cout << "Entrada (diagonal):\n";
        for(const auto& linha : entrada[0]) {
            for(float val : linha) std::cout << val << " ";
            std::cout << "\n";
        }
        std::cout << "Saída esperada (cantos):\n";
        for(const auto& linha : saidaEsperada[0]) {
            for(float val : linha) std::cout << val << " ";
            std::cout << "\n";
        }
        std::vector<float> historicoErro;
        
        for(int epoca = 0; epoca < epocas; ++epoca) {
            auto saida = conv.propagar(entrada, true);
            
            float erro = 0.0f;
            auto gradSaida = zeros3D(1, 2, 2);
            
            float diff1 = saida[0][0][0] - saidaEsperada[0][0][0];
            float diff2 = saida[0][1][1] - saidaEsperada[0][1][1];
            
            erro = 0.5f * (diff1 * diff1 + diff2 * diff2);
            gradSaida[0][0][0] = diff1;
            gradSaida[0][1][1] = diff2;
            
            historicoErro.push_back(erro);
            
            if(epoca % 50 == 0) {
                std::cout << "Época " << epoca << " - Erro: " << erro;
                if(erro < 0.1f) std::cout << " ✅";
                std::cout << "\n";
            }
            conv.retropropagar(gradSaida, taxaAprendizado, 0.001f);
        }
        auto saidaFinal = conv.propagar(entrada, false);
        std::cout << "\nSaída final:\n";
        for(const auto& linha : saidaFinal[0]) {
            for(float val : linha) std::cout << val << " ";
            std::cout << "\n";
        }
        float erroFinal = 0.5f * (
            pow(saidaFinal[0][0][0] - saidaEsperada[0][0][0], 2) +
            pow(saidaFinal[0][1][1] - saidaEsperada[0][1][1], 2)
        );
        std::cout << "Erro final: " << erroFinal << " | ";
        if(erroFinal < 0.05f) {
            std::cout << "✅ APRENDIZADO CONVOLUCIONAL BEM-SUCEDIDO\n";
        } else if(erroFinal < 0.1f) {
            std::cout << "⚠️  APRENDIZADO PARCIAL\n";
        } else {
            std::cout << "❌ FALHA NO APRENDIZADO\n";
        }
    }
    std::cout << "\n⚡ TESTE 3: Performance com Dimensões Reais\n";
    {
        auto inicio = std::chrono::high_resolution_clock::now();
        
        CamadaConv conv(32, 32, 3, 16, 3, 1, 1, 0.2f);
        
        const int loteTam = 8;
        std::vector<std::vector<std::vector<std::vector<float>>>> batch(
            loteTam, 
            std::vector<std::vector<std::vector<float>>>(
                3,
                std::vector<std::vector<float>>(
                    32,
                    std::vector<float>(32, 1.0f))));
        
        std::vector<std::vector<std::vector<std::vector<float>>>> saidas;
        for(int i = 0; i < loteTam; ++i) {
            saidas.push_back(conv.propagar(batch[i], true));
        }
        for(int i = 0; i < loteTam; ++i) {
            auto grad = saidas[i];
            for(auto& canal : grad) {
                for(auto& linha : canal) {
                    for(auto& val : linha) {
                        val = (val - 0.5f) / loteTam;
                    }
                }
            }
            conv.retropropagar(grad, 0.001f);
        }
        auto fim = std::chrono::high_resolution_clock::now();
        auto duracao = std::chrono::duration_cast<std::chrono::milliseconds>(fim - inicio);
        
        std::cout << "Processamento de " << loteTam << " imagens 32x32x3 → 16 filtros\n";
        std::cout << "Tempo total: " << duracao.count() << "ms\n";
        
        double operacoes = loteTam * 16 * 32 * 32 * 3 * 3 * 3 * 2;
        double segundos = duracao.count() / 1000.0;
        
        std::cout << "Performance: " << operacoes / segundos / 1e6 << " MFLOPS\n";
        
        if(duracao.count() < 1000) {
            std::cout << "✅ PERFORMANCE ADEQUADA\n";
        } else {
            std::cout << "⚠️  PERFORMANCE A MELHORAR\n";
        }
    }
    std::cout << "\n📐 TESTE 4: Verificação de Dimensões\n";
    {
        std::vector<std::tuple<int, int, int, int, int, int, int>> testes = {
            {28, 28, 1, 8, 3, 1, 0},   // MNIST-like
            {32, 32, 3, 16, 5, 1, 2},  // CIFAR-like  
            {64, 64, 3, 32, 7, 2, 1}   // imagem maior
        };
        for(const auto& teste : testes) {
            auto [h, w, c, f, k, s, p] = teste;
            
            CamadaConv conv(h, w, c, f, k, s, p);
            auto saida = conv.propagar(zeros3D(c, h, w), false);
            
            int saidaH = (h + 2*p - k) / s + 1;
            int saidaW = (w + 2*p - k) / s + 1;
            
            bool dimensoesCorretas = (saida.size() == f) &&  (saida[0].size() == saidaH) && (saida[0][0].size() == saidaW);
            
            std::cout << "Entrada " << h << "x" << w << "x" << c << " → Saída " << saidaH << "x" << saidaW << "x" << f<< " - " << (dimensoesCorretas ? "✅" : "❌") << "\n";
        }
    }
}
// ATENÇÃO:
class CamadaAtencao {
public:
    CamadaAtencao(int dimEntrada, int dimConC, int dimV, float temperaturaInicial = 0.45f, float lambdaFocoInicial = 0.00005f); 

    std::vector<std::vector<float>> propagar(const std::vector<std::vector<float>>& entrada, bool treino = true);
    
    std::vector<std::vector<float>> retropropagar(const std::vector<std::vector<float>>& gradSaida, float taxaAprendizado, float lambda = 1e-4f);
    
    void setTemperatura(float temp) { temperatura_ = temp; }
    void setLambdaFoco(float foco) { lambdaFoco_ = foco; }
    
    int dimEntrada_, dimQC_, dimV_;
    
    float escala_; 
    float temperatura_;
    float lambdaFoco_;
    
    // pesos
    std::vector<std::vector<float>> pCon_, pC_, pV_;

    // buffers adam
    std::vector<std::vector<float>> mCon_, vCon_;
    std::vector<std::vector<float>> mC_, vC_;
    std::vector<std::vector<float>> mV_, vV_;
    int iteracao_;

    // cache para retroprop
    std::vector<std::vector<float>> cacheEntrada_;
    std::vector<std::vector<float>> cacheCon_, cacheC_, cacheV_;
    std::vector<std::vector<float>> cachePesosAtencao_; // pesos de atenção após o softmax
};

inline CamadaAtencao::CamadaAtencao(int dimEntrada, int dimConC, int dimV, float temperaturaInicial, float lambdaFocoInicial)
    : dimEntrada_(dimEntrada), dimQC_(dimConC), dimV_(dimV), iteracao_(1), temperatura_(temperaturaInicial), lambdaFoco_(lambdaFocoInicial) {
    
    pCon_ = iniPesosXavier(dimEntrada, dimConC);
    pC_ = iniPesosXavier(dimEntrada, dimConC);
    pV_ = iniPesosXavier(dimEntrada, dimV);

    // fator de escala para a atenção
    escala_ = sqrt(static_cast<float>(dimConC));
    if(escala_ < 1.0f) escala_ = 1.0f; 

    // buffers adam
    mCon_ = matrizZeros(dimEntrada, dimConC); vCon_ = matrizZeros(dimEntrada, dimConC);
    mC_ = matrizZeros(dimEntrada, dimConC); vC_ = matrizZeros(dimEntrada, dimConC);
    mV_ = matrizZeros(dimEntrada, dimV);  vV_ = matrizZeros(dimEntrada, dimV);
}

inline std::vector<std::vector<float>> CamadaAtencao::propagar(const std::vector<std::vector<float>>& entrada, bool treino) {
    auto CON = multMatrizes(entrada, pCon_);
    auto C = multMatrizes(entrada, pC_);
    auto V = multMatrizes(entrada, pV_);

    auto pontos = multMatrizes(CON, transpor(C));

    // fatorEscalaComTemp = sqrt(d_k) * temperatura
    float fatorEscalaComTemp = escala_ * temperatura_;
    auto pontosEscalados = multMatriz(pontos, 1.0f / fatorEscalaComTemp);

    // softmax pra obter os pesos de atenção(mais nítidos devido a temp < 1)
    auto pesosAtencao = softmaxLote(pontosEscalados);

    auto saida = multMatrizes(pesosAtencao, V);

    if(treino) {
        cacheEntrada_ = entrada;
        cacheCon_ = CON;
        cacheC_ = C;
        cacheV_ = V;
        cachePesosAtencao_ = pesosAtencao;
    }
    return saida;
}

inline std::vector<std::vector<float>> CamadaAtencao::retropropagar(const std::vector<std::vector<float>>& gradSaida, float taxaAprendizado, float lambda) {
    auto gradV = multMatrizes(transpor(cachePesosAtencao_), gradSaida);
    auto gradPesosAtencao = multMatrizes(gradSaida, transpor(cacheV_));
    
    std::vector<std::vector<float>> gradpontos(gradPesosAtencao.size(), std::vector<float>(gradPesosAtencao[0].size()));
    for(size_t i = 0; i < gradPesosAtencao.size(); ++i) {
        gradpontos[i] = derivadaSoftmax(cachePesosAtencao_[i], gradPesosAtencao[i]);
    }
    // injeta o gradiente de penalidade RF(Regulamentação de Foco)
    // O gradiente da penalidade é adicionado ao gradiente de erro principal
    if(lambdaFoco_ > 0.0f) {
        for(size_t i = 0; i < gradpontos.size(); ++i) {
            for(size_t j = 0; j < gradpontos[0].size(); ++j) {
                // penaliza a uniformidade(força o foco)
                gradpontos[i][j] += (2.0f * lambdaFoco_ * cachePesosAtencao_[i][j]);
            }
        }
    }
    // desfaz a escala dos pontos(com a Temperatura)
    float fatorEscalaComTemp = escala_ * temperatura_;
    gradpontos = multMatriz(gradpontos, 1.0f / fatorEscalaComTemp);

    // calcula gradientes em relação a Q e K
    auto gradK = multMatrizes(transpor(gradpontos), cacheCon_);
    auto gradQ = multMatrizes(gradpontos, cacheC_);

    // calcula gradientes pra as matrizes de pesos pQ, pK, pV
    auto gradPQ = multMatrizes(transpor(cacheEntrada_), gradQ);
    auto gradPK = multMatrizes(transpor(cacheEntrada_), gradK);
    auto gradPV = multMatrizes(transpor(cacheEntrada_), gradV);
    
    // atualiza os pesos usando adam
    pCon_ = attPesosAdam(pCon_, gradPQ, mCon_, vCon_, taxaAprendizado, 0.9f, 0.999f, 1e-8f, iteracao_, lambda);
    pC_ = attPesosAdam(pC_, gradPK, mC_, vC_, taxaAprendizado, 0.9f, 0.999f, 1e-8f, iteracao_, lambda);
    pV_ = attPesosAdam(pV_, gradPV, mV_, vV_, taxaAprendizado, 0.9f, 0.999f, 1e-8f, iteracao_, lambda);
    iteracao_++;

    // gradiente a ser propagado pra a camada anterior
    auto gradEntrada = somarMatriz(
        somarMatriz(
            multMatrizes(gradQ, transpor(pCon_)),
            multMatrizes(gradK, transpor(pC_))),
        multMatrizes(gradV, transpor(pV_)));
    
    return gradEntrada;
}

void testeCA() {
    std::cout << "\n\n=== TESTE CAMADA ATENÇÃO ===";

    const int tamSequencia = 4;
    const int dimEntrada = 32;
    const int dimConC = 16;
    const int dimV = 32;
    const int epocas = 1000;
    const float taxaAprendizado = 0.001f;

    CamadaAtencao camada(dimEntrada, dimConC, dimV);

    std::cout << "\n--- 📐 Verificação de Dimensões ---\n";
    auto entradaTeste = matriz(tamSequencia, dimEntrada, 0.1f);
    auto saidaTeste = camada.propagar(entradaTeste, false);

    bool dimensoesCorretas = (saidaTeste.size() == tamSequencia) && (saidaTeste[0].size() == dimV);
    std::cout << "Status: " << (dimensoesCorretas ? "✅ SUCESSO" : "❌ FALHA") << "\n";

    std::cout << "\n--- 🎯 Treino com Dados Consistentes ---\n";
    std::cout << "Objetivo: 1º elemento → 3º elemento (dados FIXOS para aprendizado)\n\n";

    auto entradaBase = matriz(tamSequencia, dimEntrada, 0.5f);
    for(int j = 0; j < dimEntrada; j++) {
        entradaBase[2][j] = entradaBase[0][j] + 0.3f; // padrão claro
    }
    std::vector<float> historicoErro;
    float melhorErro = 1.0f;

    for(int i = 0; i < epocas; ++i) {
        auto entradaTreino = entradaBase;
        for(int k = 0; k < tamSequencia; k++) {
            entradaTreino[k] = addRuido(entradaTreino[k], 0.02f);
        }
        auto saidaEsperada = entradaTreino[2]; // alvo: elemento 2
        auto saida = camada.propagar(entradaTreino, true);
        
        float erroEpoca = mse(saida[0], saidaEsperada);
        historicoErro.push_back(erroEpoca);
        melhorErro = std::min(melhorErro, erroEpoca);

        auto gradSaida = matrizZeros(tamSequencia, dimV);
        for(int j = 0; j < dimV; ++j) {
            float diff = saida[0][j] - saidaEsperada[j];
            gradSaida[0][j] = 2.0f * diff / dimV;
        }
        camada.retropropagar(gradSaida, taxaAprendizado, 1e-5f);

        if((i + 1) % 100 == 0) {
            std::cout << "Época [" << std::setw(3) << i + 1 << "] - Erro: " 
                      << std::fixed << std::setprecision(4) << erroEpoca;
            if(erroEpoca < 0.1f) std::cout << " ✅";
            std::cout << "\n";
        }
    }
    std::cout << "\n--- 📊 Resultados Finais ---\n";
    std::cout << "Melhor erro alcançado: " << std::fixed << std::setprecision(4) << melhorErro << "\n";
    
    bool sucesso = melhorErro < 0.1f;
    std::cout << "Status: " << (sucesso ? "✅ APRENDIZADO BEM-SUCEDIDO" : "⚠️  APRENDIZADO PARCIAL") << "\n";

    std::cout << "\n--- 🔍 Análise dos Pesos de Atenção ---\n";
    auto entradaFinal = entradaBase;
    camada.propagar(entradaFinal, false);
    auto pesosFinais = camada.cachePesosAtencao_;

    std::cout << "Distribuição de atenção do 1º elemento:\n";
    int indiceMaiorPeso = -1;
    float maiorPeso = -1.0f;
    
    for(int j = 0; j < tamSequencia; ++j) {
        float peso = pesosFinais[0][j];
        std::cout << "  Elemento " << j + 1 << ": " << std::fixed << std::setprecision(3) 
                  << peso << " (" << std::setprecision(1) << peso * 100.0f << "%)";
        if(j == 2) std::cout << " ← ALVO";
        std::cout << "\n";
        
        if(peso > maiorPeso) {
            maiorPeso = peso;
            indiceMaiorPeso = j;
        }
    }
    bool focoCorreto = (indiceMaiorPeso == 2);
    std::cout << "\n🎯 Foco principal: Elemento " << indiceMaiorPeso + 1 
              << " - " << (focoCorreto ? "✅ CORRETO!" : "❌ INCORRETO!") << "\n";

    std::cout << "\n--- 📈 Resumo do Desempenho ---\n";
    if(sucesso && focoCorreto) {
        std::cout << "🎉 EXCELENTE: Camada de atenção funcionando perfeitamente!\n";
    } else if(sucesso) {
        std::cout << "⚠️  BOM: Aprendizado ocorreu mas o foco não está ideal\n";
    } else {
        std::cout << "🔧 AJUSTES NECESSÁRIOS: Considere aumentar dimensões ou épocas\n";
    }
}