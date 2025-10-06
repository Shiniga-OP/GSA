#pragma once
#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>
#include <chrono>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include "util.h"
#include "ativas.h"

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
            ativ1[i][j] = std::tanh(z1[i][j]);
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
                float tanh_x = std::tanh(z1_i[j]);
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
    std::cout << "\n=== TESTE - APRENDIZADO IDENTIDADE ===\n\n";
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
            erroFinal += std::abs(diff);
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
    
    float escala = std::sqrt(2.0f / (filtroTamanho * filtroTamanho * entradaCanais));
    
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
                                     (1 - std::pow(0.9f, iteracao_));
                    float vCorrigido = vFiltros_[filtro][canal][fi][fj] / 
                                     (1 - std::pow(0.999f, iteracao_));
                    
                    filtros_[filtro][canal][fi][fj] -= 
                        taxaAprendizado * mCorrigido / (std::sqrt(vCorrigido) + 1e-8f);
                }
            }
        }
        float gBias = gradBiases[filtro] + lambda * biases_[filtro];
        mBiases_[filtro] = 0.9f * mBiases_[filtro] + 0.1f * gBias;
        vBiases_[filtro] = 0.999f * vBiases_[filtro] + 0.001f * gBias * gBias;
        
        float mBiasCorrigido = mBiases_[filtro] / (1 - std::pow(0.9f, iteracao_));
        float vBiasCorrigido = vBiases_[filtro] / (1 - std::pow(0.999f, iteracao_));
        
        biases_[filtro] -= taxaAprendizado * mBiasCorrigido / (std::sqrt(vBiasCorrigido) + 1e-8f);
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
    std::cout << "\n=== TESTE AVANÇADO - CAMADA CONVOLUCIONAL ===\n\n";
    // detecção de bordas
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
    // aprendizado de Padrões
    std::cout << "🎯 TESTE 2: Aprendizado de Padrões Convolucionais\n";
    {
        const int epocas = 200;
        const float taxaAprendizado = 0.02f;
        
        CamadaConv conv(4, 4, 1, 2, 2, 1, 0, 0.0f);
        
        std::vector<std::vector<std::vector<float>>> entrada = {{
            {1, 1, 0, 0},
            {1, 1, 0, 0}, 
            {0, 0, 1, 1},
            {0, 0, 1, 1}
        }};
        auto saidaEsperada = zeros3D(2, 3, 3);
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                saidaEsperada[0][i][j] = (i < 2 && j < 2) ? 1.0f : 0.0f;  // Canto superior esquerdo
                saidaEsperada[1][i][j] = (i >= 1 && j >= 1) ? 1.0f : 0.0f; // Canto inferior direito
            }
        }
        // treino
        std::vector<float> historicoErro;
        
        for(int epoca = 0; epoca < epocas; ++epoca) {
            auto saida = conv.propagar(entrada, true);
            
            // calcula erro
            float erro = 0.0f;
            auto gradSaida = zeros3D(2, 3, 3);
            
            for(int f = 0; f < 2; ++f) {
                for(int i = 0; i < 3; ++i) {
                    for(int j = 0; j < 3; ++j) {
                        float diff = saida[f][i][j] - saidaEsperada[f][i][j];
                        erro += diff * diff;
                        gradSaida[f][i][j] = 2.0f * diff;
                    }
                }
            }
            historicoErro.push_back(erro);
            
            if(epoca % 20 == 0) {
                std::cout << "Época " << epoca << " - Erro: " << erro << "\n";
            }
            conv.retropropagar(gradSaida, taxaAprendizado, 0.001f);
        }
        auto saidaFinal = conv.propagar(entrada, false);
        float erroFinal = 0.0f;
        
        for(int f = 0; f < 2; ++f) {
            for(int i = 0; i < 3; ++i) {
                for(int j = 0; j < 3; ++j) {
                    float diff = saidaFinal[f][i][j] - saidaEsperada[f][i][j];
                    erroFinal += std::abs(diff);
                }
            }
        }
        std::cout << "Erro final: " << erroFinal << " | ";
        if(erroFinal < 1.0f) {
            std::cout << "✅ APRENDIZADO CONVOLUCIONAL BEM-SUCEDIDO\n";
        } else {
            std::cout << "⚠️  APRENDIZADO PARCIAL\n";
        }
    }
    // lerformance com Imagens Grandes
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
    // verificação de Dimensões
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

class CamadaAtencao {
public:
    CamadaAtencao(int dimEntrada, int dimConC, int dimV);

    std::vector<std::vector<float>> propagar(const std::vector<std::vector<float>>& entrada, bool treino = true);
    
    std::vector<std::vector<float>> retropropagar(const std::vector<std::vector<float>>& gradSaida, float taxaAprendizado, float lambda = 1e-4f);
    int dimEntrada_; // dimensão dos vetores de cada elemento da sequência de entrada
    int dimQC_; // dimensão interna dos vetores de Consulta (Con) e Chave (C)
    int dimV_; // dimensão interna dos vetores de Valor (V)
    float escala_; // fator de escala(raiz quadrada de dimQC_)
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
    std::vector<std::vector<float>> cachePesosAtencao_;
};

inline CamadaAtencao::CamadaAtencao(int dimEntrada, int dimConC, int dimV)
    : dimEntrada_(dimEntrada), dimQC_(dimConC), dimV_(dimV), iteracao_(1) {
    // pesos usando Xavier/Glorot para manter a variância do sinal
    pCon_ = iniPesosXavier(dimEntrada, dimConC);
    pC_ = iniPesosXavier(dimEntrada, dimConC);
    pV_ = iniPesosXavier(dimEntrada, dimV);

    // fator de escala para a atenção
    escala_ = std::sqrt(static_cast<float>(dimConC));
    if (escala_ < 1.0f) escala_ = 1.0f; // Evita amplificar o sinal para dimensões pequenas

    // acumuladores do otimizador adam com zeros
    mCon_ = matrizZeros(dimEntrada, dimConC); vCon_ = matrizZeros(dimEntrada, dimConC);
    mC_ = matrizZeros(dimEntrada, dimConC); vC_ = matrizZeros(dimEntrada, dimConC);
    mV_ = matrizZeros(dimEntrada, dimV);  vV_ = matrizZeros(dimEntrada, dimV);
}

inline std::vector<std::vector<float>> CamadaAtencao::propagar(const std::vector<std::vector<float>>& entrada, bool treino) {
    // 1 projeção da entrada para obter Consulta (Q), Chave (K) e Valor (V)
    auto Q = multMatrizes(entrada, pCon_);
    auto K = multMatrizes(entrada, pC_);
    auto V = multMatrizes(entrada, pV_);

    // 2. calculo dos pontos de atenção: Q * K^T
    auto pontos = multMatrizes(Q, transpor(K));

    // 3. escala dos pontos pra estabilizar o gradiente
    auto pontosEscalados = multMatriz(pontos, 1.0f / escala_);

    // 4. aplicação do softmax pra obter os pesos de atenção
    auto pesosAtencao = softmaxLote(pontosEscalados);

    // 5. calculo da saída: produto dos pesos de atenção com os Valores (V)
    auto saida = multMatrizes(pesosAtencao, V);

    // se estiver em modo de treino, armazena os valores intermediários para a retroprop
    if(treino) {
        cacheEntrada_ = entrada;
        cacheCon_ = Q;
        cacheC_ = K;
        cacheV_ = V;
        cachePesosAtencao_ = pesosAtencao;
    }

    return saida;
}

inline std::vector<std::vector<float>> CamadaAtencao::retropropagar(const std::vector<std::vector<float>>& gradSaida, float taxaAprendizado, float lambda) {
    // passo 1: calcula gradientes em relação a V e aos pesos de atenção
    auto gradV = multMatrizes(transpor(cachePesosAtencao_), gradSaida);
    auto gradPesosAtencao = multMatrizes(gradSaida, transpor(cacheV_));

    // passo 2: retroprop o gradiente atraves da função softmax
    std::vector<std::vector<float>> gradpontos(gradPesosAtencao.size(), std::vector<float>(gradPesosAtencao[0].size()));
    for(size_t i = 0; i < gradPesosAtencao.size(); ++i) {
        gradpontos[i] = derivadaSoftmax(cachePesosAtencao_[i], gradPesosAtencao[i]);
    }
    // passo 3: desfaz a escala dos pontos
    gradpontos = multMatriz(gradpontos, 1.0f / escala_);

    // passo 4: calcula gradientes em relação a Q e K
    auto gradK = multMatrizes(transpor(gradpontos), cacheCon_);
    auto gradQ = multMatrizes(gradpontos, cacheC_);

    // passo 5: calcula gradientes pra as matrizes de pesos pQ, pK, pV
    auto gradPQ = multMatrizes(transpor(cacheEntrada_), gradQ);
    auto gradPK = multMatrizes(transpor(cacheEntrada_), gradK);
    auto gradPV = multMatrizes(transpor(cacheEntrada_), gradV);
    
    // passo 6: atualiza os pesos usando adam
    pCon_ = attPesosAdam(pCon_, gradPQ, mCon_, vCon_, taxaAprendizado, 0.9f, 0.999f, 1e-8f, iteracao_, lambda);
    pC_ = attPesosAdam(pC_, gradPK, mC_, vC_, taxaAprendizado, 0.9f, 0.999f, 1e-8f, iteracao_, lambda);
    pV_ = attPesosAdam(pV_, gradPV, mV_, vV_, taxaAprendizado, 0.9f, 0.999f, 1e-8f, iteracao_, lambda);
    iteracao_++;

    // passo 7: calcula o gradiente a ser propagado pra a camada anterior
    auto gradEntrada = somarMatriz(
        somarMatriz(
            multMatrizes(gradQ, transpor(pCon_)),
            multMatrizes(gradK, transpor(pC_))),
        multMatrizes(gradV, transpor(pV_)));
    
    return gradEntrada;
}

void testeCA() {
    std::cout << "\n\n=============================================";
    std::cout << "\n===   INICIANDO TESTE DA CAMADA ATENÇÃO   ===";
    std::cout << "\n============================================= (V2 - Teste Dinâmico)\n";

    const int tamSequencia = 4;
    const int dimEntrada = 8;
    const int dimConC = 16;
    const int dimV = 8;
    
    CamadaAtencao camada(dimEntrada, dimConC, dimV);

    std::cout << "\n--- 📐 Teste 1: Verificação de Dimensões ---\n";
    auto entradaTeste = matriz(tamSequencia, dimEntrada, 1.0f);
    auto saidaTeste = camada.propagar(entradaTeste, false);

    bool dimensoesCorretas = (saidaTeste.size() == tamSequencia) && (saidaTeste[0].size() == dimV);
    std::cout << "Dimensão da Entrada: " << tamSequencia << "x" << dimEntrada << "\n";
    std::cout << "Dimensão da Saída:   " << saidaTeste.size() << "x" << saidaTeste[0].size() << "\n";
    std::cout << "Status: " << (dimensoesCorretas ? "✅ SUCESSO" : "❌ FALHA") << "\n";
    if(!dimensoesCorretas) {
        std::cout << "TESTE INTERROMPIDO DEVIDO A DIMENSÕES INCORRETAS.\n";
        return;
    }
    std::cout << "\n--- 🎯 Teste 2: Aprendizado de Foco Dinâmico ---\n";
    std::cout << "Objetivo: Fazer o 1º elemento prestar atenção no 3º em dados ALEATÓRIOS.\n\n";

    const int epocas = 300;
    const float taxaAprendizado = 0.01f;
    float erroMedio = 0.0f;
    std::vector<float> historicoErro;

    const int indiceConsulta = 0; // elemento que vai "perguntar"
    const int indiceAlvo = 2; // elemento que deve ser "encontrado"

    for(int i = 0; i < epocas; ++i) {
        // dados novos a cada epoca
        auto entradaTreino = matriz(tamSequencia, dimEntrada, 1.0f);
        auto saidaEsperada = entradaTreino[indiceAlvo];
        auto saida = camada.propagar(entradaTreino, true);
        
        // calculo do erro(apenas pra o elemento de consulta)
        float erroEpoca = mse(saida[indiceConsulta], saidaEsperada);
        historicoErro.push_back(erroEpoca);

        // preparação do gradiente da saída
        auto gradSaida = matrizZeros(tamSequencia, dimV);
        for(int j = 0; j < dimV; ++j) {
            float diff = saida[indiceConsulta][j] - saidaEsperada[j];
            gradSaida[indiceConsulta][j] = 2.0f * diff / dimV;
        }
        // retroprop
        camada.retropropagar(gradSaida, taxaAprendizado, 1e-5f);

        if((i + 1) % 50 == 0) {
            // calcula a média do erro das últimas 50 épocas para suavizar a curva
            float somaErro = 0.0f;
            for(size_t k = historicoErro.size() - 50; k < historicoErro.size(); ++k) {
                somaErro += historicoErro[k];
            }
            erroMedio = somaErro / 50.0f;
            std::cout << "Época [" << std::setw(3) << i + 1 << "/" << epocas << "] - Erro Médio (últimas 50): " << std::fixed << std::setprecision(6) << erroMedio << "\n";
        }
    }
    bool aprendizadoSucedido = erroMedio < 0.05;
    std::cout << "\nStatus do Treinamento: " << (aprendizadoSucedido ? "✅ APRENDIZADO BEM-SUCEDIDO" : "⚠️ APRENDIZADO INCOMPLETO") << "\n";

    std::cout << "\n--- 🔍 Teste 3: Análise dos Pesos de Atenção (após treino dinâmico) ---\n";
    auto entradaFinal = matriz(tamSequencia, dimEntrada, 1.0f);
    camada.propagar(entradaFinal, false);
    auto pesosFinais = camada.cachePesosAtencao_;

    std::cout << "Pesos de atenção para o " << indiceConsulta + 1 << "º elemento (linha " << indiceConsulta << "):\n[";
    int indiceMaiorPeso = -1;
    float maiorPeso = -1.0f;
    for(int j = 0; j < tamSequencia; ++j) {
        std::cout << std::fixed << std::setprecision(4) << pesosFinais[indiceConsulta][j] << (j == tamSequencia - 1 ? "" : ", ");
        if(pesosFinais[indiceConsulta][j] > maiorPeso) {
            maiorPeso = pesosFinais[indiceConsulta][j];
            indiceMaiorPeso = j;
        }
    }
    std::cout << "]\nO " << indiceConsulta + 1 << "º elemento está focando no " << indiceMaiorPeso + 1 << "º elemento com " << std::fixed << std::setprecision(2) << maiorPeso * 100.0f << "% de atenção.\n";

    bool focoCorreto = (indiceMaiorPeso == indiceAlvo);
    std::cout << "Status do Foco: " << (focoCorreto ? "🎯 FOCO CORRETO!" : "❌ FOCO INCORRETO!") << "\n";

    std::cout << "\n=============================================";
    std::cout << "\n===      TESTE DA CAMADA ATENÇÃO FIM      ===";
    std::cout << "\n=============================================\n\n";
}