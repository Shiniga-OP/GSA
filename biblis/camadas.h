#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>
#include <chrono>

#include "util.h"
#include "ativas.h"

class CamadaDensa {
public:
    CamadaDensa(int dimEntrada, int dimSaida, float taxaDropout = 0.0f);
    std::vector<std::vector<float>> propagar(const std::vector<std::vector<float>>& x, bool treino = true);
    std::vector<std::vector<float>> retropropagar(const std::vector<std::vector<float>>& dY, float taxa, float lambda = 0.001f);
    
    float getErro() const { return ultimoErro; }

private:
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
