#include "biblis/camadas.h"
#include "biblis/toke.h"
#include <algorithm>
#include <memory>

class GeradorSequencias {
public:
    GeradorSequencias(int vocabTam, int dimModelo, int numCamadas, int tamCtx)
        : vocabTam_(vocabTam), dimModelo_(dimModelo), 
          numCamadas_(numCamadas), tamCtx_(tamCtx) {
        
        embedding_ = std::make_unique<CamadaDensa>(vocabTam, dimModelo, 0.1f);
        
        for(int i = 0; i < numCamadas; ++i) {
            auto densa = std::make_unique<CamadaDensa>(dimModelo, dimModelo, 0.1f);
            auto atencao = std::make_unique<CamadaAtencao>(dimModelo, dimModelo/2, dimModelo);
            
            camadasDensas_.push_back(std::move(densa));
            camadasAtencao_.push_back(std::move(atencao));
        }
        saida_ = std::make_unique<CamadaDensa>(dimModelo, vocabTam, 0.1f);
        
        std::cout << "Configuração:\n";
        std::cout << "   Dimensão do Modelo: " << dimModelo << "\n";
        std::cout << "   Camadas: " << numCamadas << "\n";
        std::cout << "   Contexto: " << tamCtx << " tokens\n";
        std::cout << "   Tamanho do Vocabulario: " << vocabTam << "\n";
    }
    
    std::vector<std::vector<float>> propagar(const std::vector<int>& tokens, bool treino = true) {
        if(tokens.empty()) return {};
        
        auto entrada = tokensPraOneHot(tokens);
        auto x = embedding_->propagar(entrada, treino);
        
        for(int i = 0; i < numCamadas_; ++i) {
            auto densa_saida = camadasDensas_[i]->propagar(x, treino);
            auto atencao_saida = camadasAtencao_[i]->propagar(densa_saida, treino);
            x = atencao_saida;
        }
        
        auto logits = saida_->propagar(x, treino);
        return softmaxLote(logits);
    }
    
    std::vector<int> gerar(const std::vector<int>& ctx, int maxTokens, float temp = 0.5f) {
        std::vector<int> resultado = ctx;
        
        std::cout << "\nGERANDO: [";
        for(size_t i = 0; i < ctx.size(); ++i) {
            std::cout << ctx[i];
            if(i < ctx.size() - 1) std::cout << " ";
        }
        std::cout << "] → ";
        
        for(int i = 0; i < maxTokens; ++i) {
            std::vector<int> ctxAtual = resultado;
            if(ctxAtual.size() > tamCtx_) {
                ctxAtual.erase(ctxAtual.begin(), ctxAtual.end() - tamCtx_);
            }
            
            auto probs = propagar(ctxAtual, false);
            int proximoToken = sampleToken(probs.back(), temp);
            
            resultado.push_back(proximoToken);
            std::cout << proximoToken << " ";
            
            if(resultado.size() >= 30) break;
        }
        return resultado;
    }
    
    void treinar(const std::vector<std::vector<int>>& dados, int epocas = 100, float taxa = 0.001f) {
        std::cout << "\nTREINANDO...\n";
        
        for(int epoca = 0; epoca < epocas; ++epoca) {
            float erroTotal = 0.0f;
            int amostras = 0;
            
            for(const auto& sequencia : dados) {
                for(size_t pos = 1; pos < sequencia.size(); ++pos) {
                    std::vector<int> ctx(sequencia.begin(), sequencia.begin() + pos);
                    int target = sequencia[pos];
                    
                    auto probs = propagar(ctx, true);
                    auto grad = calcularGrad(probs, target);
                    
                    retropropagar(grad, taxa);
                    
                    if(!probs.empty()) {
                        float prob = probs.back()[target];
                        erroTotal += -log(std::max(prob, 1e-8f));
                        amostras++;
                    }
                }
            }
            if(epoca % 10 == 0 && amostras > 0) {
                std::cout << "Época " << epoca << " - Perda: " << (erroTotal / amostras) << "\n";
            }
        }
    }

private:
    int vocabTam_, dimModelo_, numCamadas_, tamCtx_;
    std::unique_ptr<CamadaDensa> embedding_;
    std::vector<std::unique_ptr<CamadaDensa>> camadasDensas_;
    std::vector<std::unique_ptr<CamadaAtencao>> camadasAtencao_;
    std::unique_ptr<CamadaDensa> saida_;
    
    std::vector<std::vector<float>> tokensPraOneHot(const std::vector<int>& tokens) {
        std::vector<std::vector<float>> oneHot(tokens.size(), std::vector<float>(vocabTam_, 0.0f));
        for(size_t i = 0; i < tokens.size(); ++i) {
            if(tokens[i] >= 0 && tokens[i] < vocabTam_) {
                oneHot[i][tokens[i]] = 1.0f;
            }
        }
        return oneHot;
    }
    
    std::vector<std::vector<float>> calcularGrad(const std::vector<std::vector<float>>& probs, int target) {
        auto grad = probs;
        int lote = probs.size();
        
        for(int i = 0; i < lote; ++i) {
            for(int j = 0; j < vocabTam_; ++j) {
                grad[i][j] = (probs[i][j] - (j == target ? 1.0f : 0.0f)) / lote;
            }
        }
        return grad;
    }
    
    void retropropagar(const std::vector<std::vector<float>>& grad, float taxa) {
        auto grad_atual = saida_->retropropagar(grad, taxa, 1e-4f);
        
        for(int i = numCamadas_ - 1; i >= 0; --i) {
            grad_atual = camadasAtencao_[i]->retropropagar(grad_atual, taxa, 1e-4f);
            grad_atual = camadasDensas_[i]->retropropagar(grad_atual, taxa, 1e-4f);
        }
        embedding_->retropropagar(grad_atual, taxa, 1e-4f);
    }
    
    int sampleToken(const std::vector<float>& probs, float temp) {
        std::vector<float> probs_temp(probs.size());
        float soma = 0.0f;
        
        for(size_t i = 0; i < probs.size(); ++i) {
            probs_temp[i] = pow(probs[i] + 1e-8f, 1.0f / temp);
            soma += probs_temp[i];
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float amostra = dist(gen) * soma;
        
        float acumulado = 0.0f;
        for(size_t i = 0; i < probs_temp.size(); ++i) {
            acumulado += probs_temp[i];
            if(amostra <= acumulado) return i;
        }
        return 0;
    }
};

int main() {
    std::cout << "=== GERADOR DE SEQUENCIAS ===\n";
    const int vocabTam = 15;
    const int dimModelo = 16;
    const int numCamadas = 2;
    const int tamCtx = 4;

    std::vector<std::vector<int>> dados = {
        {1, 2, 3, 1, 2, 3},
        {4, 5, 6, 4, 5, 6},
        {7, 8, 9, 7, 8, 9}
    };
    try {
        std::cout << "\nIniciando gerador...\n";
        GeradorSequencias gerador(vocabTam, dimModelo, numCamadas, tamCtx);
        
        std::cout << "Treinando com " << dados.size() << " padrões...\n";
        gerador.treinar(dados, 100, 0.002f);
        
        std::cout << "\n=== TESTES DE GERAÇÃO ===\n";
        
        std::cout << "1. [1 2] esperado 3";
        auto gerado1 = gerador.gerar({1, 2}, 1);
        if(gerado1[2] == 3) std::cout << "✅ CORRETO\n";
        else std::cout << "❌ ERRADO\n";
        std::cout << "\n";
        
        std::cout << "2. [4 5] esperado 6";
        auto gerado2 = gerador.gerar({4, 5}, 1);
        if(gerado2[2] == 6) std::cout << "✅ CORRETO\n";
        else std::cout << "❌ ERRADO\n";
        std::cout << "\n";
        
        std::cout << "3. [7 8] esperado 9";
        auto gerado3 = gerador.gerar({7, 8}, 1);
        if(gerado3[2] == 9) std::cout << "✅ CORRETO\n";
        else std::cout << "❌ ERRADO\n";
        std::cout << "\n";
    } catch(const std::exception& e) {
        std::cout << "❌ ERRO: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

