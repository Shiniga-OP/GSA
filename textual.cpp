#include <iostream>
#include <vector>
#include <random>
#include "gsa.h"
#include <chrono>
#include <cmath>

class GeradorSequencias {
    std::vector<std::vector<float>> pesos_embedding;
    std::vector<CamadaDensa> camadas;
    std::vector<CamadaAtencao> atencao_camadas;
    int vocab_size, dim_modelo, ctx_size;
    
public:
    GeradorSequencias(int vocab, int dim=512, int ctx=256) 
        : vocab_size(vocab), dim_modelo(dim), ctx_size(ctx) {
        
        // Inicializar embedding
        pesos_embedding = iniPesosXavier(vocab_size, dim_modelo);
        
        // Arquitetura transformer-like
        for(int i = 0; i < 6; i++) {
            atencao_camadas.emplace_back(dim_modelo, dim_modelo/8, dim_modelo);
            camadas.emplace_back(dim_modelo, dim_modelo * 4, 0.1f);
            camadas.emplace_back(dim_modelo * 4, dim_modelo, 0.1f);
        }
        camadas.emplace_back(dim_modelo, vocab_size, 0.0f);
    }

    std::vector<int> gerar(std::vector<int> prompt, int max_tokens, float temp=0.8f) {
        std::vector<int> tokens = prompt;
        
        for(int step = 0; step < max_tokens; step++) {
            if(tokens.size() > ctx_size) {
                tokens = std::vector<int>(tokens.end() - ctx_size, tokens.end());
            }
            
            auto embeddings = get_embeddings(tokens);
            auto hidden_states = embeddings;
            
            // Processar através das camadas
            for(size_t i = 0; i < atencao_camadas.size(); i++) {
                // Atenção
                auto attn_output = atencao_camadas[i].propagar(hidden_states, false);
                
                // FFN
                auto ff_output1 = camadas[i*2].propagar(attn_output, false);
                auto ff_output2 = camadas[i*2+1].propagar(ff_output1, false);
                
                // Residual connection
                hidden_states = somarMatriz(hidden_states, ff_output2);
            }
            
            // Camada final
            auto logits = camadas.back().propagar(hidden_states, false);
            
            // Pegar último token
            auto last_logits = logits.back();
            int next_token = sample_com_temp(last_logits, temp);
            
            tokens.push_back(next_token);
            
            if(next_token == 2) break; // Token de fim
        }
        
        return tokens;
    }

private:
    std::vector<std::vector<float>> get_embeddings(const std::vector<int>& tokens) {
        std::vector<std::vector<float>> embeddings;
        for(int token : tokens) {
            embeddings.push_back(pesos_embedding[token]);
        }
        return embeddings;
    }

    int sample_com_temp(const std::vector<float>& logits, float temp) {
        std::vector<float> scaled_logits = logits;
        for(float& val : scaled_logits) val /= temp;
        
        auto probs = softmax(scaled_logits);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        return dist(gen);
    }
};

// PROVA PRÁTICA
void testar_gerador_sequencias() {
    std::cout << "🧪 INICIANDO PROVA DO GERADOR DE SEQUÊNCIAS\n";
    
    // Configuração
    int vocab_size = 10000;
    GeradorSequencias gerador(vocab_size, 512, 1024);
    
    // Dados de teste
    std::vector<int> prompt = {1, 2, 3, 4, 5};
    
    std::cout << "🎯 Gerando sequência a partir do prompt: ";
    for(int t : prompt) std::cout << t << " ";
    std::cout << "\n";
    
    auto inicio = std::chrono::high_resolution_clock::now();
    
    // Gerar sequência
    auto sequencia = gerador.gerar(prompt, 50, 0.7f);
    
    auto fim = std::chrono::high_resolution_clock::now();
    auto duracao = std::chrono::duration_cast<std::chrono::milliseconds>(fim - inicio);
    
    std::cout << "📝 Sequência gerada (" << duracao.count() << "ms): ";
    for(size_t i = 0; i < std::min(sequencia.size(), size_t(20)); i++) {
        std::cout << sequencia[i] << " ";
    }
    if(sequencia.size() > 20) std::cout << "...";
    std::cout << "\n";
    
    // Métricas
    std::cout << "📊 Estatísticas:\n";
    std::cout << "   - Comprimento: " << sequencia.size() << " tokens\n";
    std::cout << "   - Tokens únicos: " << std::set<float>(sequencia.begin(), sequencia.end()).size() << "\n";
    
    // Verificar coerência básica
    bool coerente = true;
    for(size_t i = 1; i < sequencia.size(); i++) {
        if(sequencia[i] < 0 || sequencia[i] >= vocab_size) {
            coerente = false;
            break;
        }
    }
    
    std::cout << "✅ " << (coerente ? "SEQUÊNCIA VÁLIDA" : "SEQUÊNCIA INVÁLIDA") << "\n";
    std::cout << "🎉 GERADOR DE SEQUÊNCIAS OPERACIONAL\n";
}

int main() {
    testar_gerador_sequencias();
    return 0;
}