#include <iostream>
#include <vector>
#include <math.h>
#include <iomanip>
#include "biblis/util.h" // Assumindo que suas funções de matrizes e Adam estão aqui

// Camada de Auto-Atenção para Produção
class CamadaAtencao {
public:
    int dim_modelo;
    float escala;
    int iteracao;

    std::vector<std::vector<float>> w_q, w_k, w_v, w_o;
    std::vector<std::vector<float>> m_q, v_q, m_k, v_k, m_v, v_v, m_o, v_o;
    std::vector<std::vector<float>> entrada_cache, q_cache, k_cache, v_cache, atencao_cache;

    CamadaAtencao(int dim) : dim_modelo(dim), iteracao(1) {
        w_q = iniPesosXavier(dim, dim);
        w_k = iniPesosXavier(dim, dim);
        w_v = iniPesosXavier(dim, dim);
        w_o = iniPesosXavier(dim, dim);

        m_q = v_q = m_k = v_k = m_v = v_v = m_o = v_o = matrizZeros(dim, dim);
        escala = 1.0f / sqrt((float)dim);
    }

    std::vector<std::vector<float>> propagar(const std::vector<std::vector<float>>& entrada) {
        entrada_cache = entrada;
        q_cache = multMatrizes(entrada, w_q);
        k_cache = multMatrizes(entrada, w_k);
        v_cache = multMatrizes(entrada, w_v);

        auto k_t = transpor(k_cache);
        auto scores = multMatrizes(q_cache, k_t);

        for (auto& linha : scores) {
            for (float& val : linha) val *= escala;
        }

        atencao_cache = softmaxLote(scores);
        auto contexto = multMatrizes(atencao_cache, v_cache);
        return multMatrizes(contexto, w_o);
    }

    std::vector<std::vector<float>> retropropagar(const std::vector<std::vector<float>>& grad_saida, float taxa) {
        auto contexto = multMatrizes(atencao_cache, v_cache);
        auto g_wo = multMatrizes(transpor(contexto), grad_saida);
        auto g_contexto = multMatrizes(grad_saida, transpor(w_o));

        auto g_atencao_pos_softmax = multMatrizes(g_contexto, transpor(v_cache));
        auto g_v = multMatrizes(transpor(atencao_cache), g_contexto);

        std::vector<std::vector<float>> g_scores(atencao_cache.size());
        for (size_t i = 0; i < atencao_cache.size(); ++i) {
            g_scores[i] = derivadaSoftmax(atencao_cache[i], g_atencao_pos_softmax[i]);
            for (float& val : g_scores[i]) val *= escala;
        }

        auto g_q = multMatrizes(g_scores, k_cache);
        auto g_k = multMatrizes(transpor(g_scores), q_cache);

        auto g_wq = multMatrizes(transpor(entrada_cache), g_q);
        auto g_wk = multMatrizes(transpor(entrada_cache), g_k);
        auto g_wv = multMatrizes(transpor(entrada_cache), g_v);

        w_q = attPesosAdam(w_q, g_wq, m_q, v_q, taxa, 0.9f, 0.999f, 1e-8f, iteracao);
        w_k = attPesosAdam(w_k, g_wk, m_k, v_k, taxa, 0.9f, 0.999f, 1e-8f, iteracao);
        w_v = attPesosAdam(w_v, g_wv, m_v, v_v, taxa, 0.9f, 0.999f, 1e-8f, iteracao);
        w_o = attPesosAdam(w_o, g_wo, m_o, v_o, taxa, 0.9f, 0.999f, 1e-8f, iteracao);
        
        iteracao++;
        
        auto g_x = multMatrizes(g_q, transpor(w_q));
        auto g_xk = multMatrizes(g_k, transpor(w_k));
        auto g_xv = multMatrizes(g_v, transpor(w_v));

        for (size_t i = 0; i < g_x.size(); ++i) {
            for (size_t j = 0; j < g_x[0].size(); ++j) g_x[i][j] += g_xk[i][j] + g_xv[i][j];
        }
        return g_x;
    }
};

// --- FUNÇÕES DE TESTE E DEBUG VISUAL ---

void exibirProgresso(int epoca, float erro, const std::vector<std::vector<float>>& atencao) {
    if (epoca % 100 == 0) {
        std::cout << "Época: " << std::setw(4) << epoca << " | Erro: " << std::fixed << std::setprecision(6) << erro;
        // Mostra o valor máximo de atenção do primeiro token para verificar foco
        float max_at = 0;
        for(float v : atencao[0]) if(v > max_at) max_at = v;
        std::cout << " | Força do Foco: " << max_at * 100 << "%" << std::endl;
    }
}

void testarAprendizadoRelacional() {
    std::cout << "\n=== TESTE DE VALIDAÇÃO: ATENÇÃO CONTEXTUAL ===\n";
    std::cout << "Objetivo: O Token 0 deve aprender a buscar informação no Token 2\n";
    
    int dim = 3;
    int seq = 3;
    CamadaAtencao at(dim);
    float taxa = 0.005f;

    // Entrada: 3 tokens distintos
    // T0: [1,0,0], T1: [0,1,0], T2: [0,0,1]
    std::vector<std::vector<float>> entrada = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    
    // Alvo: Queremos que a saída do Token 0 seja igual ao vetor do Token 2 [0,0,1]
    // Isso prova que ele "olhou" para o T2 e extraiu seu valor.
    std::vector<std::vector<float>> alvo = {{0, 0, 1}, {0, 1, 0}, {1, 0, 0}};

    for (int epoca = 0; epoca <= 1000; epoca++) {
        auto saida = at.propagar(entrada);
        float erro_total = 0;
        std::vector<std::vector<float>> grad(seq, std::vector<float>(dim));

        for (int i = 0; i < seq; i++) {
            for (int j = 0; j < dim; j++) {
                float d = saida[i][j] - alvo[i][j];
                erro_total += 0.5f * d * d;
                grad[i][j] = d;
            }
        }

        at.retropropagar(grad, taxa);
        exibirProgresso(epoca, erro_total, at.atencao_cache);
    }

    // --- VERIFICAÇÃO FINAL ---
    auto final = at.propagar(entrada);
    std::cout << "\nRESULTADO FINAL (Saída do Token 0):\n";
    std::cout << "Esperado: [ 0.00  0.00  1.00 ]\n";
    std::cout << "Obtido:   [ ";
    for(float v : final[0]) std::cout << std::fixed << std::setprecision(2) << v << "  ";
    std::cout << "]\n";

    std::cout << "\nMAPA DE ATENÇÃO (Quem olha para quem):\n";
    for (int i = 0; i < seq; i++) {
        std::cout << "Token " << i << " foca em: ";
        for (int j = 0; j < seq; j++) {
            float val = at.atencao_cache[i][j];
            if (val > 0.8) std::cout << "TOKEN " << j << " (" << val*100 << "%) <--- ALVO ATINGIDO\n";
        }
    }
}

int main() {
    testarAprendizadoRelacional();
    return 0;
}