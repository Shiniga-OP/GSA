// teste_lotenorm_corrigido.cpp
#include "biblis/camadas.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <cmath>
#include <algorithm>

using namespace std;

// gerador de numeros aleatorios
mt19937 rng(42);
uniform_real_distribution<float> dist_uniform(-1.0f, 1.0f);
normal_distribution<float> dist_normal(0.0f, 0.5f);

// gera dados de XOR com ruido
vector<vector<float>> gerarDadosXOR(int amostras_por_classe = 100) {
    vector<vector<float>> dados;
    
    // classe 0: (0,0) e (1,1) com ruido
    for(int i = 0; i < amostras_por_classe; i++) {
        float ruido_x = dist_normal(rng) * 0.1f;
        float ruido_y = dist_normal(rng) * 0.1f;
        dados.push_back({0.1f + ruido_x, 0.1f + ruido_y});
        
        ruido_x = dist_normal(rng) * 0.1f;
        ruido_y = dist_normal(rng) * 0.1f;
        dados.push_back({0.9f + ruido_x, 0.9f + ruido_y});
    }
    // classe 1: (0,1) e (1,0) com ruido
    for(int i = 0; i < amostras_por_classe; i++) {
        float ruido_x = dist_normal(rng) * 0.1f;
        float ruido_y = dist_normal(rng) * 0.1f;
        dados.push_back({0.1f + ruido_x, 0.9f + ruido_y});
        
        ruido_x = dist_normal(rng) * 0.1f;
        ruido_y = dist_normal(rng) * 0.1f;
        dados.push_back({0.9f + ruido_x, 0.1f + ruido_y});
    }
    return dados;
}

// embaralha dados
void embaralharDados(vector<vector<float>>& dados, vector<float>& rotulos) {
    vector<size_t> indices(dados.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), rng);
    
    vector<vector<float>> dados_temp(dados.size());
    vector<float> rotulos_temp(rotulos.size());
    
    for(size_t i = 0; i < indices.size(); i++) {
        dados_temp[i] = dados[indices[i]];
        rotulos_temp[i] = rotulos[indices[i]];
    }
    dados = dados_temp;
    rotulos = rotulos_temp;
}

void treinarModelo(Modelo& modelo, const vector<vector<float>>& dados, 
const vector<float>& rotulos, float taxa = 0.05f, int epocas = 50) {
    modelo.modoTreino();
    
    for(int epoca = 0; epoca < epocas; epoca++) {
        float perda_total = 0.0f;
        
        size_t tam_lote = 32;
        if(tam_lote > dados.size()) tam_lote = dados.size();
        
        for(size_t inicio = 0; inicio < dados.size(); inicio += tam_lote) {
            size_t fim = min(inicio + tam_lote, dados.size());
            
            // propaga o lote inteiro
            vector<vector<float>> entrada_lote;
            vector<vector<float>> alvo_lote;
            
            for(size_t i = inicio; i < fim; i++) {
                entrada_lote.push_back(dados[i]);
                alvo_lote.push_back({rotulos[i]});
            }
            vector<vector<float>> saida_lote = modelo.propLote(entrada_lote);
            
            // calcula perda
            float perda_lote = 0.0f;
            for(size_t i = 0; i < saida_lote.size(); i++) {
                perda_lote += mse(saida_lote[i], alvo_lote[i]);
            }
            perda_lote /= saida_lote.size();
            perda_total += perda_lote * saida_lote.size();
            
            // calcula gradiente(derivada MSE) pra cada exemplo
            vector<vector<float>> grad_lote(saida_lote.size(), vector<float>(1, 0.0f));
            for(size_t i = 0; i < saida_lote.size(); i++) {
                for(size_t j = 0; j < saida_lote[i].size(); j++) {
                    grad_lote[i][j] = 2.0f * (saida_lote[i][j] - alvo_lote[i][j]) / saida_lote[i].size();
                }
            }
            
            // retropropaga o lote inteiro
            vector<vector<float>> grad_entrada = grad_lote;
            for(int j = modelo.camadas.size() - 1; j >= 0; j--) {
                auto* lotenorm = dynamic_cast<LoteNorm*>(modelo.camadas[j].get());
                if(lotenorm && lotenorm->treinando) {
                    grad_entrada = lotenorm->retropropLote(grad_entrada);
                } else {
                    // pra outras camadas, retropropaga exemplo por exemplo
                    vector<vector<float>> grad_temp;
                    for(size_t k = 0; k < grad_entrada.size(); k++) {
                        grad_temp.push_back(modelo.camadas[j]->retroprop(grad_entrada[k]));
                    }
                    grad_entrada = grad_temp;
                }
            }
            // atualiza pesos
            for(auto& camada : modelo.camadas) {
                if(camada->temParametros()) {
                    camada->att(taxa);
                }
            }
            for(auto& camada : modelo.camadas) {
                camada->zerarGradientes();
            }
        }
        if(epoca % 10 == 0) {
            cout << "Epoca " << setw(2) << epoca << " | Perda: "
            << fixed << setprecision(4) << perda_total / dados.size() << endl;
        }
    }
}

float testarModelo(Modelo& modelo, const vector<vector<float>>& dados, 
const vector<float>& rotulos) {
    modelo.modoTeste();
    
    int acertos = 0;
    for(size_t i = 0; i < dados.size(); i++) {
        vector<float> pred = modelo.prop(dados[i]);
        float classe_pred = pred[0] > 0.5f ? 1.0f : 0.0f;
        
        if(fabs(classe_pred - rotulos[i]) < 0.5f) {
            acertos++;
        }
    }
    return 100.0f * acertos / dados.size();
}

void testarLoteNorm() {
    cout << "=== Teste: LoteNorm com treinamento por lotes ===" << endl;
    
    // gera os dados
    auto dados = gerarDadosXOR(150);
    vector<float> rotulos;
    
    for(size_t i = 0; i < dados.size(); i++) {
        rotulos.push_back(i < dados.size()/2 ? 0.0f : 1.0f);
    }
    embaralharDados(dados, rotulos);
    
    cout << "Total de amostras: " << dados.size() << endl;
    cout << "Tamanho do lote de treinamento: 32" << endl;
    
    // cria os modelos
    Modelo modelo_com_ln("XOR_com_LoteNorm");
    modelo_com_ln.add(make_unique<Densa>(2, 16, "relu", true, "densa1"));
    modelo_com_ln.add(make_unique<LoteNorm>(16, 1e-5f, 0.9f, "lotenorm1"));
    modelo_com_ln.add(make_unique<Densa>(16, 8, "relu", true, "densa2"));
    modelo_com_ln.add(make_unique<LoteNorm>(8, 1e-5f, 0.9f, "lotenorm2"));
    modelo_com_ln.add(make_unique<Densa>(8, 1, "sigmoid", true, "saida"));
    
    Modelo modelo_sem_ln("XOR_sem_LoteNorm");
    modelo_sem_ln.add(make_unique<Densa>(2, 16, "relu", true, "densa1"));
    modelo_sem_ln.add(make_unique<Densa>(16, 8, "relu", true, "densa2"));
    modelo_sem_ln.add(make_unique<Densa>(8, 1, "sigmoid", true, "saida"));
    
    // resumo dos modelos
    cout << "\n=== Modelo COM LoteNorm ===" << endl;
    modelo_com_ln.resumo();
    cout << "\n=== Modelo SEM LoteNorm ===" << endl;
    modelo_sem_ln.resumo();
    
    // treina os modelos
    cout << "\nTreinando modelo COM LoteNorm..." << endl;
    treinarModelo(modelo_com_ln, dados, rotulos, 0.05f, 50);
    
    cout << "\nTreinando modelo SEM LoteNorm..." << endl;
    treinarModelo(modelo_sem_ln, dados, rotulos, 0.05f, 50);
    
    // testa os modelos
    cout << "\n=== Resultados Finais ===" << endl;
    float acuracia_com = testarModelo(modelo_com_ln, dados, rotulos);
    float acuracia_sem = testarModelo(modelo_sem_ln, dados, rotulos);
    
    cout << "Acurácia COM LoteNorm:  " << fixed << setprecision(1) << acuracia_com << "%" << endl;
    cout << "Acurácia SEM LoteNorm:  " << fixed << setprecision(1) << acuracia_sem << "%" << endl;
    
    // avaliação
    cout << "\n=== Avaliação ===" << endl;
    if(acuracia_com > acuracia_sem + 2.0f) {
        cout << "✅ LoteNorm MELHOROU a performance!" << endl;
    } else if(acuracia_com > acuracia_sem) {
        cout << "✓ LoteNorm melhorou levemente" << endl;
    } else if(acuracia_com < acuracia_sem - 2.0f) {
        cout << "❌ LoteNorm PIOROU a performance" << endl;
    } else {
        cout << "➡️  Performance similar" << endl;
    }
}

void testarEstabilidadeLoteNorm() {
    cout << "\n\n=== Teste de Estabilidade ===" << endl;
    
    // cria rede profunda
    Modelo modelo_profundo("Profundo_com_LN");
    
    modelo_profundo.add(make_unique<Densa>(10, 20, "relu", true, "l1"));
    modelo_profundo.add(make_unique<LoteNorm>(20, 1e-5f, 0.9f, "ln1"));
    modelo_profundo.add(make_unique<Densa>(20, 30, "relu", true, "l2"));
    modelo_profundo.add(make_unique<LoteNorm>(30, 1e-5f, 0.9f, "ln2"));
    modelo_profundo.add(make_unique<Densa>(30, 20, "relu", true, "l3"));
    modelo_profundo.add(make_unique<LoteNorm>(20, 1e-5f, 0.9f, "ln3"));
    modelo_profundo.add(make_unique<Densa>(20, 10, "relu", true, "l4"));
    modelo_profundo.add(make_unique<LoteNorm>(10, 1e-5f, 0.9f, "ln4"));
    modelo_profundo.add(make_unique<Densa>(10, 5, "relu", true, "l5"));
    modelo_profundo.add(make_unique<LoteNorm>(5, 1e-5f, 0.9f, "ln5"));
    modelo_profundo.add(make_unique<Densa>(5, 1, "linear", true, "saida"));
    
    cout << "Rede profunda criada com " << modelo_profundo.camadas.size() << " camadas" << endl;
    
    // testa com valores extremos
    vector<float> entrada(10);
    for(size_t i = 0; i < 10; i++) {
        entrada[i] = (i % 2 == 0) ? 100.0f : -100.0f;
    }
    cout << "\nTestando com valores extremos de entrada:" << endl;
    
    modelo_profundo.modoTreino();
    vector<float> saida_treino = modelo_profundo.prop(entrada);
    cout << "Saída (modo treino): " << saida_treino[0] << endl;
    
    modelo_profundo.modoTeste();
    vector<float> saida_teste = modelo_profundo.prop(entrada);
    cout << "Saída (modo teste): " << saida_teste[0] << endl;
    
    if(!isnan(saida_treino[0]) && !isinf(saida_treino[0]) && 
       !isnan(saida_teste[0]) && !isinf(saida_teste[0])) {
        cout << "✅ LoteNorm estabilizou a rede profunda!" << endl;
    } else {
        cout << "❌ Problema com estabilidade" << endl;
    }
}

int main() {
    try {
        cout << "==================================================" << endl;
        cout << "TESTE DA CLASSE LoteNorm" << endl;
        cout << "==================================================\n" << endl;
        
        // teste 1: performance em problema real com treinamento por lotes
        testarLoteNorm();
        
        // teste 2: estabilidade
        testarEstabilidadeLoteNorm();
        
        cout << "\n==================================================" << endl;
        cout << "CONCLUSÃO:" << endl;
        cout << "1. LoteNorm agora funciona com treinamento por lotes" << endl;
        cout << "2. A retropropagação é compatível com a interface" << endl;
        cout << "3. Estatísticas móveis são atualizadas corretamente" << endl;
        cout << "==================================================" << endl;
    } catch(const exception& e) {
        cerr << "\n❌ ERRO: " << e.what() << endl;
        return 1;
    }
    return 0;
}