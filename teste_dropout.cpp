// teste_dropout.cpp
#include "biblis/camadas.h"
#include <iostream>
#include <iomanip>
#include <random>

using namespace std;

int main() {
    cout << "=== VALIDAÇÃO MATEMÁTICA DO DROPOUT ===" << endl;
    cout << fixed << setprecision(6);
    
    // teste 1: esperança mantida
    {
        cout << "\n1. Teste de ESPERANÇA (média):" << endl;
        cout << "   Entrada constante = 1.0, taxa = 0.5" << endl;
        cout << "   Esperado: E[saída] ≈ 1.0" << endl;
        
        Dropout dropout(0.5f, "teste", 12345);
        dropout.treinando = true;
        
        const int N = 100000;
        const int DIM = 1;
        vector<float> entrada(DIM, 1.0f);
        
        double soma = 0.0;
        double soma_quadrados = 0.0;
        
        for(int i = 0; i < N; i++) {
            auto saida = dropout.prop(entrada);
            soma += saida[0];
            soma_quadrados += saida[0] * saida[0];
        }
        double media = soma / N;
        double variancia = (soma_quadrados / N) - (media * media);
        
        cout << "   Média obtida: " << media << endl;
        cout << "   Variância: " << variancia << endl;
        
        // calculo correto da variancia teorica
        float taxa = 0.5f; // taxa do dropout
        double var_teorica = taxa / (1.0f - taxa); // pra entrada=1
        
        if(abs(media - 1.0) < 0.01) cout << "   ✓ ESPERANÇA CORRETA" << endl;
        else cout << "   ✗ ERRO: Esperança incorreta" << endl;
        
        if(abs(variancia - var_teorica) < 0.1) cout << "   ✓ VARIÂNCIA CORRETA" << endl;
        else cout << "   ✗ ERRO: Variância incorreta" << endl;
    }
    // teste 2: modo teste vs treino
    {
        cout << "\n2. Teste MODO TREINO vs TESTE:" << endl;
        
        Dropout dropout(0.3f, "teste", 42);
        vector<float> entrada = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        
        // modo treino
        dropout.treinando = true;
        auto saida_treino = dropout.prop(entrada);
        
        // modo teste
        dropout.treinando = false;
        auto saida_teste = dropout.prop(entrada);
        
        cout << "   Entrada:    ";
        for(auto v : entrada) cout << setw(8) << v << " ";
        cout << "\n   Treino:     ";
        for(auto v : saida_treino) cout << setw(8) << v << " ";
        cout << "\n   Teste:      ";
        for(auto v : saida_teste) cout << setw(8) << v << " ";
        cout << endl;
        
        // verifica se teste = entrada
        bool teste_correto = true;
        for(size_t i = 0; i < entrada.size(); i++) {
            if(abs(saida_teste[i] - entrada[i]) > 0.0001f) {
                teste_correto = false;
                break;
            }
        }
        if(teste_correto) cout << "   ✓ MODO TESTE CORRETO" << endl;
        else cout << "   ✗ ERRO: Modo teste não preserva entrada" << endl;
    }
    // teste 3: retropropagação consistente
    {
        cout << "\n3. Teste de CONSISTÊNCIA na retropropagação:" << endl;
        
        Dropout dropout(0.4f, "teste", 777);
        dropout.treinando = true;
        
        vector<float> entrada = {1.0f, -2.0f, 3.0f};
        
        // propagação
        auto saida = dropout.prop(entrada);
        
        // gradiente de teste
        vector<float> gradiente = {0.5f, 1.0f, -0.5f};
        
        // retropropagação
        auto grad_entrada = dropout.retroprop(gradiente);
        
        cout << "   Entrada:       ";
        for(auto v : entrada) cout << setw(8) << v << " ";
        cout << "\n   Saída:         ";
        for(auto v : saida) cout << setw(8) << v << " ";
        cout << "\n   Gradiente:     ";
        for(auto v : gradiente) cout << setw(8) << v << " ";
        cout << "\n   Grad entrada:  ";
        for(auto v : grad_entrada) cout << setw(8) << v << " ";
        cout << endl;
        
        // verifica regra: se saida[i] = 0, então grad_entrada[i] = 0
        // se saida[i] ≠ 0, então grad_entrada[i] = gradiente[i] / (1-taxa)
        bool consistente = true;
        float fator = 1.0f / (1.0f - 0.4f);  // ≈ 1.66667
        
        for(size_t i = 0; i < entrada.size(); i++) {
            if(saida[i] == 0.0f) {
                if(abs(grad_entrada[i]) > 0.0001f) {
                    consistente = false;
                    break;
                }
            } else {
                float esperado = gradiente[i] * fator;
                if(abs(grad_entrada[i] - esperado) > 0.0001f) {
                    consistente = false;
                    break;
                }
            }
        }
        if(consistente) cout << "   ✓ RETROPROPAGAÇÃO CONSISTENTE" << endl;
        else cout << "   ✗ ERRO: Inconsistência na retropropagação" << endl;
    }
    cout << "\n\n=== TESTE OVERFIT COM DROPOUT ===" << endl;
    
    // dados simples
    int n_exemplos = 200;
    int n_recursos = 5;
    
    // gera dados simples
    vector<vector<float>> X(n_exemplos, vector<float>(n_recursos, 0.0f));
    vector<vector<float>> y(n_exemplos, vector<float>(1, 0.0f));
    
    mt19937 gen(42);
    normal_distribution<float> dist(0.0f, 1.0f);
    
    for(int i = 0; i < n_exemplos; i++) {
        for(int j = 0; j < n_recursos; j++) {
            X[i][j] = dist(gen);
        }
        y[i][0] = (X[i][0] * X[i][0] + sin(X[i][1])) > 0.5f ? 1.0f : 0.0f;
    }
    // divide
    int n_treino = 150;
    int n_teste = 50;
    
    // modelo sem dropout
    {
        cout << "\n1. SEM DROPOUT:" << endl;
        Modelo sem("sem_dropout");
        sem.adicionar(make_unique<Densa>(n_recursos, 10, "relu", true));
        sem.adicionar(make_unique<Densa>(10, 10, "relu", true));
        sem.adicionar(make_unique<Densa>(10, 1, "sigmoid", true));
        
        // treino
        float melhor_teste = 1.0f;
        for(int epoca = 0; epoca < 100; epoca++) {
            float perda_treino = 0.0f;
            
            for(int i = 0; i < n_treino; i++) {
                perda_treino += sem.treinar(X[i], y[i], mse, 0.01f);
            }
            perda_treino /= n_treino;
            
            // Teste
            sem.modoTeste();
            float perda_teste = 0.0f;
            for(int i = n_treino; i < n_treino + n_teste; i++) {
                auto pred = sem.prop(X[i]);
                perda_teste += mse(pred, y[i]);
            }
            perda_teste /= n_teste;
            sem.modoTreino();
            
            if(perda_teste < melhor_teste) {
                melhor_teste = perda_teste;
            }
            if(epoca % 20 == 0) {
                cout << "   Epoca " << epoca << ": treino=" << perda_treino 
                     << ", teste=" << perda_teste;
                if(perda_treino < 0.1f && perda_teste > perda_treino * 2.0f) {
                    cout << " ← OVERFIT";
                }
                cout << endl;
            }
        }
        cout << "   Melhor teste: " << melhor_teste << endl;
    }
    // modelo com dropout
    {
        cout << "\n2. COM DROPOUT (30%):" << endl;
        Modelo com("com_dropout");
        com.adicionar(make_unique<Densa>(n_recursos, 10, "relu", true));
        com.adicionar(make_unique<Dropout>(0.3f));
        com.adicionar(make_unique<Densa>(10, 10, "relu", true));
        com.adicionar(make_unique<Dropout>(0.3f));
        com.adicionar(make_unique<Densa>(10, 1, "sigmoid", true));
        
        // Treino
        float melhor_teste = 1.0f;
        for(int epoca = 0; epoca < 100; epoca++) {
            float perda_treino = 0.0f;
            
            for(int i = 0; i < n_treino; i++) {
                perda_treino += com.treinar(X[i], y[i], mse, 0.01f);
            }
            perda_treino /= n_treino;
            
            // teste
            com.modoTeste();
            float perda_teste = 0.0f;
            for(int i = n_treino; i < n_treino + n_teste; i++) {
                auto pred = com.prop(X[i]);
                perda_teste += mse(pred, y[i]);
            }
            perda_teste /= n_teste;
            com.modoTreino();
            
            if(perda_teste < melhor_teste) {
                melhor_teste = perda_teste;
            }
            if(epoca % 20 == 0) {
                cout << "   Epoca " << epoca << ": treino=" << perda_treino 
                     << ", teste=" << perda_teste << endl;
            }
        }
        cout << "   Melhor teste: " << melhor_teste << endl;
    }
    return 0;
}