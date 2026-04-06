// biblis/camadas.h
#pragma once
#include <functional>
#include "util.h"
#include "camadas/camada.h"
#include "camadas/densa.h"
#include "camadas/dropout.h"
#include "camadas/lotenorm.h"
#include "camadas/conv2d.h"
#include "camadas/maxreuso2d.h"
#include "camadas/flatten.h"
#include "camadas/atencao.h"
#include "camadas/norm.h"
#include "camadas/perda.h"
#include "camadas/embedding.h"
#include "camadas/posicional.h"
#include "camadas/bloco.h"

// modelo pra gerenciamento de camadas:
class Modelo {
public:
    vector<unique_ptr<Camada>> camadas;
    string nome;
    bool treinando;
    
    Modelo(const string& nome = "") : nome(nome), treinando(true) {}
    
    void add(unique_ptr<Camada> camada) {
        camadas.push_back(std::move(camada));
    }
    
    // propagação
    vector<float> prop(const vector<float>& entrada) {
        vector<float> resultado = entrada;
        
        for(const auto& camada : camadas) {
            resultado = camada->prop(resultado);
        }
        return resultado;
    }
    
    // propagação em lote
    vector<vector<float>> propLote(const vector<vector<float>>& entrada) {
        vector<vector<float>> resultado = entrada;
        
        for(const auto& camada : camadas) {
            resultado = camada->propLote(resultado);
        }
        return resultado;
    }
    
    // retropropagação atraves do modelo inteiro
    vector<float> retroprop(const vector<float>& gradiente) {
        vector<float> grad = gradiente;
        
        // retropropaga na ordem inversa
        for(int i = camadas.size() - 1; i >= 0; i--) {
            grad = camadas[i]->retroprop(grad).vetor;
        }
        return grad;
    }
    
    // atualização dos pesos
    void att(float taxaAprendizado) {
        for(auto& camada : camadas) {
            if(camada->temParametros()) {
                camada->att(taxaAprendizado);
            }
        }
    }
    
    void zerarGradientes() {
        for(auto& camada : camadas) {
            if(camada->temParametros()) {
                camada->zerarGradientes();
            }
        }
    }
    
    void modoTreino() {
        treinando = true;
        for(auto& camada : camadas) {
            auto* dropout = dynamic_cast<Dropout*>(camada.get());
            if(dropout) dropout->treinando = true;
            
            auto* lotenorm = dynamic_cast<LoteNorm*>(camada.get());
            if(lotenorm) lotenorm->treinando = true;
        }
    }
    
    void modoTeste() {
        treinando = false;
        for(auto& camada : camadas) {
            auto* dropout = dynamic_cast<Dropout*>(camada.get());
            if(dropout) dropout->treinando = false;
            
            auto* lotenorm = dynamic_cast<LoteNorm*>(camada.get());
            if(lotenorm) lotenorm->treinando = false;
        }
    }
    
    float treinar(const vector<float>& entrada, const vector<float>& alvo, 
    function<float(const vector<float>&, const vector<float>&)> perda = mse,
    function<vector<float>(const vector<float>&, const vector<float>&)> derivadaPerda = derivadaMse,
    float taxaAprendizado = 0.01f) {
        // garante que ta no modo treino
        modoTreino();
        
        // propagação
        vector<float> saida = prop(entrada);
        
        // calcula a perda
        float erro = perda(saida, alvo);
        
        // gradiente inicial
        vector<float> gradiente = derivadaPerda(saida, alvo);
        
        // repropagação
        for(int i = camadas.size() - 1; i >= 0; i--) {
            gradiente = camadas[i]->retroprop(gradiente).vetor;
        }
        // atualiza pesos(apenas camadas com parametros)
        for(auto& camada : camadas) {
            if(camada->temParametros()) {
                camada->att(taxaAprendizado);
            }
        }
        // zera gradientes(apenas camadas com parametros)
        for(auto& camada : camadas) {
            if(camada->temParametros()) {
                camada->zerarGradientes();
            }
        }
        return erro;
    }
    
    // pra CNNs:
    float treinarMapa(const vector<vector<vector<float>>>& entrada3D, 
    const vector<float>& alvo,
    function<float(const vector<float>&, const vector<float>&)> perda = mse,
    function<vector<float>(const vector<float>&, const vector<float>&)> derivadaPerda = derivadaMse,
    float taxaAprendizado = 0.01f) {
        modoTreino();
        zerarGradientes();
        
        // propagação pra CNN
        vector<vector<vector<float>>> resultado3D = entrada3D;
        
        // propaga atraves de todas as camadas
        for(size_t i = 0; i < camadas.size(); i++) {
            auto* conv2d = dynamic_cast<Conv2D*>(camadas[i].get());
            auto* maxpool = dynamic_cast<MaxReuso2D*>(camadas[i].get());
            auto* flatten = dynamic_cast<Flatten*>(camadas[i].get());
            auto* densa = dynamic_cast<Densa*>(camadas[i].get());
            
            if(conv2d) resultado3D = conv2d->propMapa(resultado3D);
            else if(maxpool) resultado3D = maxpool->propMapa(resultado3D);
            else if(flatten) resultado3D = flatten->propMapa(resultado3D);
            else if(densa) {
                // verifica dimensão antes de processar
                vector<float> entrada1D = resultado3D[0][0];
                if(entrada1D.size() != densa->entradaDim) {
                    throw runtime_error("[" + densa->nome + "]: Dimensão de entrada incorreta. Recebido: " + 
                    to_string(entrada1D.size()) + ", Esperado: " + to_string(densa->entradaDim));
                }
                auto saida1D = densa->prop(entrada1D);
                
                // coloca de volta no formato 3D pra consistencia
                resultado3D = vector<vector<vector<float>>>(1, vector<vector<float>>(1, saida1D));
            }
        }
        // extrai resultado final
        vector<float> saidaFinal = resultado3D[0][0];
        
        // calcula perda
        float erro = perda(saidaFinal, alvo);
        
        // calcula gradiente inicial
        vector<float> gradiente1D = derivadaPerda(saidaFinal, alvo);
        // converte gradiente 1D pra 3D pra retropropagação
        vector<vector<vector<float>>> gradiente3D(1, vector<vector<float>>(1, gradiente1D));
        
        // retropropagação na ordem inversa
        for(int i = camadas.size() - 1; i >= 0; i--) {
            auto* conv2d = dynamic_cast<Conv2D*>(camadas[i].get());
            auto* maxpool = dynamic_cast<MaxReuso2D*>(camadas[i].get());
            auto* flatten = dynamic_cast<Flatten*>(camadas[i].get());
            auto* densa = dynamic_cast<Densa*>(camadas[i].get());
            
            if(densa) {
                // densa: gradiente ja ta em formato 3D(1x1xN)
                gradiente3D = densa->retropropMapa(gradiente3D);
            } else if(flatten) {
                // flatten: gradiente chega em formato 3D, retropropaga pra 3D
                gradiente3D = flatten->retropropMapa(gradiente3D);
            } else if(maxpool) {
                // maxPool: retropropaga em formato 3D
                gradiente3D = maxpool->retropropMapa(gradiente3D);
            } else if(conv2d) {
                // conv2D: retropropaga em formato 3D
                gradiente3D = conv2d->retropropMapa(gradiente3D);
            }
        }
        // atualiza pesos
        att(taxaAprendizado);
        
        return erro;
    }
    // info
    size_t numParametros() const {
        size_t total = 0;
        for(const auto& camada : camadas) {
            total += camada->numParametros();
        }
        return total;
    }
    
    void resumo() const {
        cout << "=== Modelo: " << nome << " ===" << endl;
        cout << "Numero de camadas: " << camadas.size() << endl;
        cout << "Total de parametros: " << numParametros() << endl;
        cout << "Camadas:" << endl;
        
        for(size_t i = 0; i < camadas.size(); i++) {
            cout << "  [" << i << "] " << camadas[i]->nome 
            << " (" << camadas[i]->tipo << ")" 
            << " - Parametros: " << camadas[i]->numParametros() << endl;
        }
    }
};