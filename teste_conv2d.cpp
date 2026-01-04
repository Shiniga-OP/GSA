// teste_conv2d.h
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <random>
#include <algorithm>
#include "biblis/camadas.h"

using namespace std;

// função pra criar uma imagem de teste(8x8 com padrão simples)
vector<vector<vector<float>>> criarImagemTeste() {
    // imagem 1 canal(escala cinza), 8x8
    vector<vector<vector<float>>> imagem(1, vector<vector<float>>(8, vector<float>(8, 0.0f)));
    
    // adiciona alguns padrões
    for(int i = 2; i < 6; i++) {
        for(int j = 2; j < 6; j++) {
            imagem[0][i][j] = 1.0f;  // quadrado branco no centro
        }
    }
    // linhas diagonais
    for(int i = 0; i < 8; i++) {
        imagem[0][i][i] = 0.5f;
        imagem[0][i][7-i] = 0.5f;
    }
    return imagem;
}

void testeConv2DBasico() {
    cout << "=== Teste Conv2D Básico ===" << endl;
    
    // cria camada Conv2D: 4 filtros 3x3, entrada 1 canal
    Conv2D conv2d(4, 3, 3, 1, 1, 1, "relu", true, "conv_teste");
    
    // cria imagem de teste 8x8
    auto imagem = criarImagemTeste();
    
    cout << "Imagem original (8x8):" << endl;
    gravarImg2D(imagem[0]);
    
    // propagação
    auto saida = conv2d.propMapa(imagem);
    
    cout << "\nSaída após convolução:" << endl;
    cout << "Número de filtros: " << saida.size() << endl;
    cout << "Dimensões de cada mapa: " << saida[0].size() << "x" << saida[0][0].size() << endl;
    
    // imprime o primeiro filtro
    cout << "\nPrimeiro filtro de saída:" << endl;
    gravarImg2D(saida[0]);
    
    // teste de retropropagação
    cout << "\n=== Teste Retropropagação ===" << endl;
    
    // cria gradiente falso(mesmas dimensões da saida)
    vector<vector<vector<float>>> gradiente(saida.size());
    for(size_t i = 0; i < saida.size(); i++) {
        gradiente[i] = vector<vector<float>>(saida[0].size(),
        vector<float>(saida[0][0].size(), 0.1f));
    }
    auto gradEntrada = conv2d.retropropMapa(gradiente);
    
    cout << "Gradiente de entrada calculado." << endl;
    cout << "Dimensões: " << gradEntrada.size() << "x"
    << gradEntrada[0].size() << "x" << gradEntrada[0][0].size() << endl;
    
    // teste de atualização
    cout << "\n=== Teste Atualização de Pesos ===" << endl;
    conv2d.att(0.01f);
    cout << "Pesos atualizados com SGD." << endl;
    
    // informações da camada
    cout << "\n=== Informações da Camada ===" << endl;
    cout << "Nome: " << conv2d.nome << endl;
    cout << "Tipo: " << conv2d.tipo << endl;
    cout << "Número de parâmetros: " << conv2d.numParametros() << endl;
    cout << "Tem parâmetros: " << (conv2d.temParametros() ? "Sim" : "Não") << endl;
}

void testeCNN() {
    cout << "\n\n=== Teste CNN de modelo ===" << endl;
    
    // cria um modelo CNN
    Modelo cnn("CNN_Teste");
    
    // pra calcular as dimensões manualmente:
    // 1. Conv2D(8 filtros 3x3, espaco=1) em entrada 8x8 -> saida: 8x8 x 8 canais
    // 2. MaxReuso2D(2x2, passo=2) em 8x8 -> saida: 4x4 x 8 canais  
    // 3. Conv2D(16 filtros 3x3, espaco=1) em 4x4 -> saída: 4x4 x 16 canais
    // 4. MaxReuso2D(2x2, passo=2) em 4x4 -> saida: 2x2 x 16 canais
    // 5. Flatten: 2 * 2 * 16 = 64 elementos
    // 6. Densa: entrada 64, saida 10
    
    cout << "Dimensões esperadas:" << endl;
    cout << "Após conv1: 8x8 x 8 canais" << endl;
    cout << "Após reuso1: 4x4 x 8 canais" << endl;
    cout << "Após conv2: 4x4 x 16 canais" << endl;
    cout << "Após reuso2: 2x2 x 16 canais" << endl;
    cout << "Após flatten: 64 elementos" << endl;
    cout << "Densa: 64 -> 10" << endl;
    
    // adiciona camadas com dimensão pra Densa
    cnn.add(make_unique<Conv2D>(8, 3, 3, 1, 1, 1, "relu", true, "conv1"));
    cnn.add(make_unique<MaxReuso2D>(2, 2, "reuso1"));
    cnn.add(make_unique<Conv2D>(16, 3, 3, 8, 1, 1, "relu", true, "conv2"));
    cnn.add(make_unique<MaxReuso2D>(2, 2, "reuso2"));
    cnn.add(make_unique<Flatten>("flatten"));
    
    // dimensão correta: 2 * 2 * 16 = 64
    size_t entradaDensa = 2 * 2 * 16;
    cnn.add(make_unique<Densa>(entradaDensa, 10, "softmax", true, "dense_saida"));
    
    // imprime resumo
    cnn.resumo();
    
    // cria imagem de entrada(8x8) em formato 3D
    auto imagem = criarImagemTeste(); // retorna formato 3D
    
    // cria alvo falso(10 classes)
    vector<float> alvo(10, 0.0f);
    alvo[3] = 1.0f; // classe 3
    
    cout << "\n=== Teste Treino (uma epoca) ===" << endl;
    
    // primeiro faz apenas propagação pra ver as dimensões
    cout << "\n=== Teste de Propagação (sem treino) ===" << endl;
    cnn.modoTeste();
    
    vector<vector<vector<float>>> resultado = imagem;
    for(size_t i = 0; i < cnn.camadas.size(); i++) {
        auto& camada = cnn.camadas[i];
        auto* conv2d = dynamic_cast<Conv2D*>(camada.get());
        auto* maxreuso = dynamic_cast<MaxReuso2D*>(camada.get());
        auto* flatten = dynamic_cast<Flatten*>(camada.get());
        auto* densa = dynamic_cast<Densa*>(camada.get());
        
        cout << "\nCamada " << i << ": " << camada->nome << " (" << camada->tipo << ")" << endl;
        
        if(conv2d) {
            resultado = conv2d->propMapa(resultado);
            cout << "Dimensões após " << camada->nome << ": " 
                 << resultado.size() << " canais, " 
                 << resultado[0].size() << "x" << resultado[0][0].size() << endl;
        } else if(maxreuso) {
            resultado = maxreuso->propMapa(resultado);
            cout << "Dimensões após " << camada->nome << ": " 
                 << resultado.size() << " canais, " 
                 << resultado[0].size() << "x" << resultado[0][0].size() << endl;
        } else if(flatten) {
            resultado = flatten->propMapa(resultado);
            cout << "Dimensões após " << camada->nome << ": " 
                 << resultado.size() << " canais, " 
                 << resultado[0].size() << " linha(s), " 
                 << resultado[0][0].size() << " colunas" << endl;
            cout << "Total de elementos: " << resultado[0][0].size() << endl;
        } else if(densa) {
            // extrai o vetor 1D do formato 3D
            vector<float> entrada1D = resultado[0][0];
            cout << "Entrada para Densa: " << entrada1D.size() << " elementos" << endl;
            cout << "Densa espera: " << densa->entradaDim << " elementos" << endl;
            
            if(entrada1D.size() != densa->entradaDim) {
                cout << "ERRO: Dimensão incorreta! Entrada: " << entrada1D.size() 
                     << ", Esperado: " << densa->entradaDim << endl;
                return; // saia antes de causar erro
            }
            auto saida = densa->prop(entrada1D);
            cout << "Saída da rede: ";
            for(float val : saida) cout << val << " ";
            cout << endl;
            
            // encontra a classe prevista
            int predita = distance(saida.begin(), max_element(saida.begin(), saida.end()));
            cout << "Classe prevista: " << predita << endl;
            
            // pra manter o formato 3D, coloca a saida de volta no formato
            resultado = vector<vector<vector<float>>>(1, vector<vector<float>>(1, saida));
        }
    }
    // agora tenta o treino se as dimensões tiverem corretas
    cout << "\n=== Teste Treino (uma epoca) ===" << endl;
    
    auto perdaMSE = [](const vector<float>& saida, const vector<float>& alvo) {
        float soma = 0.0f;
        for(size_t i = 0; i < saida.size(); i++) {
            float diff = saida[i] - alvo[i];
            soma += diff * diff;
        }
        return soma / saida.size();
    };
    try {
        // treina:
        float erro = cnn.treinarMapa(imagem, alvo, perdaMSE, 0.01f);
        cout << "Erro após treino: " << erro << endl;
    } catch(const exception& e) {
        cerr << "Erro durante treino: " << e.what() << endl;
    }
}

void testeSerializacao() {
    cout << "\n\n=== Teste Serialização Conv2D ===" << endl;
    
    // cria camada
    Conv2D conv(4, 3, 3, 1, 1, 0, "relu", true, "conv_serial");
    
    // salva
    conv.salvar("conv_teste.bin");
    cout << "Camada salva em conv_teste.bin" << endl;
    
    // carrega em nova camada
    Conv2D conv2(1, 1, 1, 1, 1, 0, "linear", true, "conv_carregada");
    conv2.carregar("conv_teste.bin");
    
    cout << "Camada carregada: " << conv2.nome << endl;
    cout << "Número de filtros: " << conv2.filtros << endl;
    cout << "Tamanho filtro: " << conv2.alturaFiltro << "x" << conv2.larguraFiltro << endl;
}

int main() {
    try {
        testeConv2DBasico();
        testeCNN();
        testeSerializacao();
        cout << "\n=== Todos os testes passaram! ===" << endl;
    } catch(const exception& e) {
        cerr << "Erro: " << e.what() << endl;
        return 1;
    }
    return 0;
}