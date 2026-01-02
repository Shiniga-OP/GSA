# GSA
em desenvolvimento com a biblioteca própria.

## conteúdo:
1. ativações: degrau, sigmoid, tanh, ReLU, GELU, e etc, + deverivadas.
2. tokenizadores: tokenzador sub palavra.
3. utilitários: pesos, atualização de pesos, perda, erro, saída, métricas, matrizes 2D/3D + vetores.
4. camaadas: camada densa.
5. otimizadores: Adam, AdamW, SGD, AdaGrad, RMSprop, AdaDelta, Nesterov.

implementação feita do zero, sem bibliotecas de IA prontas (escrita no Android via Termux e CodeEditor).

## arquivos:

* teste_xor.cpp: testa o aprendizado não linear e treinamento por lote da camada densa.
* teste.cpp:
```Cpp
// teste.cpp
#include "biblis/camadas.h"
#include <cassert>
#include <iomanip>

// teste pra camada densa
void testeCD1() {
    cout << "=== TESTE CAMADA DENSA ===\n\n";
    
    // teste 1: camada linear simples
    cout << "1. Teste camada linear (sem ativação):\n";
    Densa camada1(3, 2, "linear");
    
    // define pesos manualmente pra teste deterministico
    camada1.defPesos({{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}});
    camada1.defBias({0.1f, 0.2f});
    
    vector<float> entrada = {1.0f, 2.0f, 3.0f};
    vector<float> saida = camada1.prop(entrada);
    
    cout << "Entrada: ";
    for(float x : entrada) cout << x << " ";
    cout << "\nSaída: ";
    for(float x : saida) cout << x << " ";
    cout << "Esperado: 1.5 3.4\n\n";
    
    // teste 2: retropropagação
    cout << "2. Teste retropropagação:\n";
    vector<float> gradSaida = {0.1f, -0.2f};
    vector<float> gradEntrada = camada1.retroprop(gradSaida);
    
    cout << "Gradiente de entrada: ";
    for(float x : gradEntrada) cout << x << " ";
    cout << "\n\n";
    
    // teste 3: atualização de pesos
    cout << "3. Teste atualização de pesos:\n";
    camada1.att(0.01f);
    camada1.zerarGradientes();
    cout << "Pesos atualizados e gradientes zerados.\n\n";
    
    // teste 4: camada com ativação sigmoid
    cout << "4. Teste camada com sigmoid:\n";
    Densa camada2(2, 2, "sigmoid");
    camada2.defPesos({{0.5f, -0.3f}, {0.2f, 0.8f}});
    camada2.defBias({0.1f, -0.1f});
    
    vector<float> entrada2 = {0.5f, 0.8f};
    vector<float> saida2 = camada2.prop(entrada2);
    
    cout << "Entrada: ";
    for(float x : entrada2) cout << x << " ";
    cout << "\nSaída (sigmoid): ";
    for(float x : saida2) cout << x << " ";
    cout << "\n\n";
    
    // teste 5: processamento em lote
    cout << "5. Teste processamento em lote:\n";
    Densa camada3(2, 1, "relu");
    vector<vector<float>> lote = {
        {1.0f, 2.0f},
        {3.0f, 4.0f},
        {5.0f, 6.0f}
    };
    auto saidaLote = camada3.propLote(lote);
    cout << "Lote com " << saidaLote.size() << " exemplos processados.\n";
    
    // teste 6: informações da camada
    cout << "\n6. Informações da camada:\n";
    cout << "Tipo: " << camada1.tipo << endl;
    cout << "Tem parâmetros: " << (camada1.temParametros() ? "Sim" : "Não") << endl;
    cout << "Número de parâmetros: " << camada1.numParametros() << endl;
}
void testeCD2() {
    // teste 7: camada sem bias
    cout << "7. Teste camada SEM bias:\n";
    Densa camadaSemBias(2, 3, "linear", false);
    camadaSemBias.defPesos({{0.1f, 0.2f}, {0.3f, 0.4f}, {0.5f, 0.6f}});
    
    vector<float> entrada7 = {1.0f, 2.0f};
    vector<float> saida7 = camadaSemBias.prop(entrada7);
    
    cout << "Entrada: ";
    for(float x : entrada7) cout << x << " ";
    cout << "\nSaída (sem bias): ";
    for(float x : saida7) cout << x << " ";
    cout << "\nEsperado: 0.5 1.1 1.7\n\n";
    
    // verificação
    assert(fabs(saida7[0] - 0.5f) < 1e-6);
    assert(fabs(saida7[1] - 1.1f) < 1e-6);
    assert(fabs(saida7[2] - 1.7f) < 1e-6);
    
    // teste 8: diferentes funções de ativação
    cout << "8. Teste diferentes funções de ativação:\n";
    
    // ReLU
    Densa camadaReLU(2, 2, "relu");
    camadaReLU.defPesos({{1.0f, -1.0f}, {0.5f, 0.5f}});
    camadaReLU.defBias({-0.5f, 0.0f});
    
    vector<float> entrada8 = {1.0f, -1.0f};
    vector<float> saidaReLU = camadaReLU.prop(entrada8);
    cout << "ReLU entrada [" << entrada8[0] << ", " << entrada8[1] << "]: ";
    cout << "[" << saidaReLU[0] << ", " << saidaReLU[1] << "]\n";
    
    // tanh
    Densa camadaTanh(2, 1, "tanh");
    camadaTanh.defPesos({{0.5f, -0.3f}});
    camadaTanh.defBias({0.2f});
    
    vector<float> entradaTanh = {0.8f, 0.2f};
    vector<float> saidaTanh = camadaTanh.prop(entradaTanh);
    cout << "Tanh entrada [" << entradaTanh[0] << ", " << entradaTanh[1] << "]: ";
    cout << "[" << saidaTanh[0] << "]\n\n";
    
    // teste 9: softmax(especial)
    cout << "9. Teste camada com softmax:\n";
    Densa camadaSoftmax(3, 3, "softmax");
    camadaSoftmax.defPesos({
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f},
        {0.7f, 0.8f, 0.9f}
    });
    camadaSoftmax.defBias({0.1f, 0.2f, 0.3f});
    
    vector<float> entrada9 = {1.0f, 0.5f, 0.2f};
    vector<float> saidaSoftmax = camadaSoftmax.prop(entrada9);
    
    cout << "Entrada: ";
    for(float x : entrada9) cout << x << " ";
    cout << "\nSaída softmax: ";
    for(float x : saidaSoftmax) cout << fixed << setprecision(4) << x << " ";
    
    // verifica se a soma é aproximadamente 1
    float soma = 0.0f;
    for(float x : saidaSoftmax) soma += x;
    cout << "\nSoma: " << soma << " (deve ser ~1.0)\n\n";
    assert(fabs(soma - 1.0f) < 1e-4);
    
    // teste 10: retropropagação com ativação
    cout << "10. Teste retropropagação completa com sigmoid:\n";
    Densa camadaSigmoid(2, 2, "sigmoid");
    camadaSigmoid.defPesos({{0.5f, -0.5f}, {0.3f, 0.7f}});
    camadaSigmoid.defBias({0.1f, -0.1f});
    
    // propagação
    vector<float> entrada10 = {0.6f, 0.4f};
    vector<float> saida10 = camadaSigmoid.prop(entrada10);
    
    // retropropagação
    vector<float> gradSaida10 = {0.1f, -0.1f};
    vector<float> gradEntrada10 = camadaSigmoid.retroprop(gradSaida10);
    
    cout << "Gradiente entrada: [" << gradEntrada10[0] << ", " << gradEntrada10[1] << "]\n";
    
    // atualiza e verifica se gradientes foram zerados
    camadaSigmoid.att(0.01f);
    camadaSigmoid.zerarGradientes();
    
    // teste 11: serialização(salvar/carregar)
    cout << "\n11. Teste serialização (salvar/carregar):\n";
    Densa camadaOriginal(3, 2, "tanh");
    camadaOriginal.defPesos({{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}});
    camadaOriginal.defBias({0.7f, 0.8f});
    
    // salva
    camadaOriginal.salvar("teste_camada.bin");
    cout << "Camada salva em \"teste_camada.bin\"\n";
    
    // carrega em nova camada
    Densa camadaCarregada(3, 2, "linear"); // tipo diferente inicialmente
    camadaCarregada.carregar("teste_camada.bin");
    
    // testa se são iguais
    vector<float> testeEntrada = {1.0f, 2.0f, 3.0f};
    vector<float> saidaOriginal = camadaOriginal.prop(testeEntrada);
    vector<float> saidaCarregada = camadaCarregada.prop(testeEntrada);
    
    bool iguais = true;
    for(size_t i = 0; i < saidaOriginal.size(); i++) {
        if(fabs(saidaOriginal[i] - saidaCarregada[i]) > 1e-6) {
            iguais = false;
            break;
        }
    }
    cout << "Camada carregada " << (iguais ? "IGUAL" : "DIFERENTE") << " da original\n";
    assert(iguais);
    
    // teste 12: propriedades da camada
    cout << "\n12. Teste propriedades:\n";
    Densa camadaGrande(100, 50, "relu");
    cout << "Camada 100x50 com ReLU:\n";
    cout << "Número de parâmetros: " << camadaGrande.numParametros() << endl;
    cout << "Deve ser: " << (100*50 + 50) << " (pesos + bias)\n";
    
    assert(camadaGrande.numParametros() == (100*50 + 50));
    
    // teste 13: exceções
    cout << "\n13. Teste exceções:\n";
    try {
        Densa camadaErro(2, 3, "linear");
        vector<float> entradaErrada = {1.0f}; // tamanho errado
        camadaErro.prop(entradaErrada);
        cout << "ERRO: Deveria ter lançado exceção!\n";
        assert(false);
    } catch(const invalid_argument& e) {
        cout << "Exceção capturada corretamente: " << e.what() << "\n";
    }
    // teste 14: lote com diferentes tamanhos
    cout << "\n14. Teste lote misto:\n";
    Densa camadaLote(4, 2, "linear");
    vector<vector<float>> loteMisto = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {0.5f, 1.5f, 2.5f, 3.5f},
        {0.0f, 0.0f, 1.0f, 1.0f}
    };
    auto saidaLote = camadaLote.propLote(loteMisto);
    cout << "Lote de " << saidaLote.size() << " exemplos processado.\n";
    cout << "Primeira saída: [" << saidaLote[0][0] << ", " << saidaLote[0][1] << "]\n";
}

void testeCD() {
    cout << fixed << setprecision(4);
    testeCD1();
    testeCD2();
    cout << "\n=== FIM CAMADA DENSA ===\n";
}

// teste de utilitarios:
void _testeMatrizes() {
    cout << "\n=== TESTES MATRIZES ===\n\n";
    auto t3d = tensor3D(2, 2, 2, 0.1f);
    cout << "Tensor 3D:\n";
    for(const auto& m : t3d) {
        for(const auto& l : m) {
            for (float v : l) cout << v << " ";
            cout << "\n";
        }
        cout << "\n";
    }
    auto z3d = zeros3D(2, 2, 2);
    cout << "Zeros 3D:\n";
    for(const auto& m : z3d) {
        for(const auto& l : m) {
            for(float v : l) cout << v << " ";
            cout << "\n";
        }
        cout << "\n";
    }
    auto mapeado = mapear3D(t3d, [](float x) { return x * 2.0f; });
    cout << "Mapear 3D (x2):\n";
    for(const auto& m : mapeado) {
        for(const auto& l : m) {
            for(float v : l) cout << v << " ";
            cout << "\n";
        }
        cout << "\n";
    }
    auto soma3d = somar3D(t3d, z3d);
    cout << "Soma 3D:\n";
    for(const auto& m : soma3d) {
        for(const auto& l : m) {
            for (float v : l) cout << v << " ";
            cout << "\n";
        }
        cout << "\n";
    }
    auto m = matriz(2, 3, 0.1f);
    cout << "Matriz 2x3:\n";
    for(const auto& l : m) {
        for(float v : l) cout << v << " ";
        cout << "\n";
    }
    vector<float> v1 = {1.0f, 2.0f};
    vector<float> v2 = {3.0f, 4.0f};
    auto ext = exterior(v1, v2);
    cout << "Produto Externo:\n";
    for(const auto& l : ext) {
        for(float v : l) cout << v << " ";
        cout << "\n";
    }
    auto m2 = matriz(2, 3, 0.1f);
    auto soma_m = somarMatriz(m, m2);
    cout << "Soma Matriz:\n";
    for(const auto& l : soma_m) {
        for(float v : l) cout << v << " ";
        cout << "\n";
    }
    auto m3 = matriz(3, 2, 0.1f);
    auto prod_m = multMatrizes(m, m3);
    cout << "Multiplicação Matrizes:\n";
    for(const auto& l : prod_m) {
        for(float v : l) cout << v << " ";
        cout << "\n";
    }
}
void testeU() {
    cout << "\n=== TESTES SAIDA ===\n\n";
    vector<float> entrada = {1.0f, 2.0f, 3.0f};
    vector<float> saida_softmax = softmax(entrada, 1.0f);
    cout << "Softmax: ";
    for(float x : saida_softmax) cout << x << " ";
    cout << endl;
    vector<float> grad = {0.1f, 0.2f, 0.3f};
    vector<float> saida_derivada = derivadaSoftmax(saida_softmax, grad);
    cout << "Derivada Softmax: ";
    for(float x : saida_derivada) cout << x << " ";
    cout << endl;
    vector<vector<float>> matriz = {{1.0f, 2.0f, 3.0f}, {1.0f, 3.0f, 2.0f}};
    vector<vector<float>> saida_lote = softmaxLote(matriz, 1.0f);
    cout << "Softmax Lote:\n";
    for(const auto& linha : saida_lote) {
        for(float x : linha) cout << x << " ";
        cout << endl;
    }
    cout << "Argmax: " << argmax(entrada) << endl;
    vector<float> saida_ruido = addRuido(entrada, 0.01f);
    cout << "Add Ruido: ";
    for(float x : saida_ruido) cout << x << " ";
    cout << endl;
    cout << "\n=== TESTES ERRO ===\n\n";
    vector<float> saida = {0.8f, 0.2f, 0.5f};
    vector<float> esperado = {1.0f, 0.0f, 0.5f};
    vector<float> y = {1.0f, 0.0f, 0.0f};
    vector<float> yChapeu = {0.7f, 0.2f, 0.1f};
    vector<float> ancora = {0.5f, 0.5f};
    vector<float> positiva = {0.6f, 0.6f};
    vector<float> negativa = {0.4f, 0.4f};
    vector<float> saida1 = {0.1f, 0.9f};
    vector<float> saida2 = {0.2f, 0.8f};
    cout << "erroAbsolutoMedio: " << erroAbsolutoMedio(saida, esperado) << endl;
    cout << "erroQuadradoEsperado: " << erroQuadradoEsperado(saida, esperado) << endl;
    auto derivErro = derivadaErro(saida, esperado);
    cout << "derivadaErro: [";
    for(float val : derivErro) cout << val << " ";
    cout << "]\n";
    cout << "entropiaCruzada: " << entropiaCruzada(y, yChapeu) << endl;
    auto derivEntropia = derivadaEntropiaCruzada(y, yChapeu);
    cout << "derivadaEntropiaCruzada: [";
    for(float val : derivEntropia) cout << val << " ";
    cout << "]\n";
    cout << "huberPerda: " << huberPerda(saida, esperado) << endl;
    auto derivHuber = derivadaHuber(saida, esperado);
    cout << "derivadaHuber: [";
    for(float val : derivHuber) cout << val << " ";
    cout << "]\n";
    cout << "perdaTripleto: " << perdaTripleto(ancora, positiva, negativa) << endl;
    cout << "contrastivaPerda (rotulo=1): " << contrastivaPerda(saida1, saida2, 1) << endl;
    cout << "contrastivaPerda (rotulo=0): " << contrastivaPerda(saida1, saida2, 0) << endl;
    cout << "\n=== TESTES REGULARIZAÇÃO E MÉTRICAS ===\n\n";
    vector<vector<float>> pesos = {{1.0f, -2.0f, 0.0f}, {3.0f, -4.0f, 5.0f}};
    vector<vector<float>> tensor = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    vector<float> vetor = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    vector<vector<float>> saidas = {{0.8f, 0.1f, 0.1f}, {0.2f, 0.7f, 0.1f}};
    vector<vector<float>> esperados = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    vector<vector<int>> matrizConfusao = {{5, 2}, {1, 8}};
    vector<float> p = {0.4f, 0.6f};
    vector<float> q = {0.5f, 0.5f};
    vector<float> pontos = {0.8f, 0.6f, 0.4f, 0.2f};
    vector<int> rotulos = {1, 0, 1, 0};
    auto l1 = regularL1(pesos, 0.1f);
    cout << "L1 Regularization: [";
    for(const auto& linha : l1) {
        for(float val : linha) cout << val << " ";
    }
    cout << "]\n";
    auto l2 = regularL2(pesos, 0.1f);
    cout << "L2 Regularization: [";
    for(const auto& linha : l2) {
        for(float val : linha) cout << val << " ";
    }
    cout << "]\n";
    auto dropoutRes = dropout(tensor, 0.3f);
    cout << "Dropout: [";
    for(const auto& linha : dropoutRes) {
        for(float val : linha) cout << val << " ";
    }
    cout << "]\n";
    auto normEnt = normEntrada(vetor);
    cout << "Normalização Entrada: [";
    for(float val : normEnt) cout << val << " ";
    cout << "]\n";
    auto normZ = normZPonto(vetor);
    cout << "Normalização Z: [";
    for(float val : normZ) cout << val << " ";
    cout << "]\n";
    cout << "Acurácia: " << acuracia(saidas, esperados) << "\n";
    cout << "Precisão: " << precisao(matrizConfusao) << "\n";
    cout << "Recall: " << recall(matrizConfusao) << "\n";
    cout << "F1-ponto: " << f1Ponto(matrizConfusao) << "\n";
    cout << "MSE: " << mse({1.0f, 2.0f}, {1.5f, 1.8f}) << "\n";
    cout << "KL Divergence: " << klDivergencia(p, q) << "\n";
    cout << "ROC AUC: " << rocAuc(pontos, rotulos) << "\n";
    cout << "\n=== TESTES PESOS ===\n\n";
    vector<vector<float>> pesos2 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    vector<vector<float>> grad2 = {{0.1f, 0.2f}, {0.3f, 0.4f}};
    vector<vector<float>> velocidade = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    vector<vector<float>> m = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    vector<vector<float>> v = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    auto xavier = iniPesosXavier(2, 3);
    cout << "Xavier: ";
    for(const auto& linha : xavier) {
        for(float val : linha) cout << val << " ";
    }
    cout << "\n";
    auto novosPesos = attPesos(pesos2, grad2, 0.01f);
    cout << "attPesos: ";
    for(const auto& linha : novosPesos) {
        for(float val : linha) cout << val << " ";
    }
    cout << "\n";
    _testeMatrizes();
}

int main() {
    testeU(); // testa todos os utilitários
    cout << "\n";
    testeCD(); // testa a camada densa
    return 0;
}
```