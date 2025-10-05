#include "biblis/ativas.h"
#include "biblis/toke.h"
#include "biblis/util.h"
#include "biblis/camadas.h"

class Percepron {
    public:
    std::vector<float> pesos;
    std::vector<float> bias;
    float taxaAprendizado;
    
    Percepron() {
        pesos = vetor(2);
        bias = vetor(1);
        taxaAprendizado = 0.01f;
    }
    int prever(std::vector<float> entrada) {
        float soma = bias[0];
        for(size_t i = 0; i < entrada.size(); i++) {
            soma += pesos[i] * entrada[i];
        }
        return degrau(soma);
    }
    void treinar(std::vector<float> entrada, int saidaAlvo) {
        int saida = prever(entrada);
        
        int erro = saidaAlvo - saida;
        
        for(size_t i = 0; i < entrada.size(); i++) {
            pesos[i] += entrada[i] * erro * taxaAprendizado;
        }
        bias[0] += erro * taxaAprendizado;
    }
};

int main() {
    Percepron p;
    // treina por 100 epocas:
    for(size_t epoca = 0; epoca < 100; epoca++) {
        // dados de treino:
        p.treinar({1, 1}, 1); // exemplo 1
        p.treinar({1, 0}, 0); // exemplo 2
        p.treinar({0, 1}, 0); // exemplo 3
        p.treinar({0, 0}, 0); // exemplo 4
    }
    printf("[1, 1]: %i\n", p.prever({1, 1})); // esperado: 1
    printf("[1, 0]: %i\n", p.prever({1, 0})); // esperado: 0
    printf("[0, 1]: %i\n", p.prever({0, 1})); // esperado: 0
    printf("[0, 0]: %i\n", p.prever({0, 0})); // esperado: 0
    return 0;
}