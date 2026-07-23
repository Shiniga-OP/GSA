// teste_causal.cpp
// Testa duas coisas na camada MultiCabeca (biblis/camadas/multicabeca.h):
//   1) CAUSALIDADE: mudar a entrada numa posicao futura (t > q) nao pode
//      alterar nem a saida[q] (forward) nem gradEntrada[q] (backward).
//   2) CORRETUDE DO GRADIENTE: gradEntrada analitico (retroprop) comparado
//      contra diferencas finitas centrais, posicao a posicao.
//
// Compilar (a partir da pasta que contem a pasta biblis/):
//   g++ -O2 -std=c++17 -I. teste_causal.cpp -o teste_causal
// Rodar:
//   ./teste_causal
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "biblis/camadas/multicabeca.h"

static void preencherAleatorio(float* v, int n, unsigned int* seed) {
    for(int i = 0; i < n; i++) {
        *seed = (*seed) * 1664525u + 1013904223u;
        float u = (float)((*seed) >> 8) / (float)(1u << 24); // [0,1)
        v[i] = (u * 2.0f - 1.0f) * 0.5f; // [-0.5, 0.5]
    }
}

int main() {
    int dim = 16;
    int nCab = 4;
    int seq = 5;
    int seqMax = 8;
    unsigned int seed = 12345u;

    printf("=== teste_causal (MultiCabeca) ===\n");
    printf("dim=%d nCab=%d seq=%d seqMax=%d\n\n", dim, nCab, seq, seqMax);

    MultiCabeca mha(dim, nCab, seqMax);
    mha.seqAtual = seq;

    float* entrada = (float*)malloc(seq * dim * sizeof(float));
    float* saida1  = (float*)malloc(seq * dim * sizeof(float));
    float* saida2  = (float*)malloc(seq * dim * sizeof(float));
    preencherAleatorio(entrada, seq * dim, &seed);

    // ---------------------------------------------------------------
    // TESTE 1: causalidade no FORWARD
    // muda so a entrada da ULTIMA posicao (t = seq-1, o "futuro" mais
    // extremo) e verifica se saida[0..seq-2] muda.
    // ---------------------------------------------------------------
    mha.prop(entrada, saida1);

    float* entradaMod = (float*)malloc(seq * dim * sizeof(float));
    memcpy(entradaMod, entrada, seq * dim * sizeof(float));
    for(int i = 0; i < dim; i++) entradaMod[(seq - 1) * dim + i] += 10.0f; // altera so o ultimo token

    mha.seqAtual = seq;
    mha.prop(entradaMod, saida2);

    float maxDiffPassado = 0.0f;
    for(int t = 0; t < seq - 1; t++) {
        for(int i = 0; i < dim; i++) {
            float d = fabsf(saida1[t*dim+i] - saida2[t*dim+i]);
            if(d > maxDiffPassado) maxDiffPassado = d;
        }
    }
    float diffUltimo = 0.0f;
    for(int i = 0; i < dim; i++) {
        diffUltimo += fabsf(saida1[(seq-1)*dim+i] - saida2[(seq-1)*dim+i]);
    }

    printf("[Forward] max|dif| em posicoes passadas (t<seq-1) ao mudar t=seq-1: %.8f\n", maxDiffPassado);
    printf("[Forward] soma|dif| na propria posicao alterada (t=seq-1): %.8f (esperado > 0)\n", diffUltimo);
    bool forwardCausal = maxDiffPassado < 1e-5f;
    printf("Forward respeita causalidade? %s\n\n", forwardCausal ? "SIM (OK)" : "NAO (FALHOU)");

    // ---------------------------------------------------------------
    // TESTE 2: causalidade no BACKWARD
    // gradEntrada[q] para q < seq-1 nao deve depender de gradSaida[seq-1]
    // (ou seja: mudar gradSaida so na ultima posicao so pode alterar
    // gradEntrada na ultima posicao ou anteriores conforme a causalidade,
    // nunca criar dependencia "de tras pra frente" nova).
    // Estrategia mais direta: comparar gradEntrada[q<seq-1] usando dois
    // gradSaida diferentes, que so diferem na posicao seq-1.
    // ---------------------------------------------------------------
    mha.seqAtual = seq;
    mha.prop(entrada, saida1); // repropaga com entrada original p/ recalcular buffers internos (P, Q, K, V...)

    float* gradSaidaA = (float*)calloc(seq * dim, sizeof(float));
    float* gradSaidaB = (float*)calloc(seq * dim, sizeof(float));
    preencherAleatorio(gradSaidaA, seq * dim, &seed);
    memcpy(gradSaidaB, gradSaidaA, seq * dim * sizeof(float));
    for(int i = 0; i < dim; i++) gradSaidaB[(seq-1)*dim + i] += 3.0f; // perturba so grad da ultima posicao

    float* gradEntradaA = (float*)malloc(seq * dim * sizeof(float));
    float* gradEntradaB = (float*)malloc(seq * dim * sizeof(float));

    mha.retroprop(gradSaidaA, gradEntradaA);
    // retroprop usa buffers internos (P,Q,K,V,ultEnt) da ULTIMA chamada de prop();
    // como nao chamamos prop() de novo, os dois retroprop usam o mesmo estado direto,
    // isolando a diferenca so ao gradSaida, que e o que queremos testar.
    mha.retroprop(gradSaidaB, gradEntradaB);

    float maxDiffGradPassado = 0.0f;
    for(int t = 0; t < seq - 1; t++) {
        for(int i = 0; i < dim; i++) {
            float d = fabsf(gradEntradaA[t*dim+i] - gradEntradaB[t*dim+i]);
            if(d > maxDiffGradPassado) maxDiffGradPassado = d;
        }
    }
    printf("[Backward] max|dif| em gradEntrada[t<seq-1] ao perturbar gradSaida[seq-1]: %.8f\n", maxDiffGradPassado);
    bool backwardCausal = maxDiffGradPassado < 1e-5f;
    printf("Backward respeita causalidade (grad futuro nao vaza pro passado)? %s\n\n",
           backwardCausal ? "SIM (OK)" : "NAO (FALHOU)");

    // ---------------------------------------------------------------
    // TESTE 3: corretude do gradiente via diferencas finitas centrais
    // compara gradEntradaA (analitico, de gradSaidaA) com dL/dentrada
    // numerico, onde L = soma(saida .* gradSaidaA) (produto interno,
    // que e exatamente o que gradSaida representa numa retropropagacao real)
    // ---------------------------------------------------------------
    float* bufPerda = (float*)malloc(seq * dim * sizeof(float));
    auto perdaLinear = [&](const float* ent) -> float {
        mha.seqAtual = seq;
        mha.prop(ent, bufPerda);
        float L = 0.0f;
        for(int i = 0; i < seq*dim; i++) L += bufPerda[i] * gradSaidaA[i];
        return L;
    };

    float eps = 1e-3f;
    int amostras = 0;
    float maxErroRel = 0.0f;
    for(int t = 0; t < seq; t++) {
        for(int i = 0; i < dim; i += 3) { // amostra algumas dims p/ nao ser lento
            float* alvo = &entrada[t*dim+i];
            float orig = *alvo;

            *alvo = orig + eps;
            float Lp = perdaLinear(entrada);
            *alvo = orig - eps;
            float Lm = perdaLinear(entrada);
            *alvo = orig;

            float gradNumerico = (Lp - Lm) / (2.0f*eps);
            float gradAnalitico = gradEntradaA[t*dim+i];

            float erro = fabsf(gradNumerico - gradAnalitico);
            float escala = fmaxf(1e-3f, fabsf(gradNumerico));
            float erroRel = erro / escala;
            if(erroRel > maxErroRel) maxErroRel = erroRel;
            amostras++;
        }
    }
    printf("[Grad numerico] amostras=%d, maior erro relativo=%.6f\n", amostras, maxErroRel);
    bool gradOk = maxErroRel < 0.05f; // 5% de tolerancia (fp32 + GELU/softmax nao-linear)
    printf("Gradiente bate com diferencas finitas? %s\n\n", gradOk ? "SIM (OK)" : "NAO (FALHOU)");

    printf("=== RESUMO ===\n");
    printf("Forward causal:  %s\n", forwardCausal ? "OK" : "FALHOU");
    printf("Backward causal: %s\n", backwardCausal ? "OK" : "FALHOU");
    printf("Grad numerico:   %s\n", gradOk ? "OK" : "FALHOU");

    bool tudoOk = forwardCausal && backwardCausal && gradOk;
    printf("\nRESULTADO GERAL: %s\n", tudoOk ? "PASSOU" : "FALHOU");

    free(entrada); free(saida1); free(saida2); free(entradaMod);
    free(gradSaidaA); free(gradSaidaB); free(gradEntradaA); free(gradEntradaB);
    free(bufPerda);
    return tudoOk ? 0 : 1;
}
