// treino_transformer.cpp
#include "biblis/camadas.h"
#include "biblis/ativas.h"
#include "biblis/util.h"
#include "biblis/otimis/adam.h"

// hiperparametros
static const size_t DIM = 64;
static const size_t DIM_AT = 32;
static const size_t VOCAB = 32;
static const size_t TAM_CTX = 8; // tamanho do contexto(janela)
static const size_t N_BLOCOS = 2; // numero de blocos transformer empilhados
static const size_t EPOCAS = 40;
static const float TAXA = 1e-3f; // adam tolera taxa menor

// encode posicional senoidal
// retorna vetor de tamanho DIM para a posição pos
vector<float> posicional(size_t pos) {
    vector<float> pe(DIM);
    for(size_t i = 0; i < DIM; i += 2) {
        float freq = 1.0f / powf(10000.0f, (float)i / (float)DIM);
        pe[i] = sinf((float)pos * freq);
        if(i + 1 < DIM)
            pe[i+1] = cosf((float)pos * freq);
    }
    return pe;
}

// soma dois vetores em dst
void somarIn(vector<float>& dst, const vector<float>& fonte) {
    for(size_t i = 0; i < dst.size(); i++) dst[i] += fonte[i];
}

// === bloco transformer ===
// agrupa as camadas de um bloco e expoe prop/retroprop
struct BlocoTransformer {
    CamadaNorm norm1;
    CamadaAtencao atencao;
    CamadaNorm  norm2;
    Densa ffnAlta;
    Densa ffnBaixa;

    BlocoTransformer(size_t id, float taxa = 1e-3f)
        : norm1(DIM, 1e-5f, "b" + to_string(id) + "_norm1"),
        atencao(DIM, DIM_AT, DIM, "b" + to_string(id) + "_at"),
        norm2(DIM, 1e-5f, "b" + to_string(id) + "_norm2"),
        ffnAlta(DIM, DIM * 4, "gelu", true, "b" + to_string(id) + "_ffnAlta"),
        ffnBaixa(DIM * 4, DIM, "linear", true, "b" + to_string(id) + "_ffnDn")
    {
        norm1.defOtimizador(unique_ptr<Otimizador>(new Adam(taxa)));
        norm2.defOtimizador(unique_ptr<Otimizador>(new Adam(taxa)));
        ffnAlta.defOtimizador(unique_ptr<Otimizador>(new Adam(taxa)));
        ffnBaixa.defOtimizador(unique_ptr<Otimizador>(new Adam(taxa)));
        atencao.defOtimizadores(
            unique_ptr<Otimizador>(new Adam(taxa)),
            unique_ptr<Otimizador>(new Adam(taxa)),
            unique_ptr<Otimizador>(new Adam(taxa))
        );
    }

    // prop: x -> saida do bloco
    // chaves: embeddings de contexto para a atenção
    vector<float> prop(const vector<float>& x, const vector<vector<float>>& chaves) {
        // sub-bloco 1: atenção + residual
        vector<float> xn1 = norm1.prop(x);
        vector<float> atSaida = atencao.prop(xn1, chaves);
        vector<float> x2(DIM);
        for(size_t i = 0; i < DIM; i++) x2[i] = x[i] + atSaida[i];

        // sub-bloco 2: FFN + residual
        vector<float> xn2 = norm2.prop(x2);
        vector<float> meio = ffnAlta.prop(xn2);
        vector<float> ffnS = ffnBaixa.prop(meio);
        vector<float> x3(DIM);
        for(size_t i = 0; i < DIM; i++) x3[i] = x2[i] + ffnS[i];

        return x3;
    }

    // retroprop: recebe gradiente de x3, propaga até x
    // retorna gradiente de x (entrada do bloco)
    vector<float> retroprop(const vector<float>& gradX3, const vector<float>& x,
    const vector<vector<float>>& chaves) {
        // precisamos re-calcular os intermediarios para retroprop
        // (as camadas guardam internamente o estado do ultimo prop)
        // sub-bloco 2 retroprop
        vector<float> gradFFNSaida = gradX3; // residual: grad passa direto
        vector<float> gradX2Res = gradX3; // residual: branch x2

        vector<float> gradMeio = ffnBaixa.retroprop(gradFFNSaida).vetor;
        vector<float> gradXn2 = ffnAlta.retroprop(gradMeio).vetor;
        vector<float> gradX2Norm = norm2.retroprop(gradXn2).vetor;

        vector<float> gradX2(DIM);
        for(size_t i = 0; i < DIM; i++) gradX2[i] = gradX2Res[i] + gradX2Norm[i];

        // sub-bloco 1 retroprop
        vector<float> gradAtSaida = gradX2;
        vector<float> gradXRes = gradX2;

        GradGenerico gAt = atencao.retroprop(gradAtSaida);
        vector<float> gradXn1 = norm1.retroprop(gAt.gradEstado).vetor;

        vector<float> gradX(DIM);
        for(size_t i = 0; i < DIM; i++) gradX[i] = gradXRes[i] + gradXn1[i];

        return gradX;
    }

    void att(float taxa) {
        norm1.att(taxa);
        atencao.att(taxa);
        norm2.att(taxa);
        ffnAlta.att(taxa);
        ffnBaixa.att(taxa);
    }

    void zerarGradientes() {
        norm1.zerarGradientes();
        atencao.zerarGradientes();
        norm2.zerarGradientes();
        ffnAlta.zerarGradientes();
        ffnBaixa.zerarGradientes();
    }
};

// dados sinteticos
// gera sequencias de tamanho TAM_CTX+1 com padrão periodico simples:
// token[t] = (token[t-1] + passo) % VOCAB
// isso é aprendível: dado o contexto, o modelo deve prever o proximo token
struct Dados {
    struct Amostra {
        vector<size_t> contexto; // TAM_CTX tokens
        size_t alvo; // próximo token
    };
    vector<Amostra> amostras;

    Dados() {
        // 4 sequencias com passos diferentes
        size_t passos[] = {1, 2, 3, 5};
        for(size_t p : passos) {
            for(size_t inicio = 0; inicio < VOCAB; inicio++) {
                Amostra a;
                for(size_t t = 0; t < TAM_CTX; t++)
                a.contexto.push_back((inicio + t * p) % VOCAB);
                a.alvo = (inicio + TAM_CTX * p) % VOCAB;
                amostras.push_back(a);
            }
        }
    }
};

// modelo
int main() {
    // camadas compartilhadas
    Embedding emb(VOCAB, DIM, "emb");
    emb.defOtimizador(unique_ptr<Otimizador>(new Adam(TAXA)));
    vector<BlocoTransformer*> blocos;
    for(size_t i = 0; i < N_BLOCOS; i++) blocos.push_back(new BlocoTransformer(i, TAXA));
    CamadaNorm normFinal(DIM, 1e-5f, "normFinal");
    normFinal.defOtimizador(unique_ptr<Otimizador>(new Adam(TAXA)));
    Densa projecao(DIM, VOCAB, "linear", true, "proj");
    projecao.defOtimizador(unique_ptr<Otimizador>(new Adam(TAXA)));
    CamadaPerda perda;

    Dados ds;
    size_t N = ds.amostras.size();

    printf("Transformer: %zu blocos, dim=%zu, ctx=%zu, vocab=%zu\n", N_BLOCOS, DIM, TAM_CTX, VOCAB);
    printf("Dados: %zu amostras\n\n", N);

    for(size_t ep = 0; ep < EPOCAS; ep++) {
        float perdaTotal = 0.0f;
        size_t acertos  = 0;

        for(const auto& am : ds.amostras) {
            // === prop ===

            // embedding do token atual(ultima posição do contexto)
            size_t tokAtual = am.contexto.back();
            vector<float> x = emb.prop(tokAtual);
            // soma encode posicional da posição TAM_CTX-1
            vector<float> pe = posicional(TAM_CTX - 1);
            somarIn(x, pe);

            // embeddings de contexto com encode posicional
            vector<vector<float>> chaves;
            for(size_t t = 0; t < am.contexto.size(); t++) {
                vector<float> ek = emb.prop(am.contexto[t]);
                vector<float> pek = posicional(t);
                somarIn(ek, pek);
                chaves.push_back(ek);
            }
            // propaga pelos N blocos
            // guardamos x de entrada de cada bloco para o retroprop
            vector<vector<float>> xsBlocos;
            xsBlocos.push_back(x);
            for(size_t b = 0; b < N_BLOCOS; b++) {
                x = blocos[b]->prop(xsBlocos.back(), chaves);
                xsBlocos.push_back(x);
            }
            vector<float> xFinal = normFinal.prop(x);
            vector<float> logits = projecao.prop(xFinal);

            float erro = perda.prop(logits, am.alvo);
            perdaTotal += erro;

            // acurácia
            size_t pred = (size_t)(max_element(logits.begin(), logits.end()) - logits.begin());
            if(pred == am.alvo) acertos++;

            // retroprop
            vector<float> gradLogits = perda.retroprop();
            vector<float> gradXFinal = projecao.retroprop(gradLogits).vetor;
            vector<float> gradX = normFinal.retroprop(gradXFinal).vetor;

            // retroprop pelos blocos na ordem inversa
            for(int b = (int)N_BLOCOS - 1; b >= 0; b--) {
                gradX = blocos[b]->retroprop(gradX, xsBlocos[b], chaves);
            }
            emb.retroprop(gradX);

            // atualização
            emb.att(TAXA);
            for(size_t b = 0; b < N_BLOCOS; b++) blocos[b]->att(TAXA);
            normFinal.att(TAXA);
            projecao.att(TAXA);

            emb.zerarGradientes();
            for(size_t b = 0; b < N_BLOCOS; b++) blocos[b]->zerarGradientes();
            normFinal.zerarGradientes();
            projecao.zerarGradientes();
        }
        float perdaMedia = perdaTotal / (float)N;
        float acuracia = (float)acertos / (float)N * 100.0f;
        printf("época %2zu | perda: %.4f | acuracia: %.1f%%\n", ep + 1, perdaMedia, acuracia);
    }
    // teste de geração
    printf("\n=== geração ===\n");
    // sequencia de teste: passo=1, inicio=0 -> 0,1,2,3,4,5,6,7 -> deve prever 8
    vector<size_t> seqTeste = {0, 1, 2, 3, 4, 5, 6, 7};
    size_t esperado = 8;

    size_t tokAtual = seqTeste.back();
    vector<float> x = emb.prop(tokAtual);
    somarIn(x, posicional(TAM_CTX - 1));

    vector<vector<float>> chaves;
    for(size_t t = 0; t < seqTeste.size(); t++) {
        vector<float> ek = emb.prop(seqTeste[t]);
        somarIn(ek, posicional(t));
        chaves.push_back(ek);
    }
    for(size_t b = 0; b < N_BLOCOS; b++) x = blocos[b]->prop(x, chaves);

    vector<float> xFinal = normFinal.prop(x);
    vector<float> logits = projecao.prop(xFinal);

    size_t pred = (size_t)(max_element(logits.begin(), logits.end()) - logits.begin());
    printf("contexto: [0,1,2,3,4,5,6,7]\n");
    printf("esperado: %zu | previsto: %zu | %s\n", esperado, pred, pred == esperado ? "OK" : "ERROU");

    for(size_t b = 0; b < N_BLOCOS; b++) delete blocos[b];
    return 0;
}