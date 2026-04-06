// modelo.cpp
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <memory>
#include <chrono>

using namespace std;

#include "biblis/util.h"
#include "biblis/ativas.h"
#include "biblis/otimis/otimizador.h"
#include "biblis/otimis/adamw.h"
#include "biblis/camadas/camada.h"
#include "biblis/camadas/embedding.h"
#include "biblis/camadas/posicional.h"
#include "biblis/camadas/norm.h"
#include "biblis/camadas/densa.h"
#include "biblis/camadas/perda.h"
#include "biblis/tokes/bpe.h"
#include "biblis/camadas/multicabeca.h"
#include "biblis/camadas/bloco2.h"

struct ConfigModelo {
    size_t dim        = 128;
    size_t numCabecas = 4;
    size_t numBlocos  = 2;
    size_t dimOculta  = 512;
    size_t seqMax     = 256;
    size_t vocabTam   = 0;
    float  taxa       = 3e-4f;
    float  pesoDecay  = 0.01f;
};

class Modelo {
public:
    ConfigModelo cfg;

    Embedding emb;
    CamadaPosicional posicional;
    vector<unique_ptr<BlocoTransformerV2>> blocos;
    CamadaNorm normFinal;
    Densa projecao;
    CamadaPerda perda;

    TokenizadorBPE tok;

    Modelo(const ConfigModelo& c, TokenizadorBPE tokenizador)
        : cfg(c),
          emb(c.vocabTam, c.dim, "emb"),
          posicional(c.dim, c.seqMax, false, "pos"),
          normFinal(c.dim, 1e-5f, "norm_final"),
          projecao(c.dim, c.vocabTam, "linear", true, "proj"),
          tok(std::move(tokenizador)) {

        for(size_t i = 0; i < c.numBlocos; i++) {
            blocos.push_back(make_unique<BlocoTransformerV2>(
                c.dim, c.numCabecas, c.dimOculta, "gelu",
                "bloco" + to_string(i)));
        }
        _defOtimizadores();
        _imprimirInfo();
    }

    vector<vector<float>> propLote(const vector<size_t>& ids) {
        size_t T = ids.size();

        vector<vector<float>> x(T, vector<float>(cfg.dim));
        for(size_t t = 0; t < T; t++) {
            vector<float> e = emb.prop(ids[t]);
            x[t] = posicional.prop(e, t);
        }

        for(auto& bloco : blocos)
            x = bloco->propLote(x);

        x = normFinal.propLote(x);

        vector<vector<float>> logits(T);
        for(size_t t = 0; t < T; t++)
            logits[t] = projecao.prop(x[t]);

        return logits;
    }

    float treinarSequencia(const vector<size_t>& ids, size_t tamLote = 1) {
        if(ids.size() < 2) return 0.0f;

        vector<size_t> entrada(ids.begin(), ids.begin() + ids.size() - 1);
        vector<size_t> alvos(ids.begin() + 1, ids.end());
        size_t T = entrada.size();

        vector<size_t> idsEntrada(T);
        vector<vector<float>> x(T, vector<float>(cfg.dim));
        for(size_t t = 0; t < T; t++) {
            idsEntrada[t] = entrada[t];
            vector<float> e = emb.prop(entrada[t]);
            x[t] = posicional.prop(e, t);
        }

        for(auto& bloco : blocos)
            x = bloco->propLote(x);

        x = normFinal.propLote(x);

        vector<vector<float>> logits(T);
        for(size_t t = 0; t < T; t++)
            logits[t] = projecao.prop(x[t]);

        float perdaTotal = 0.0f;
        vector<vector<float>> gradX(T, vector<float>(cfg.dim, 0.0f));

        for(size_t t = 0; t < T; t++) {
            float p = perda.prop(logits[t], alvos[t]);
            perdaTotal += p;
            vector<float> gLogits = perda.retroprop();
            for(size_t i = 0; i < cfg.vocabTam; i++)
                gLogits[i] /= (float)(T * tamLote);
            auto gProj = projecao.retroprop(gLogits);
            gradX[t] = gProj.vetor;
        }
        perdaTotal /= (float)T;

        gradX = normFinal.retropropLote(gradX);

        for(int b = (int)cfg.numBlocos - 1; b >= 0; b--)
            gradX = blocos[b]->retropropLote(gradX);

        for(size_t t = 0; t < T; t++) {
            emb.idCache = idsEntrada[t];
            emb.retroprop(gradX[t]);
        }

        return perdaTotal;
    }

    void attPesos() {
        emb.att(cfg.taxa);
        for(auto& bloco : blocos) bloco->att(cfg.taxa);
        normFinal.att(cfg.taxa);
        projecao.att(cfg.taxa);
    }

    void zerarGradientes() {
        emb.zerarGradientes();
        for(auto& bloco : blocos) bloco->zerarGradientes();
        normFinal.zerarGradientes();
        projecao.zerarGradientes();
    }

    string gerar(const string& prompt, size_t maxNovos = 100,
                 float temperatura = 0.8f, size_t topK = 40) {

        vector<size_t> ids = _codificar(prompt);
        if(ids.empty()) ids.push_back(0);

        mt19937 rng(random_device{}());

        for(size_t n = 0; n < maxNovos; n++) {
            vector<size_t> janela(ids);
            if(janela.size() > cfg.seqMax)
                janela = vector<size_t>(ids.end() - cfg.seqMax, ids.end());

            auto logits = propLote(janela);
            vector<float> ultimoLogit = logits.back();

            if(temperatura != 1.0f)
                for(float& v : ultimoLogit) v /= temperatura;

            if(topK > 0 && topK < ultimoLogit.size()) {
                vector<float> copia = ultimoLogit;
                sort(copia.begin(), copia.end(), greater<float>());
                float limiar = copia[topK - 1];
                for(float& v : ultimoLogit)
                    if(v < limiar) v = -1e9f;
            }

            vector<float> probs = softmax(ultimoLogit);
            discrete_distribution<size_t> dist(probs.begin(), probs.end());
            size_t proximo = dist(rng);

            if(proximo == 2) break;
            ids.push_back(proximo);
        }

        vector<size_t> gerados(ids.begin() + _codificar(prompt).size(), ids.end());
        vector<int> geradosInt(gerados.begin(), gerados.end());
        return tok.decodificar(geradosInt);
    }

    size_t numParametros() const {
        size_t total = emb.numParametros() + normFinal.numParametros() + projecao.numParametros();
        for(auto& b : blocos) total += b->numParametros();
        return total;
    }

    void salvar(const string& dir) {
        _criarDir(dir);
        emb.salvar(dir + "/emb.bin");
        posicional.salvar(dir + "/pos.bin");
        for(size_t i = 0; i < blocos.size(); i++)
            blocos[i]->salvar(dir + "/bloco" + to_string(i));
        normFinal.salvar(dir + "/norm_final.bin");
        projecao.salvar(dir + "/proj.bin");
    }

    void carregar(const string& dir) {
        emb.carregar(dir + "/emb.bin");
        posicional.carregar(dir + "/pos.bin");
        for(size_t i = 0; i < blocos.size(); i++)
            blocos[i]->carregar(dir + "/bloco" + to_string(i));
        normFinal.carregar(dir + "/norm_final.bin");
        projecao.carregar(dir + "/proj.bin");
    }

private:
    void _defOtimizadores() {
        emb.defOtimizador(make_unique<AdamW>(cfg.taxa, 0.9f, 0.999f, 1e-8f, cfg.pesoDecay));
        normFinal.defOtimizador(make_unique<AdamW>(cfg.taxa, 0.9f, 0.999f, 1e-8f, 0.0f));
        projecao.defOtimizador(make_unique<AdamW>(cfg.taxa, 0.9f, 0.999f, 1e-8f, cfg.pesoDecay));
        for(auto& bloco : blocos) {
            bloco->norm1.defOtimizador(make_unique<AdamW>(cfg.taxa, 0.9f, 0.999f, 1e-8f, 0.0f));
            bloco->norm2.defOtimizador(make_unique<AdamW>(cfg.taxa, 0.9f, 0.999f, 1e-8f, 0.0f));
            bloco->oculta1.defOtimizador(make_unique<AdamW>(cfg.taxa, 0.9f, 0.999f, 1e-8f, cfg.pesoDecay));
            bloco->oculta2.defOtimizador(make_unique<AdamW>(cfg.taxa, 0.9f, 0.999f, 1e-8f, cfg.pesoDecay));

            vector<unique_ptr<Otimizador>> oQ, oK, oV;
            for(size_t h = 0; h < cfg.numCabecas; h++) {
                oQ.push_back(make_unique<AdamW>(cfg.taxa, 0.9f, 0.999f, 1e-8f, cfg.pesoDecay));
                oK.push_back(make_unique<AdamW>(cfg.taxa, 0.9f, 0.999f, 1e-8f, cfg.pesoDecay));
                oV.push_back(make_unique<AdamW>(cfg.taxa, 0.9f, 0.999f, 1e-8f, cfg.pesoDecay));
            }
            bloco->atencao.defOtimizadores(
                std::move(oQ), std::move(oK), std::move(oV),
                make_unique<AdamW>(cfg.taxa, 0.9f, 0.999f, 1e-8f, cfg.pesoDecay));
        }
    }

    void _imprimirInfo() const {
        printf("=== Modelo ===\n");
        printf("  dim=%zu  cabecas=%zu  blocos=%zu  oculta=%zu  seqMax=%zu\n",
               cfg.dim, cfg.numCabecas, cfg.numBlocos, cfg.dimOculta, cfg.seqMax);
        printf("  vocab=%zu\n", cfg.vocabTam);
        printf("  parametros: %zu\n", numParametros());
        printf("==============\n");
    }

    vector<size_t> _codificar(const string& texto) const {
        vector<int> ids = const_cast<TokenizadorBPE&>(tok).codificar(texto);
        return vector<size_t>(ids.begin(), ids.end());
    }

    void _criarDir(const string& caminho) const {
        system(("mkdir -p \"" + caminho + "\" 2>/dev/null").c_str());
    }
};

// ============================================================
// utilitários de treino
// ============================================================

string carregarTexto(const string& caminho) {
    ifstream a(caminho);
    if(!a) throw runtime_error("Não foi possível abrir: " + caminho);
    ostringstream ss;
    ss << a.rdbuf();
    return ss.str();
}

// separa por linha — nenhuma sequência cruza fronteira de conversa
vector<vector<size_t>> prepararSequencias(
    const string& texto,
    TokenizadorBPE& tok,
    size_t seqMax)
{
    vector<vector<size_t>> seqs;
    istringstream iss(texto);
    string linha;
    while(getline(iss, linha)) {
        if(linha.size() < 4) continue;
        vector<int> ids = tok.codificar(linha);
        if((size_t)ids.size() < 2) continue;
        for(size_t i = 0; i + 1 < ids.size(); i += seqMax) {
            size_t fim = min(i + seqMax + 1, ids.size());
            if(fim - i < 2) continue;
            vector<size_t> seq;
            for(size_t j = i; j < fim; j++)
                seq.push_back((size_t)ids[j]);
            seqs.push_back(seq);
        }
    }
    return seqs;
}

template<typename T>
void embaralhar(vector<T>& v) {
    mt19937 rng(random_device{}());
    shuffle(v.begin(), v.end(), rng);
}

// ============================================================
// main
// ============================================================
int main() {
    const string arquivoTexto  = "dados.txt";
    const string arquivoMerges = "bpe_merges.txt";
    const string dirModelo     = "checkpoint";
    const int    numMerges     = 3000;
    const int    numEpocas     = 3;
    const size_t tamLote       = 4;

    printf("=== Carregando texto ===\n");
    string texto = carregarTexto(arquivoTexto);
    printf("Texto: %zu bytes\n", texto.size());

    // --------------------------------------------------------
    // BPE: treina ou carrega merges
    // --------------------------------------------------------
    TreinadorBPE treinadorBpe;
    {
        ifstream teste(arquivoMerges);
        if(teste) {
            printf("=== Carregando BPE de '%s' ===\n", arquivoMerges.c_str());
            treinadorBpe.carregar(arquivoMerges);
        } else {
            printf("=== Treinando BPE (%d merges) ===\n", numMerges);
            vector<string> paragrafos;
            istringstream iss(texto);
            string linha;
            string paragrafo;
            while(getline(iss, linha)) {
                if(linha.empty()) {
                    if(!paragrafo.empty()) {
                        paragrafos.push_back(paragrafo);
                        paragrafo.clear();
                    }
                } else {
                    paragrafo += linha + " ";
                }
            }
            if(!paragrafo.empty()) paragrafos.push_back(paragrafo);
            treinadorBpe.treinar(paragrafos, numMerges);
            treinadorBpe.salvar(arquivoMerges);
        }
    }

    TokenizadorBPE tok(treinadorBpe.merges);
    {
        string amostra = texto.substr(0, min(texto.size(), (size_t)500000));
        vector<string> textos = {amostra};
        tok.construirVocab(textos);
    }

    // --------------------------------------------------------
    // monta modelo
    // --------------------------------------------------------
    ConfigModelo cfg;
    cfg.dim        = 64;
    cfg.numCabecas = 2;
    cfg.numBlocos  = 1;
    cfg.dimOculta  = 64;
    cfg.seqMax     = 64;
    cfg.taxa       = 3e-4f;
    cfg.pesoDecay  = 0.01f;
    cfg.vocabTam   = (size_t)tok.vocabTam();

    Modelo modelo(cfg, tok);

    {
        ifstream teste(dirModelo + "/emb.bin");
        if(teste) {
            printf("=== Checkpoint encontrado, carregando ===\n");
            modelo.carregar(dirModelo);
        }
    }

    // --------------------------------------------------------
    // prepara sequências por linha (sem cruzar contextos)
    // --------------------------------------------------------
    auto sequencias = prepararSequencias(texto, tok, cfg.seqMax);
    printf("Sequências de treino: %zu\n", sequencias.size());

    // --------------------------------------------------------
    // loop de treino
    // --------------------------------------------------------
    printf("=== Iniciando treino ===\n");
    size_t totalStepsGlobal = 0;
    for(int epoca = 0; epoca < numEpocas; epoca++) {
        embaralhar(sequencias);

        float perdaEpoca = 0.0f;
        size_t totalSteps = 0;
        size_t numSeqs = sequencias.size();
        size_t totalLotes = (numSeqs + tamLote - 1) / tamLote;

        auto inicioEpoca = chrono::steady_clock::now();

        for(size_t i = 0; i < numSeqs; i += tamLote) {
            size_t fim = min(i + tamLote, numSeqs);

            auto inicioBatch = chrono::steady_clock::now();

            float perdaLote = 0.0f;
            size_t contLote = 0;
            for(size_t k = i; k < fim; k++) {
                float p = modelo.treinarSequencia(sequencias[k], tamLote);
                if(!isfinite(p)) {
                    printf("  [AVISO] perda nao finita — pulando\n");
                    modelo.zerarGradientes();
                    goto proximoLote;
                }
                perdaLote += p;
                contLote++;
            }

            modelo.attPesos();
            modelo.zerarGradientes();
            perdaLote /= (float)contLote;
            perdaEpoca += perdaLote;
            totalSteps++;
            totalStepsGlobal++;

            {
                auto agora = chrono::steady_clock::now();
                float msBatch = chrono::duration<float, milli>(agora - inicioBatch).count();
                float sEpoca  = chrono::duration<float>(agora - inicioEpoca).count();
                float etaSeg  = (totalLotes - totalSteps) * (sEpoca / totalSteps);
                printf("ep%d  %zu/%zu  perda=%.4f  %.0fms/lote  ETA=%.0fs\n",
                       epoca + 1, totalSteps, totalLotes,
                       perdaLote, msBatch, etaSeg);
                fflush(stdout);
            }

            proximoLote:;
        }

        perdaEpoca /= (float)max(totalSteps, (size_t)1);
        printf("=== Época %d — perda média: %.4f ===\n", epoca + 1, perdaEpoca);

        modelo.salvar(dirModelo);

        printf("--- Amostra ---\n");
        string saida = modelo.gerar("Era uma vez", 60, 0.8f, 40);
        printf("\"Era uma vez%s\"\n", saida.c_str());
        printf("---------------\n\n");
    }

    printf("=== Modo interativo (enter vazio pra sair) ===\n");
    string entrada;
    while(true) {
        printf("> ");
        fflush(stdout);
        if(!getline(cin, entrada) || entrada.empty()) break;
        string resposta = modelo.gerar(entrada, 120, 0.8f, 40);
        printf("%s%s\n\n", entrada.c_str(), resposta.c_str());
    }

    return 0;
}