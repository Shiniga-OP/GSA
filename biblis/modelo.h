// biblis/modelo.h
#pragma once
#include <algorithm>
#include <chrono>
#include <fstream>
#include "camadas/bloco.h"
#include "camadas/embedding.h"
#include "camadas/posicional.h"
#include "camadas/densa.h"
#include "camadas/perda.h"
#include "tokes/bpe.h"
#include "otimis/adamw.h"

class Modelo {
public:
    size_t dim, dimAtencao, dimOculta, numBlocos, seqMax, vocabTam;
    Embedding embedding;
    CamadaPosicional posicional;
    vector<unique_ptr<BlocoTransformer>> blocos;
    Densa projecao;
    CamadaPerda perda;
    int idAlmo, idFim;
    TokenizadorBPE& tok;
    size_t epocas;
    float taxa;
    size_t logACada;

    Modelo(TokenizadorBPE& tok, size_t dim, size_t dimAtencao, size_t numBlocos,
        size_t seqMax=512, size_t dimOculta=0, const string& ativ="relu",
        float taxa=1e-3f, size_t epocas=10, size_t logACada=100)
        :
          tok(tok),
          dim(dim), dimAtencao(dimAtencao), dimOculta(dimOculta>0?dimOculta:4*dim),
          numBlocos(numBlocos), seqMax(seqMax), vocabTam(tok.vocabTam()),
          embedding(tok.vocabTam(), dim, "embedding"),
          posicional(dim, seqMax, false, "posicional"),
          projecao(dim, tok.vocabTam(), "linear", false, "projecao"),
          idAlmo(0), idFim(2),
          epocas(epocas), taxa(taxa), logACada(logACada)
    {
        size_t oculta = dimOculta > 0 ? dimOculta : 4 * dim;
        for(size_t i = 0; i < numBlocos; i++) {
            auto bloco = make_unique<BlocoTransformer>(dim, dimAtencao, oculta, ativ, "bloco"+to_string(i));
            bloco->defOtimizadores(
                make_unique<AdamW>(taxa),
                make_unique<AdamW>(taxa),
                make_unique<AdamW>(taxa),
                make_unique<AdamW>(taxa),
                make_unique<AdamW>(taxa)
            );
            blocos.push_back(std::move(bloco));
        }
        embedding.defOtimizador(make_unique<AdamW>(taxa));
        projecao.defOtimizador(make_unique<AdamW>(taxa));
    }

    // treino token a token(legado, usado por treinar())
    float treinarSequencia(const vector<int>& ids, const vector<int>& alvos, float taxa) {
        if(ids.empty() || ids.size() != alvos.size()) {
            throw invalid_argument("[Transformer]: ids e alvos devem ter o mesmo tamanho");
        }
        size_t T = ids.size();
        float perdaTotal = 0.0f;
        vector<vector<float>> xs;
        xs.reserve(T);

        for(size_t t = 0; t < T; t++) {
            _zerarGradientes();

            vector<float> x = embedding.prop((size_t)ids[t]);
            x = posicional.prop(x, t);
            xs.push_back(x);

            for(auto& b : blocos) x = b->prop(x, xs);

            vector<float> logits = projecao.prop(x);
            float p = perda.prop(logits, (size_t)alvos[t]);
            perdaTotal += p;

            vector<float> gradLogits = perda.retroprop();
            auto gProj = projecao.retroprop(gradLogits);
            vector<float> grad = gProj.vetor;

            for(int b = (int)numBlocos-1; b >= 0; b--) {
                grad = blocos[b]->retroprop(grad).vetor;
            }
            posicional.retroprop(grad);
            embedding.retroprop(grad);

            _att(taxa);
        }
        return perdaTotal / (float)T;
    }

    // treinarLote: processa sequencia inteira [T x dim] de uma vez
    // retorna perda média
    float treinarLote(const vector<int>& ids, const vector<int>& alvos, float taxaAtual) {
        size_t T = ids.size();
        if(T == 0 || T != alvos.size()) {
            throw invalid_argument("[Modelo]: ids e alvos devem ter o mesmo tamanho");
        }
        _zerarGradientes();

        // embedding + posicional -> [T x dim]
        vector<vector<float>> X(T, vector<float>(dim));
        for(size_t t = 0; t < T; t++) {
            vector<float> e = embedding.prop((size_t)ids[t]);
            X[t] = posicional.prop(e, t < seqMax ? t : seqMax - 1);
        }
        // propaga pelos blocos
        for(auto& b : blocos) X = b->propLote(X);

        // projeção + perda por token
        float perdaTotal = 0.0f;
        vector<vector<float>> gradX(T, vector<float>(dim, 0.0f));

        for(size_t t = 0; t < T; t++) {
            vector<float> logits = projecao.prop(X[t]);
            perdaTotal += perda.prop(logits, (size_t)alvos[t]);
            vector<float> gLogits = perda.retroprop();
            gradX[t] = projecao.retroprop(gLogits).vetor;
        }
        // retropropaga pelos blocos
        for(int b = (int)numBlocos - 1; b >= 0; b--) {
            gradX = blocos[b]->retropropLote(gradX);
        }
        // retropropaga embedding + posicional
        for(size_t t = 0; t < T; t++) {
            posicional.retroprop(gradX[t]);
            embedding.retroprop(gradX[t]);
        }
        _att(taxaAtual);
        return perdaTotal / (float)T;
    }

    // treinarArquivo: pré-treino com arquivo de texto grande
    // le em streaming, fatiado em janelas de seqMax tokens
    // agenda: aquecimento linear por aquecimentoPassos, depois decaimento cosseno
    void treinarArquivo(
        const string& caminhoArquivo,
        size_t epocasArq = 1, size_t tamanhoJanela = 0,
        size_t passo = 0, size_t aquecimentoPassos = 1000,
        float taxaMin = 1e-5f, const string& salvaDir = "",
        size_t salvaACada = 0, size_t amostraACada = 0, // gera amostra a cada N janelas(0=nunca)
        size_t logACadaJanela = 10  // override local de logACada
    ) {
        size_t janela = tamanhoJanela > 0 ? tamanhoJanela : seqMax;
        size_t passoEfetivo = passo > 0 ? passo : janela / 2;
        if(passoEfetivo == 0) passoEfetivo = 1;
        passo = passoEfetivo;
        
        printf("[Modelo]: carregando arquivo %s\n", caminhoArquivo.c_str()); fflush(stdout);
        
        ifstream arq(caminhoArquivo);
        if(!arq) throw runtime_error("[Modelo]: não foi possível abrir " + caminhoArquivo);
        string texto((istreambuf_iterator<char>(arq)), istreambuf_iterator<char>());
        arq.close();
        
        printf("[Modelo]: %zu bytes lidos, tokenizando...\n", texto.size()); fflush(stdout);
        
        vector<int> tokens = tok.codificar(texto);
        printf("[Modelo]: %zu tokens, %zu parâmetros\n", tokens.size(), numParametros()); fflush(stdout);
        
        if(tokens.size() < janela + 1) {
            printf("[Modelo]: arquivo muito pequeno para janela %zu\n", janela); fflush(stdout);
            return;
        }
        vector<pair<size_t,size_t>> janelas;
        for(size_t i = 0; i + janela < tokens.size(); i += passo) {
            janelas.emplace_back(i, i + janela);
        }
        printf("[Modelo]: %zu janelas de %zu tokens (passo=%zu)\n",
        janelas.size(), janela, passo); fflush(stdout);
        
        size_t passoGlobal = 0;
        size_t totalPassos = epocasArq * janelas.size();
        mt19937 rng(42);
        
        for(size_t ep = 0; ep < epocasArq; ep++) {
            shuffle(janelas.begin(), janelas.end(), rng);
            
            float perdaAcum = 0.0f;
            size_t total = 0;
            float perdaMedia = 0.0f;
            auto t0 = chrono::steady_clock::now();
            auto tUlt = t0;
            
            for(size_t j = 0; j < janelas.size(); j++) {
                auto [ini, fim] = janelas[j];
                
                vector<int> entrada(tokens.begin() + ini, tokens.begin() + fim);
                vector<int> alvo(tokens.begin() + ini + 1, tokens.begin() + fim + 1);
                
                float taxaAtual = _agendar(passoGlobal, totalPassos, aquecimentoPassos, taxa, taxaMin);
                
                float p = treinarLote(entrada, alvo, taxaAtual);
                perdaAcum += p;
                total++;
                passoGlobal++;
                perdaMedia = perdaAcum / (float)total;
                
                // log por janela
                size_t logN = logACadaJanela > 0 ? logACadaJanela : (logACada > 0 ? logACada : 10);
                if((j + 1) % logN == 0 || j + 1 == janelas.size()) {
                    auto agora = chrono::steady_clock::now();
                    float segTotal = chrono::duration<float>(agora - t0).count();
                    float toksPorSeg = (float)(total * janela) / segTotal;
                    
                    // ETA
                    size_t jRestante = janelas.size() - (j + 1);
                    float segPorJanela = segTotal / (float)(j + 1);
                    float eta = segPorJanela * (float)jRestante;
                    
                    // barra simples [====>....]
                    int barW = 20;
                    float frac = (float)(j + 1) / (float)janelas.size();
                    int preenchido = (int)(frac * barW);
                    char bar[32];
                    
                    for(int k = 0; k < barW; k++) {
                        bar[k] = k < preenchido ? '=' : (k == preenchido ? '>' : '.');
                    }
                    bar[barW] = '\0';
                    
                    printf("época%zu [%s] %zu/%zu  perda=%.4f  taxa=%.2e  %.0ftok/s  ETA=%.0fs\n",
                    ep+1, bar, j+1, janelas.size(),
                    perdaMedia, taxaAtual, toksPorSeg, eta);
                    fflush(stdout);
                    tUlt = agora;
                }
                // amostra de geração
                if(amostraACada > 0 && (j + 1) % amostraACada == 0) {
                    printf("  [amostra] comando='<|usr|>: Olá'\n");
                    string s = gerar("<|usr|>: Olá", 40, 0.8f);
                    printf("  %s\n\n", s.c_str());
                    fflush(stdout);
                }
                if(salvaACada > 0 && passoGlobal % salvaACada == 0 && !salvaDir.empty()) {
                    string dir = salvaDir + "/passo" + to_string(passoGlobal);
                    salvar(dir);
                    printf("[Modelo]: salvo em %s\n", dir.c_str()); fflush(stdout);
                }
            }
            auto t1 = chrono::steady_clock::now();
            float seg = chrono::duration<float>(t1 - t0).count();
            printf("[época %zu/%zu] perda=%.4f  tempo=%.1fs\n",
            ep+1, epocasArq, perdaMedia, seg); fflush(stdout);
            
            if(!salvaDir.empty()) {
                string dir = salvaDir + "/ep" + to_string(ep+1);
                salvar(dir);
                printf("[Modelo]: salvo em %s\n", dir.c_str());
                fflush(stdout);
            }
        }
    }
    // refinamento: igual treinarArquivo mas com corpus de frases
    void refinar(
        const vector<string>& corpus,
        size_t epocasFt = 5,
        float taxaFt = 1e-4f,
        size_t aquecimentoPassos = 100,
        const string& salvaDir = ""
    ) {
        vector<vector<int>> seqs;
        for(const auto& texto : corpus) {
            vector<int> ids = tok.codificar(texto);
            if(ids.size() < 2) continue;
            ids.insert(ids.begin(), idAlmo);
            ids.push_back(idFim);
            if(ids.size() > seqMax) ids.resize(seqMax + 1);
            seqs.push_back(ids);
        }
        if(seqs.empty()) { printf("[Modelo]: corpus vazio\n"); return; }

        printf("[Modelo]: refinamento %zu sequências, %zu parâmetros\n",
            seqs.size(), numParametros());

        size_t passoGlobal = 0;
        size_t totalPassos = epocasFt * seqs.size();
        mt19937 rng(42);

        for(size_t ep = 0; ep < epocasFt; ep++) {
            shuffle(seqs.begin(), seqs.end(), rng);
            float perdaEpoca = 0.0f;
            size_t total = 0;
            auto t0 = chrono::steady_clock::now();

            for(size_t s = 0; s < seqs.size(); s++) {
                const auto& seq = seqs[s];
                vector<int> entrada(seq.begin(), seq.end() - 1);
                vector<int> alvo(seq.begin() + 1, seq.end());

                float taxaAtual = _agendar(passoGlobal, totalPassos, aquecimentoPassos, taxaFt, taxaFt * 0.1f);
                float p = treinarLote(entrada, alvo, taxaAtual);
                perdaEpoca += p;
                total++;
                passoGlobal++;

                if(logACada > 0 && (s + 1) % logACada == 0)
                    printf("  época %zu  seq %zu/%zu  perda=%.4f  taxa=%.2e\n",
                        ep+1, s+1, seqs.size(), perdaEpoca/total, taxaAtual);
            }
            auto t1 = chrono::steady_clock::now();
            float seg = chrono::duration<float>(t1 - t0).count();
            printf("[rf época %zu/%zu] perda=%.4f  tempo=%.1fs\n",
                ep+1, epocasFt, perdaEpoca/total, seg);

            if(!salvaDir.empty()) {
                string dir = salvaDir + "/rf_ep" + to_string(ep+1);
                salvar(dir);
            }
        }
    }

    // geração(interface token a token)
    vector<int> gerarVetores(const vector<int>& entrada, size_t maxNovos=64, float temp=1.0f) {
        vector<int> saida(entrada.begin(), entrada.end());

        vector<vector<float>> xs;
        xs.reserve(entrada.size() + maxNovos);
        for(size_t i = 0; i < entrada.size(); i++) {
            vector<float> xp = embedding.prop((size_t)entrada[i]);
            xp = posicional.prop(xp, i < seqMax ? i : seqMax-1);
            xs.push_back(xp);
        }
        for(size_t passo = 0; passo < maxNovos; passo++) {
            size_t pos = saida.size() - 1;
            if(pos >= seqMax) pos = seqMax - 1;
            vector<float> x = embedding.prop((size_t)saida.back());
            x = posicional.prop(x, pos);
            xs.push_back(x);

            vector<vector<float>>* pxs = &xs;
            vector<vector<float>> janela;
            if(xs.size() > seqMax) {
                janela.assign(xs.end() - seqMax, xs.end());
                pxs = &janela;
            }
            vector<float> xf = x;
            for(auto& b : blocos) xf = b->prop(xf, *pxs);

            vector<float> logits = projecao.prop(xf);
            int proximo;
            if(temp <= 0.0f) proximo = argmax(logits);
            else {
                vector<float> probs = softmax(logits, temp);
                proximo = _amostrar(probs);
            }
            saida.push_back(proximo);
            if(proximo == idFim) break;
        }
        return saida;
    }

    string gerar(const string& entrada, size_t maxNovos=64, float temp=1.0f) {
        vector<int> ids = tok.codificar(entrada);
        ids.insert(ids.begin(), idAlmo);
        vector<int> res = gerarVetores(ids, maxNovos, temp);
        vector<int> limpo;
        for(int id : res) if(id != idAlmo && id != idFim) limpo.push_back(id);
        return tok.decodificar(limpo);
    }

    // treinar() legado
    void treinar(const vector<string>& corpus, const string& salvaDir="") {
        vector<vector<int>> seqs;
        for(const auto& texto : corpus) {
            vector<int> ids = tok.codificar(texto);
            if(ids.size() < 2) continue;
            ids.insert(ids.begin(), idAlmo);
            ids.push_back(idFim);
            seqs.push_back(ids);
        }
        if(seqs.empty()) { printf("[Modelo]: corpus vazio\n"); return; }

        printf("[Modelo]: %zu sequencias, %zu parametros\n", seqs.size(), numParametros());

        mt19937 rng(42);
        for(size_t ep = 0; ep < epocas; ep++) {
            shuffle(seqs.begin(), seqs.end(), rng);
            float perdaEpoca = 0.0f;
            size_t total = 0;
            auto t0 = chrono::steady_clock::now();

            for(size_t s = 0; s < seqs.size(); s++) {
                const auto& seq = seqs[s];
                vector<int> entrada(seq.begin(), seq.end() - 1);
                vector<int> alvo(seq.begin() + 1, seq.end());
                float p = treinarSequencia(entrada, alvo, taxa);
                perdaEpoca += p;
                total++;
                if(logACada > 0 && (s+1) % logACada == 0)
                    printf("  época %zu  seq %zu/%zu  perda=%.4f\n",
                        ep+1, s+1, seqs.size(), perdaEpoca/total);
            }
            auto t1 = chrono::steady_clock::now();
            float seg = chrono::duration<float>(t1 - t0).count();
            printf("[época %zu/%zu] perda=%.4f  tempo=%.1fs\n",
                ep+1, epocas, perdaEpoca/total, seg);

            if(!salvaDir.empty()) {
                string dir = salvaDir + "/ep" + to_string(ep+1);
                salvar(dir);
                printf("[Modelo]: salva salvo em %s\n", dir.c_str());
            }
        }
    }

    void salvar(const string& dir) const {
        _criarDir(dir);
        embedding.salvar(dir+"/embedding.bin");
        posicional.salvar(dir+"/posicional.bin");
        for(size_t i = 0; i < numBlocos; i++) blocos[i]->salvar(dir+"/bloco"+to_string(i));
        projecao.salvar(dir+"/projecao.bin");
    }

    void carregar(const string& dir) {
        embedding.carregar(dir+"/embedding.bin");
        posicional.carregar(dir+"/posicional.bin");
        for(size_t i = 0; i < numBlocos; i++) blocos[i]->carregar(dir+"/bloco"+to_string(i));
        projecao.carregar(dir+"/projecao.bin");
    }

    size_t numParametros() const {
        size_t t = embedding.numParametros() + posicional.numParametros() + projecao.numParametros();
        for(const auto& b : blocos) t += b->numParametros();
        return t;
    }

    // aquecimento linear até aquecimentoPassos, depois decaimento cosseno até taxaMin
    float _agendar(size_t passo, size_t total, size_t aquecimento, float taxaMax, float taxaMin) const {
        if(passo < aquecimento) {
            return taxaMin + (taxaMax - taxaMin) * ((float)passo / (float)(aquecimento > 0 ? aquecimento : 1));
        }
        float progresso = (float)(passo - aquecimento) / (float)(total > aquecimento ? total - aquecimento : 1);
        if(progresso >= 1.0f) return taxaMin;
        float cosseno = 0.5f * (1.0f + cos((float)M_PI * progresso));
        return taxaMin + (taxaMax - taxaMin) * cosseno;
    }

    void _att(float taxaAtual) {
        embedding.att(taxaAtual);
        posicional.att(taxaAtual);
        for(auto& b : blocos) b->att(taxaAtual);
        projecao.att(taxaAtual);
    }
    void _zerarGradientes() {
        embedding.zerarGradientes();
        posicional.zerarGradientes();
        for(auto& b : blocos) b->zerarGradientes();
        projecao.zerarGradientes();
    }
    int _amostrar(const vector<float>& probs) {
        static mt19937 gen(random_device{}());
        uniform_real_distribution<float> dis(0.0f, 1.0f);
        float r = dis(gen), acum = 0.0f;
        for(size_t i = 0; i < probs.size(); i++) { acum += probs[i]; if(r < acum) return (int)i; }
        return (int)probs.size() - 1;
    }
    void _criarDir(const string& caminho) const {
        #ifdef _WIN32
            system(("mkdir \""+caminho+"\" 2>nul").c_str());
        #else
            system(("mkdir -p \""+caminho+"\" 2>/dev/null").c_str());
        #endif
    }
};