// biblis/camadas/transformer.h
#pragma once
#include "bloco.h"
#include "embedding.h"
#include "posicional.h"
#include "densa.h"
#include "perda.h"
#include "../tokes/bpe.h"
#include "../otimis/adamw.h"

class Transformer {
public:
    size_t dim, dimAtencao, dimOculta, numBlocos, seqMax, vocabTam;
    Embedding embedding;
    CamadaPosicional posicional;
    vector<unique_ptr<BlocoTransformer>> blocos;
    Densa projecao;
    CamadaPerda perda;
    int idAlmo, idFim;
    float taxa;

    Transformer(size_t vocabTam, size_t dim, size_t dimAtencao, size_t numBlocos,
        size_t seqMax=512, size_t dimOculta=0, const string& ativ="relu", float taxa=1e-3f)
        : dim(dim), dimAtencao(dimAtencao), dimOculta(dimOculta>0?dimOculta:4*dim),
          numBlocos(numBlocos), seqMax(seqMax), vocabTam(vocabTam),
          embedding(vocabTam, dim, "embedding"),
          posicional(dim, seqMax, false, "posicional"),
          projecao(dim, vocabTam, "linear", false, "projecao"),
          idAlmo(0), idFim(2), taxa(taxa)
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

    float treinarSequencia(const vector<int>& ids, const vector<int>& alvos, float taxa) {
        if(ids.empty() || ids.size() != alvos.size()) {
            throw invalid_argument("[Transformer]: ids e alvos devem ter o mesmo tamanho");
        }
        size_t T = ids.size();
        float perdaTotal = 0.0f;
        
        for(size_t t = 0; t < T; t++) {
            _zerarGradientes(); // zera antes de cada passo
            
            vector<float> x = embedding.prop((size_t)ids[t]);
            x = posicional.prop(x, t);
            for(auto& b : blocos) x = b->prop(x);
            
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
            
            _att(taxa); // atualiza depois de cada passo
        }
        return perdaTotal / (float)T;
    }

    vector<int> gerar(const vector<int>& prompt, size_t maxNovos=64, float temp=1.0f) {
        vector<int> saida(prompt.begin(), prompt.end());
        for(size_t passo = 0; passo < maxNovos; passo++) {
            vector<float> x = embedding.prop((size_t)saida.back());
            size_t pos = saida.size() - 1;
            if(pos >= seqMax) pos = seqMax - 1;
            x = posicional.prop(x, pos);
            
            for(auto& b : blocos) x = b->prop(x);
            
            vector<float> logits = projecao.prop(x);
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

    string gerarTexto(TokenizadorBPE& tok, const string& entrada, size_t maxNovos=64, float temp=1.0f) {
        vector<int> ids = tok.codificar(entrada);
        ids.insert(ids.begin(), idAlmo);
        vector<int> res = gerar(ids, maxNovos, temp);
        vector<int> limpo;
        for(int id : res) if(id != idAlmo && id != idFim) limpo.push_back(id);
        return tok.decodificar(limpo);
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

    void _att(float taxa) {
        embedding.att(taxa);
        posicional.att(taxa);
        for(auto& b : blocos) b->att(taxa);
        projecao.att(taxa);
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