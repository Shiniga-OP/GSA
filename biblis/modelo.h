// biblis/modelo.h
#pragma once
#include "camadas/transformer.h"
#include <algorithm>
#include <chrono>

class Modelo {
public:
    TokenizadorBPE& tok;
    Transformer net;
    size_t epocas;
    float taxa;
    size_t logACada;

    Modelo(TokenizadorBPE& tok, size_t dim, size_t dimAtencao, size_t numBlocos,
        size_t seqMax=512, size_t dimFFN=0, const string& ativ="relu",
        float taxa=1e-3f, size_t epocas=10, size_t logACada=100)
        : tok(tok),
          net(tok.vocabTam(),dim,dimAtencao,numBlocos,seqMax,dimFFN,ativ,taxa),
          epocas(epocas), taxa(taxa), logACada(logACada) {}

    void treinar(const vector<string>& corpus, const string& checkpointDir=""){
        vector<vector<int>> seqs;
        for(const auto& texto:corpus){
            vector<int> ids=tok.codificar(texto);
            if(ids.size()<2) continue;
            ids.insert(ids.begin(),net.idAlmo);
            ids.push_back(net.idFim);
            seqs.push_back(ids);
        }
        if(seqs.empty()){printf("[Modelo]: corpus vazio\n");return;}
        printf("[Modelo]: %zu sequencias, %zu parametros\n",seqs.size(),net.numParametros());
        mt19937 rng(42);
        for(size_t ep=0;ep<epocas;ep++){
            shuffle(seqs.begin(),seqs.end(),rng);
            float perdaEpoca=0.0f; size_t total=0;
            auto t0=chrono::steady_clock::now();
            
            for(size_t s=0;s<seqs.size();s++){
                const auto& seq=seqs[s];
                vector<int> entrada(seq.begin(),seq.end()-1);
                vector<int> alvo(seq.begin()+1,seq.end());
                float p=net.treinarSequencia(entrada,alvo,taxa);
                perdaEpoca+=p; total++;
                if(logACada>0&&(s+1)%logACada==0) {
                    printf("  ep %zu  seq %zu/%zu  perda=%.4f\n",ep+1,s+1,seqs.size(),perdaEpoca/total);
                }
            }
            auto t1=chrono::steady_clock::now();
            float seg=chrono::duration<float>(t1-t0).count();
            printf("[ep %zu/%zu] perda=%.4f  tempo=%.1fs\n",ep+1,epocas,perdaEpoca/total,seg);
            if(!checkpointDir.empty()){
                string dir=checkpointDir+"/ep"+to_string(ep+1);
                net.salvar(dir);
                printf("[Modelo]: checkpoint salvo em %s\n",dir.c_str());
            }
        }
    }

    string gerar(const string& comando, size_t maxNovos=64, float temp=1.0f){
        return net.gerarTexto(tok,comando,maxNovos,temp);
    }

    void salvar(const string& dir) {
        net.salvar(dir);
    }
    void carregar(const string& dir) {
        net.carregar(dir);
    }
};