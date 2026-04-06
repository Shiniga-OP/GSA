// bpe.h
#pragma once
#include "../util.h"

// mapa hash string->int SEM posse de chaves
// chaves são ponteiros externos(vivem em outro buffer)
// usado no treino para evitar malloc/free por par
struct EntradaMapaRef {
    const char* chave;
    int tam;
    int valor;
};

struct MapaStrIntRef {
    EntradaMapaRef* slots;
    int capacidade;
    int tamanho;

    void iniciar(int cap = 1024) {
        capacidade = cap;
        tamanho = 0;
        slots = (EntradaMapaRef*)calloc(cap, sizeof(EntradaMapaRef));
    }

    void liberar() {
        free(slots);
        slots = nullptr;
        tamanho = capacidade = 0;
    }

    void _crescer() {
        int novaCap = capacidade * 2;
        EntradaMapaRef* novos = (EntradaMapaRef*)calloc(novaCap, sizeof(EntradaMapaRef));
        for(int i = 0; i < capacidade; i++) {
            if(!slots[i].chave) continue;
            uint32_t h = _hash(slots[i].chave, slots[i].tam) % (uint32_t)novaCap;
            while(novos[h].chave) h = (h + 1) % novaCap;
            novos[h] = slots[i];
        }
        free(slots);
        slots = novos;
        capacidade = novaCap;
    }

    // retorna ponteiro pro valor; insere com valor 0 se não existe
    int* buscarOuInserir(const char* chave, int tam) {
        if(tamanho * 2 >= capacidade) _crescer();
        uint32_t h = _hash(chave, tam) % (uint32_t)capacidade;
        while(slots[h].chave) {
            if(slots[h].tam == tam && memcmp(slots[h].chave, chave, tam) == 0) {
                return &slots[h].valor;
            }
            h = (h + 1) % capacidade;
        }
        slots[h].chave = chave;
        slots[h].tam   = tam;
        slots[h].valor = 0;
        tamanho++;
        return &slots[h].valor;
    }

    int* buscar(const char* chave, int tam) const {
        uint32_t h = _hash(chave, tam) % (uint32_t)capacidade;
        while(slots[h].chave) {
            if(slots[h].tam == tam && memcmp(slots[h].chave, chave, tam) == 0)
                return &slots[h].valor;
            h = (h + 1) % capacidade;
        }
        return nullptr;
    }
};

// heap de maximo por frequência(economico)
struct NodoHeap {
    int freq;
    int idcPar;
};

struct HeapMax {
    NodoHeap* dados;
    int tam;
    int cap;

    void iniciar(int c = 1024) {
        cap = c;
        tam = 0;
        dados = (NodoHeap*)malloc(cap * sizeof(NodoHeap));
    }

    void liberar() {
        free(dados);
        dados = nullptr;
        tam = cap = 0;
    }

    void _subir(int i) {
        while(i > 0) {
            int pai = (i - 1) / 2;
            if(dados[pai].freq >= dados[i].freq) break;
            NodoHeap tmp = dados[pai]; dados[pai] = dados[i]; dados[i] = tmp;
            i = pai;
        }
    }

    void _descer(int i) {
        while(true) {
            int maior = i;
            int e = 2*i+1, d = 2*i+2;
            if(e < tam && dados[e].freq > dados[maior].freq) maior = e;
            if(d < tam && dados[d].freq > dados[maior].freq) maior = d;
            if(maior == i) break;
            NodoHeap tmp = dados[maior]; dados[maior] = dados[i]; dados[i] = tmp;
            i = maior;
        }
    }

    void empurrar(NodoHeap n) {
        if(tam == cap) {
            cap *= 2;
            dados = (NodoHeap*)realloc(dados, cap * sizeof(NodoHeap));
        }
        dados[tam++] = n;
        _subir(tam - 1);
    }

    NodoHeap topo() const { return dados[0]; }

    void pop() {
        dados[0] = dados[--tam];
        if(tam > 0) _descer(0);
    }

    bool vazio() const { return tam == 0; }
};

// TokenizadorBPE
#define ID_ALMO 0
#define ID_DES 1
#define ID_FIM 2
#define PREFIXO_ESPACO "\xC4\xA0"
#define TAM_PREFIXO 2

class TokenizadorBPE {
public:
    MapaStrInt tokenPraId;
    char** idPraToken;
    int* idPraTam;
    int capacidadeIds;
    int proximoId;

    MapaStrInt bpeRanks;

    struct EntradaCache {
        uint32_t hash;
        char* chave;
        int chaveTam;
        VetorStr tokens;
        bool ocupado;
    };
    EntradaCache* cache;
    int cacheCap;
    int cacheTam;

    TokenizadorBPE() {
        tokenPraId.iniciar(2048);
        bpeRanks.iniciar(65536);

        capacidadeIds = 4096;
        idPraToken = (char**)calloc(capacidadeIds, sizeof(char*));
        idPraTam = (int*)calloc(capacidadeIds, sizeof(int));
        proximoId = 0;

        cacheCap = 4096;
        cacheTam = 0;
        cache = (EntradaCache*)calloc(cacheCap, sizeof(EntradaCache));

        _addToken("<ALMO>", 6);
        _addToken("<DES>",  5);
        _addToken("<FIM>",  5);
    }

    ~TokenizadorBPE() {
        tokenPraId.liberar();
        bpeRanks.liberar();
        for(int i = 0; i < proximoId; i++) free(idPraToken[i]);
        free(idPraToken);
        free(idPraTam);
        for(int i = 0; i < cacheCap; i++) {
            if(cache[i].ocupado) {
                free(cache[i].chave);
                cache[i].tokens.liberar();
            }
        }
        free(cache);
    }

    void addMerges(const char** pares, int numPares) {
        for(int i = 0; i < numPares; i++) {
            const char* a = pares[i*2];
            const char* b = pares[i*2+1];
            int tamA = (int)strlen(a);
            int tamB = (int)strlen(b);
            char chave[512];
            memcpy(chave, a, tamA);
            chave[tamA] = ' ';
            memcpy(chave + tamA + 1, b, tamB);
            chave[tamA + 1 + tamB] = '\0';
            bpeRanks.inserir(chave, tamA + 1 + tamB, i);
        }
    }

    void construirVocab(const char* texto, int tamTexto) {
        _limparCache();
        int i = 0;
        while(i < tamTexto) {
            unsigned char c = (unsigned char)texto[i];
            if(c == ' ' || c == '\t' || c == '\n' || c == '\r') { i++; continue; }
            int t = _tamUTF8(c);
            if(i + t > tamTexto) t = tamTexto - i;
            if(!tokenPraId.buscar(texto + i, t)) {
                _addToken(texto + i, t);
            }
            i += t;
        }
        VetorStr tokens; tokens.iniciar();
        VetorInt resultado; resultado.iniciar();
        _encode(texto, tamTexto, &tokens);
        for(int k = 0; k < tokens.tam; k++) {
            int tTam;
            const char* t = tokens.obter(k, &tTam);
            if(!tokenPraId.buscar(t, tTam)) {
                _addToken(t, tTam);
            }
        }
        tokens.liberar();
        resultado.liberar();
        printf("Vocabulário construído: %d tokens\n", proximoId);
    }

    void codificar(const char* texto, int tamTexto, VetorInt* saida) {
        saida->limpar();
        if(tamTexto == 0) return;
        VetorStr tokens; tokens.iniciar();
        _encode(texto, tamTexto, &tokens);
        for(int i = 0; i < tokens.tam; i++) {
            int tTam;
            const char* t = tokens.obter(i, &tTam);
            int* id = tokenPraId.buscar(t, tTam);
            if(id) {
                saida->empurrar(*id);
            } else {
                int j = 0;
                while(j < tTam) {
                    int sz = _tamUTF8((unsigned char)t[j]);
                    if(j + sz > tTam) sz = tTam - j;
                    int* cid = tokenPraId.buscar(t + j, sz);
                    saida->empurrar(cid ? *cid : ID_DES);
                    j += sz;
                }
            }
        }
        tokens.liberar();
    }

    char* decodificar(const int* ids, int numIds, int* tamSaida) {
        int capBuf = numIds * 4 + 4;
        char* buf = (char*)malloc(capBuf);
        int pos = 0;

        for(int i = 0; i < numIds; i++) {
            int id = ids[i];
            if(id == ID_ALMO || id == ID_DES || id == ID_FIM) continue;
            if(id < 0 || id >= proximoId) continue;

            const char* tok = idPraToken[id];
            int tTam = idPraTam[id];

            if(tTam >= TAM_PREFIXO &&
               (unsigned char)tok[0] == 0xC4 &&
               (unsigned char)tok[1] == 0xA0) {
                if(pos > 0) {
                    if(pos + 1 >= capBuf) { capBuf *= 2; buf = (char*)realloc(buf, capBuf); }
                    buf[pos++] = ' ';
                }
                tok += TAM_PREFIXO;
                tTam -= TAM_PREFIXO;
            }

            while(pos + tTam >= capBuf) {
                capBuf *= 2;
                buf = (char*)realloc(buf, capBuf);
            }
            memcpy(buf + pos, tok, tTam);
            pos += tTam;
        }
        buf[pos] = '\0';
        *tamSaida = pos;
        return buf;
    }

    int vocabTam() const { return proximoId; }

    void salvarVocab(const char* caminho) const {
        FILE* a = fopen(caminho, "wb");
        if(!a) {
            printf("Erro ao salvar vocab: %s\n", caminho);
            return;
        }
        fprintf(a, "%d\n", proximoId);
        for(int i = 0; i < proximoId; i++) {
            fprintf(a, "%d ", idPraTam[i]);
            fwrite(idPraToken[i], 1, idPraTam[i], a);
            fputc('\n', a);
        }
        fclose(a);
    }

    void carregarVocab(const char* caminho) {
        FILE* a = fopen(caminho, "rb");
        if(!a) {
            printf("Erro ao carregar vocab: %s\n", caminho);
            return;
        }
        int num;
        fscanf(a, "%d\n", &num);
        for(int i = 0; i < num; i++) {
            int tTam;
            fscanf(a, "%d ", &tTam);
            char* buf2 = (char*)malloc(tTam + 1);
            fread(buf2, 1, tTam, a);
            buf2[tTam] = '\0';
            int tmp = fgetc(a);
            if(i >= proximoId) {
                _garantirCapId(i + 1);
                idPraToken[i] = buf2;
                idPraTam[i] = tTam;
                tokenPraId.inserir(buf2, tTam, i);
                proximoId = i + 1;
            } else {
                free(buf2);
            }
        }
        fclose(a);
        printf("Vocab carregado: %d tokens\n", proximoId);
    }

    void salvarMerges(const char* caminho) const {
        FILE* a = fopen(caminho, "w");
        if(!a) {
            printf("Erro ao salvar merges: %s\n", caminho);
            return;
        }
        int n = bpeRanks.tamanho;
        struct Par {
            const char* chave;
            int rank;
        };
        Par* pars = (Par*)malloc(n * sizeof(Par));
        int k = 0;
        for(int i = 0; i < bpeRanks.capacidade; i++) {
            if(bpeRanks.slots[i].chave) {
                pars[k++] = {
                    bpeRanks.slots[i].chave,
                    bpeRanks.slots[i].valor
                };
            }
        }
        for(int a = 0; a < k-1; a++) {
            for(int b = a+1; b < k; b++) {
                if(pars[b].rank < pars[a].rank) {
                    Par tmp = pars[a];
                    pars[a] = pars[b];
                    pars[b] = tmp;
                }
            }
        }
        for(int i = 0; i < k; i++) fprintf(a, "%s\n", pars[i].chave);
        free(pars);
        fclose(a);
    }

    void carregarMerges(const char* caminho) {
        FILE* a = fopen(caminho, "r");
        if(!a) {
            printf("Erro ao carregar merges: %s\n", caminho);
            return;
        }
        char linha[1024];
        int rank = 0;
        while(fgets(linha, sizeof(linha), a)) {
            int tam = (int)strlen(linha);
            while(tam > 0 && (linha[tam-1] == '\n' || linha[tam-1] == '\r')) tam--;
            linha[tam] = '\0';
            if(tam == 0) continue;
            bpeRanks.inserir(linha, tam, rank++);
        }
        fclose(a);
        printf("Merges carregados: %d\n", rank);
    }

    void _addToken(const char* s, int tam) {
        _garantirCapId(proximoId + 1);
        char* copia = (char*)malloc(tam + 1);
        memcpy(copia, s, tam);
        copia[tam] = '\0';
        idPraToken[proximoId] = copia;
        idPraTam[proximoId] = tam;
        tokenPraId.inserir(s, tam, proximoId);
        proximoId++;
    }

    void _garantirCapId(int necessario) {
        if(necessario <= capacidadeIds) return;
        while(capacidadeIds < necessario) capacidadeIds *= 2;
        idPraToken = (char**)realloc(idPraToken, capacidadeIds * sizeof(char*));
        idPraTam = (int*)realloc(idPraTam,   capacidadeIds * sizeof(int));
    }

    void _limparCache() {
        for(int i = 0; i < cacheCap; i++) {
            if(cache[i].ocupado) {
                free(cache[i].chave);
                cache[i].tokens.liberar();
                cache[i].ocupado = false;
            }
        }
        cacheTam = 0;
    }

    VetorStr* _buscarCache(const char* palavra, int tam, uint32_t h) {
        uint32_t slot = h % (uint32_t)cacheCap;
        while(cache[slot].ocupado) {
            if(cache[slot].hash == h &&
               cache[slot].chaveTam == tam &&
               memcmp(cache[slot].chave, palavra, tam) == 0)
                return &cache[slot].tokens;
            slot = (slot + 1) % cacheCap;
        }
        return nullptr;
    }

    VetorStr* _inserirCache(const char* palavra, int tam, uint32_t h) {
        if(cacheTam * 2 >= cacheCap) {
            int novaCap = cacheCap * 2;
            EntradaCache* novo = (EntradaCache*)calloc(novaCap, sizeof(EntradaCache));
            for(int i = 0; i < cacheCap; i++) {
                if(!cache[i].ocupado) continue;
                uint32_t ns = cache[i].hash % (uint32_t)novaCap;
                while(novo[ns].ocupado) ns = (ns + 1) % novaCap;
                novo[ns] = cache[i];
            }
            free(cache);
            cache = novo;
            cacheCap = novaCap;
        }
        uint32_t slot = h % (uint32_t)cacheCap;
        while(cache[slot].ocupado) slot = (slot + 1) % cacheCap;
        cache[slot].ocupado = true;
        cache[slot].hash = h;
        cache[slot].chaveTam = tam;
        cache[slot].chave = (char*)malloc(tam + 1);
        memcpy(cache[slot].chave, palavra, tam);
        cache[slot].chave[tam] = '\0';
        cache[slot].tokens.iniciar();
        cacheTam++;
        return &cache[slot].tokens;
    }

    void _bpe(const char* palavra, int tamPalavra, VetorStr* saida) {
        uint32_t h = _hash(palavra, tamPalavra);
        VetorStr* cached = _buscarCache(palavra, tamPalavra, h);
        if(cached) {
            for(int i = 0; i < cached->tam; i++) {
                int tTam; const char* t = cached->obter(i, &tTam);
                saida->empurrar(t, tTam);
            }
            return;
        }
        int posicoes[512];
        int tamanhos[512];
        int nChars = 0;
        {
            int i = 0;
            while(i < tamPalavra && nChars < 512) {
                int t = _tamUTF8((unsigned char)palavra[i]);
                if(i + t > tamPalavra) t = tamPalavra - i;
                posicoes[nChars]  = i;
                tamanhos[nChars] = t;
                nChars++;
                i += t;
            }
        }
        if(nChars == 1) {
            VetorStr* entrada = _inserirCache(palavra, tamPalavra, h);
            entrada->empurrar(palavra, tamPalavra);
            saida->empurrar(palavra, tamPalavra);
            return;
        }
        int* wPos = (int*)malloc(nChars * sizeof(int));
        int* wTam = (int*)malloc(nChars * sizeof(int));
        memcpy(wPos, posicoes, nChars * sizeof(int));
        memcpy(wTam, tamanhos, nChars * sizeof(int));
        int wN = nChars;

        char chavePar[1024];

        while(wN > 1) {
            int melhorRank = -1;
            int melhorIdc  = -1;

            for(int i = 0; i + 1 < wN; i++) {
                int tamA = wTam[i];
                int tamB = wTam[i+1];
                int tamChave = tamA + 1 + tamB;
                if(tamChave >= (int)sizeof(chavePar)) continue;
                memcpy(chavePar, palavra + wPos[i], tamA);
                chavePar[tamA] = ' ';
                memcpy(chavePar + tamA + 1, palavra + wPos[i+1], tamB);
                chavePar[tamChave] = '\0';

                int* rank = bpeRanks.buscar(chavePar, tamChave);
                if(rank && (melhorRank < 0 || *rank < melhorRank)) {
                    melhorRank = *rank;
                    melhorIdc  = i;
                }
            }
            if(melhorIdc < 0) break;

            wTam[melhorIdc] = wTam[melhorIdc] + wTam[melhorIdc + 1];
            for(int i = melhorIdc + 1; i + 1 < wN; i++) {
                wPos[i] = wPos[i+1];
                wTam[i] = wTam[i+1];
            }
            wN--;
        }
        VetorStr* entrada = _inserirCache(palavra, tamPalavra, h);
        for(int i = 0; i < wN; i++) {
            entrada->empurrar(palavra + wPos[i], wTam[i]);
            saida->empurrar(palavra + wPos[i], wTam[i]);
        }
        free(wPos);
        free(wTam);
    }

    // primeiraAvra controla se a proxima palavra não-espaço
    // recebe prefixo Ġ. Ela so é true no inicio absoluto do texto
    // separadores consecutivos NÃO reiniciam primeiraAvra, eles apenas
    // descartam a palavra vazia. Assim "  espaços   multiplos  " tokeniza
    // corretamente como se fosse "espaços multiplos": espaços viram prefixo Ġ
    void _encode(const char* texto, int tamTexto, VetorStr* saida) {
        char palavraBuf[4096];
        int palavraTam = 0;
        bool primeiraAvra = true;

        for(int i = 0; i <= tamTexto; i++) {
            unsigned char c = (i < tamTexto) ? (unsigned char)texto[i] : 0;
            bool separador = (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0);

            if(!separador) {
                if(palavraTam < (int)sizeof(palavraBuf) - 1) {
                    palavraBuf[palavraTam++] = (char)c;
                }
            } else {
                if(palavraTam == 0) continue;
                palavraBuf[palavraTam] = '\0';

                if(primeiraAvra) {
                    _bpe(palavraBuf, palavraTam, saida);
                    primeiraAvra = false;
                } else {
                    VetorStr subTokens; subTokens.iniciar();
                    _bpe(palavraBuf, palavraTam, &subTokens);

                    if(subTokens.tam > 0) {
                        int tTam; const char* t = subTokens.obter(0, &tTam);
                        char* comP = (char*)malloc(tTam + TAM_PREFIXO + 1);
                        comP[0] = (char)0xC4; comP[1] = (char)0xA0;
                        memcpy(comP + TAM_PREFIXO, t, tTam);
                        saida->empurrar(comP, tTam + TAM_PREFIXO);
                        free(comP);
                        for(int k = 1; k < subTokens.tam; k++) {
                            int sLen; const char* s = subTokens.obter(k, &sLen);
                            saida->empurrar(s, sLen);
                        }
                    }
                    subTokens.liberar();
                }
                palavraTam = 0;
            }
        }
    }
};

// TreinadorBPE

// algoritmo incremental com lista duplamente encadeada + indice invertido + heap economico

// complexidade por merge: O(ocorrencias_do_par) em vez de O(total_tokens)
// é O((V*L + total_ocorr) log P) onde P = numero de pares unicos

struct ParMerge {
    char a[256];
    char b[256];
};

// nó da lista duplamente encadeada de tokens de uma palavra
struct NoLista {
    int pos; // posição dentro do buffer da palavra (ponteiro externo)
    int tam; // comprimento em bytes do token atual
    int ant; // indice do nó anterior(-1 = cabeça)
    int prox; // indice do nó próximo(-1 = cauda)
    bool ativo; // false = removido por fusão
};

// reuso de nós para uma palavra
struct ListaTokens {
    NoLista* nos;
    int tam;
    int cap;
    int cabeca;

    void iniciar(int c = 8) {
        cap = c;
        tam = 0;
        cabeca = -1;
        nos = (NoLista*)malloc(c * sizeof(NoLista));
    }

    void liberar() {
        free(nos);
        nos = nullptr;
        tam = cap = 0;
        cabeca = -1;
    }

    int alocarNo() {
        if(tam == cap) {
            cap *= 2;
            nos = (NoLista*)realloc(nos, cap * sizeof(NoLista));
        }
        nos[tam].ativo = true;
        nos[tam].ant   = -1;
        nos[tam].prox  = -1;
        return tam++;
    }
};

// ocorrencia de um par em uma palavra especifica
struct OcorrPar {
    int idcVocab;
    int idcNoEsq; // nó esquerdo(nó direito = nos[idcNoEsq].prox)
};

// lista dinamica de ocorrencias de um par
struct ListaOcorr {
    OcorrPar* dados;
    int tam;
    int cap;

    void iniciar() {
        dados = nullptr;
        tam = cap = 0;
    }
    void liberar() {
        free(dados);
        dados = nullptr;
        tam = cap = 0;
    }
    void limpar() { tam = 0; }

    void empurrar(int iv, int in) {
        if(tam == cap) {
            cap = cap ? cap * 2 : 8;
            dados = (OcorrPar*)realloc(dados, cap * sizeof(OcorrPar));
        }
        dados[tam++] = {iv, in};
    }
};

struct TreinadorBPE {
    ParMerge* merges;
    int numMerges;
    int capMerges;

    TreinadorBPE() : merges(nullptr), numMerges(0), capMerges(0) {}
    ~TreinadorBPE() { free(merges); }

    void treinar(const char* texto, int tamTexto, int maxMerges) {
        free(merges);
        merges = (ParMerge*)malloc(maxMerges * sizeof(ParMerge));
        numMerges = 0;
        capMerges = maxMerges;

        // 1. frequencia de palavras
        MapaStrInt freqPalavras; freqPalavras.iniciar(65536);
        {
            int i = 0;
            while(i < tamTexto) {
                while(i < tamTexto && _eSep(texto[i])) i++;
                int ini = i;
                while(i < tamTexto && !_eSep(texto[i])) i++;
                if(i == ini) continue;
                int* f = freqPalavras.buscar(texto + ini, i - ini);
                if(f) (*f)++;
                else freqPalavras.inserir(texto + ini, i - ini, 1);
            }
        }
        int nPalavras = freqPalavras.tamanho;

        // 2. vocab com lista encadeada por palavra
        struct EntradaVocab {
            const char* palavra;
            int tamPalavra;
            int freq;
            ListaTokens lista;
        };

        EntradaVocab* vocab = (EntradaVocab*)malloc(nPalavras * sizeof(EntradaVocab));
        int vIdc = 0;

        for(int i = 0; i < freqPalavras.capacidade; i++) {
            if(!freqPalavras.slots[i].chave) continue;
            const char* pal = freqPalavras.slots[i].chave;
            int tamPal = (int)strlen(pal);
            int freq = freqPalavras.slots[i].valor;

            EntradaVocab& ev = vocab[vIdc++];
            ev.palavra = pal;
            ev.tamPalavra = tamPal;
            ev.freq = freq;

            // conta chars UTF-8
            int nC = 0;
            for(int j = 0; j < tamPal; ) {
                int t = _tamUTF8((unsigned char)pal[j]);
                if(j + t > tamPal) t = tamPal - j;
                nC++; j += t;
            }
            ev.lista.iniciar(nC + 4);

            // monta lista encadeada
            int prev = -1;
            for(int j = 0; j < tamPal; ) {
                int t = _tamUTF8((unsigned char)pal[j]);
                if(j + t > tamPal) t = tamPal - j;
                int idc = ev.lista.alocarNo();
                ev.lista.nos[idc].pos  = j;
                ev.lista.nos[idc].tam  = t;
                ev.lista.nos[idc].ant  = prev;
                ev.lista.nos[idc].prox = -1;
                if(prev >= 0) ev.lista.nos[prev].prox = idc;
                else ev.lista.cabeca = idc;
                prev = idc;
                j += t;
            }
        }

        // 3. indice invertido de pares
        int capArrPares = 8192;
        ListaOcorr* arrPares = (ListaOcorr*)malloc(capArrPares * sizeof(ListaOcorr));
        int* freqPar = (int*)calloc(capArrPares, sizeof(int));
        int nArrPares = 0;

        // mapa chave(no bufPares) -> indice em arrPares
        MapaStrIntRef idcParMapa; idcParMapa.iniciar(131072);

        char chaveTmp[512];

        // retorna indice do par em arrPares, criando se necessario
        // ATENÇÃO: apos _crescer() do bufPares os ponteiros no mapa ficam
        // invalidos. Por isso usa posicoes em vez de ponteiros no mapa,
        // recriando os ponteiros ao final, mas é mais simples manter o
        // buffer sem reallocação usando tamanho fixo generoso.
        // solução: realocar bufPares sem mover posições ja inseridas no mapa.
        // como os slots do mapa guardam ponteiro direto pro buf, o realloc
        // invalidaria tudo. Resolvemos com realloc + fixup ou usando posicoes.
        // escolha pragmática: guardar posicao (int) em vez de ponteiro no mapa,
        // e recompor o ponteiro como bufPares + posicao na busca.
        // o MapaStrIntRef armazena const char*, não funciona direto.
        // mais simples: usar um MapaStrInt normal para este mapeamento,
        // com chave copiada. So que a chave ja esta, no bufPares, podemos usar
        // MapaStrInt com inserir sem copia extra
        // guardando apenas o posicao como int e buscando via strncmp igual.
        // --> usa MapaStrInt padrão mesmo; o custo de malloc por PAR UNICO
        // é aceitavel (há poucos pares unicos comparado ao total de tokens)

        // abandona MapaStrIntRef aqui; usa MapaStrInt normal para idcParMapa
        idcParMapa.liberar();
        MapaStrInt idcPar; idcPar.iniciar(131072);

        auto _obterOuCriarPar = [&](const char* a, int tA, const char* b, int tB) -> int {
            int tamChave = tA + 1 + tB;
            memcpy(chaveTmp, a, tA);
            chaveTmp[tA] = ' ';
            memcpy(chaveTmp + tA + 1, b, tB);
            chaveTmp[tamChave] = '\0';

            int* pidc = idcPar.buscar(chaveTmp, tamChave);
            if(pidc) return *pidc;

            // novo par
            if(nArrPares == capArrPares) {
                capArrPares *= 2;
                arrPares = (ListaOcorr*)realloc(arrPares, capArrPares * sizeof(ListaOcorr));
                freqPar = (int*)realloc(freqPar, capArrPares * sizeof(int));
                freqPar[nArrPares] = 0;
            }
            int idc = nArrPares++;
            arrPares[idc].iniciar();
            freqPar[idc] = 0;
            int* p = idcPar.inserir(chaveTmp, tamChave, idc);
            (void)p;
            return idc;
        };

        // preenche indice invertido inicial
        for(int v = 0; v < nPalavras; v++) {
            EntradaVocab& ev = vocab[v];
            int n = ev.lista.cabeca;
            while(n >= 0 && ev.lista.nos[n].prox >= 0) {
                int prox = ev.lista.nos[n].prox;
                const char* a  = ev.palavra + ev.lista.nos[n].pos;
                int tA = ev.lista.nos[n].tam;
                const char* b  = ev.palavra + ev.lista.nos[prox].pos;
                int tB = ev.lista.nos[prox].tam;
                int ip = _obterOuCriarPar(a, tA, b, tB);
                arrPares[ip].empurrar(v, n);
                freqPar[ip] += ev.freq;
                n = prox;
            }
        }
        // 4. heap de maximo economico
        HeapMax heap; heap.iniciar(nArrPares + 16);
        for(int i = 0; i < nArrPares; i++) {
            if(freqPar[i] > 1) {
                heap.empurrar({freqPar[i], i});
            }
        }

        // 5. loop principal de merges
        for(int iter = 0; iter < maxMerges; iter++) {
            // descarta nós economicos desatualizados
            while(!heap.vazio() && heap.topo().freq != freqPar[heap.topo().idcPar]) {
                heap.pop();
            }
            if(heap.vazio() || heap.topo().freq <= 1) {
                printf("Parou no merge %d: freq <= 1\n", iter);
                break;
            }
            NodoHeap melhor = heap.topo(); heap.pop();
            int idcMelhor = melhor.idcPar;
            int freqMelhor = melhor.freq;

            // recupera chave "a b" do MapaStrInt (busca linear — ocorre maxMerges vezes)
            const char* chaveM    = nullptr;
            int tamChaveM = 0;
            for(int i = 0; i < idcPar.capacidade; i++) {
                if(!idcPar.slots[i].chave) continue;
                if(idcPar.slots[i].valor == idcMelhor) {
                    chaveM = idcPar.slots[i].chave;
                    tamChaveM = (int)strlen(chaveM);
                    break;
                }
            }
            const char* esp  = (const char*)memchr(chaveM, ' ', tamChaveM);
            int tamA = (int)(esp - chaveM);
            int tamB = tamChaveM - tamA - 1;
            const char* partA = chaveM;
            const char* partB = esp + 1;

            memcpy(merges[numMerges].a, partA, tamA); merges[numMerges].a[tamA] = '\0';
            memcpy(merges[numMerges].b, partB, tamB); merges[numMerges].b[tamB] = '\0';

            if(iter < 10 || iter % 100 == 0) {
                printf("Merge %4d: '%s' + '%s' (freq=%d)\n", iter,
                merges[numMerges].a, merges[numMerges].b, freqMelhor);
            }
            numMerges++;

            // aplica merge: atualiza apenas os vizinhos afetados
            ListaOcorr& ocorrs = arrPares[idcMelhor];

            for(int oi = 0; oi < ocorrs.tam; oi++) {
                int v = ocorrs.dados[oi].idcVocab;
                int nEsq = ocorrs.dados[oi].idcNoEsq;

                EntradaVocab& ev = vocab[v];
                NoLista* nos = ev.lista.nos;

                if(!nos[nEsq].ativo) continue;
                int nDir = nos[nEsq].prox;
                if(nDir < 0 || !nos[nDir].ativo) continue;

                // verifica que o par ainda é valido(pode ter sido sobrescrito)
                if(nos[nEsq].tam != tamA ||
                memcmp(ev.palavra + nos[nEsq].pos, partA, tamA) != 0) continue;
                if(nos[nDir].tam  != tamB ||
                memcmp(ev.palavra + nos[nDir].pos, partB, tamB)  != 0) continue;

                int nAnt = nos[nEsq].ant;
                int nProx = nos[nDir].prox;

                // remove par(ant, esq) do indice
                if(nAnt >= 0) {
                    int ip = _obterOuCriarPar(
                        ev.palavra + nos[nAnt].pos, nos[nAnt].tam,
                        ev.palavra + nos[nEsq].pos, nos[nEsq].tam
                    );
                    freqPar[ip] -= ev.freq;
                    if(freqPar[ip] < 0) freqPar[ip] = 0;
                    // heap atualiza economicamente: não precisa fazer nada aqui
                }

                // remove par(dir, prox) do indice
                if(nProx >= 0) {
                    int ip = _obterOuCriarPar(
                        ev.palavra + nos[nDir].pos, nos[nDir].tam,
                        ev.palavra + nos[nProx].pos, nos[nProx].tam);
                    freqPar[ip] -= ev.freq;
                    if(freqPar[ip] < 0) freqPar[ip] = 0;
                }
                // funde nEsq + nDir
                nos[nEsq].tam  = tamA + tamB;
                nos[nEsq].prox = nProx;
                if(nProx >= 0) nos[nProx].ant = nEsq;
                nos[nDir].ativo = false;

                // adiciona par(ant, esq_fundido)
                if(nAnt >= 0) {
                    int ip = _obterOuCriarPar(
                        ev.palavra + nos[nAnt].pos, nos[nAnt].tam,
                        ev.palavra + nos[nEsq].pos, nos[nEsq].tam
                    );
                    arrPares[ip].empurrar(v, nAnt);
                    freqPar[ip] += ev.freq;
                    heap.empurrar({freqPar[ip], ip});
                }

                // adiciona par(esq_fundido, prox)
                if(nProx >= 0) {
                    int ip = _obterOuCriarPar(
                        ev.palavra + nos[nEsq].pos, nos[nEsq].tam,
                        ev.palavra + nos[nProx].pos, nos[nProx].tam
                    );
                    arrPares[ip].empurrar(v, nEsq);
                    freqPar[ip] += ev.freq;
                    heap.empurrar({freqPar[ip], ip});
                }
            }
            // esvazia ocorrencias do par aplicado(ja fundido em tudo)
            ocorrs.limpar();
            freqPar[idcMelhor] = 0;
        }
        // libera tudo
        for(int v = 0; v < nPalavras; v++) vocab[v].lista.liberar();
        free(vocab);
        for(int i = 0; i < nArrPares; i++) arrPares[i].liberar();
        free(arrPares);
        free(freqPar);
        idcPar.liberar();
        freqPalavras.liberar();
        heap.liberar();

        printf("Treinamento concluído: %d merges\n", numMerges);
    }

    void salvar(const char* caminho) const {
        FILE* a = fopen(caminho, "w");
        if(!a) {
            printf("Erro ao salvar merges\n");
            return;
        }
        for(int i = 0; i < numMerges; i++) {
            fprintf(a, "%s %s\n", merges[i].a, merges[i].b);
        }
        fclose(a);
        printf("Merges salvos: %s\n", caminho);
    }

    void carregar(const char* caminho) {
        numMerges = 0;
        FILE* a = fopen(caminho, "r");
        if(!a) {
            printf("Erro ao carregar merges\n");
            return;
        }
        char a2[256], b[256];
        while(fscanf(a, "%255s %255s", a2, b) == 2) {
            if(numMerges >= capMerges) {
                capMerges = capMerges ? capMerges * 2 : 1024;
                merges = (ParMerge*)realloc(merges, capMerges * sizeof(ParMerge));
            }
            strncpy(merges[numMerges].a, a2, 255);
            strncpy(merges[numMerges].b, b, 255);
            numMerges++;
        }
        fclose(a);
        printf("Merges carregados: %d\n", numMerges);
    }

    static inline bool _eSep(char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    }
};