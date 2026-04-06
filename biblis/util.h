// biblis/util.h
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static inline uint32_t _hash(const char* s, int tam);
static inline int _tamUTF8(unsigned char c);

// array dinamico simples
struct VetorInt {
    int* dados;
    int tam;
    int cap;

    void iniciar() {
        dados = nullptr;
        tam = cap = 0;
    }
    void liberar() {
        free(dados);
        iniciar();
    }
    void limpar() {
        tam = 0;
    }

    void empurrar(int v) {
        if(tam == cap) {
            cap = cap ? cap * 2 : 8;
            dados = (int*)realloc(dados, cap * sizeof(int));
        }
        dados[tam++] = v;
    }
    int operator[](int i) const {
        return dados[i];
    }
};

// buffer plano de strings
struct VetorStr {
    char* buf;
    int* pos;
    int* tams;
    int tam;
    int capE;
    int capB;

    void iniciar() {
        buf = nullptr;
        pos = nullptr;
        tams = nullptr;
        tam = 0;
        capE = 0;
        capB = 0;
    }

    void liberar() {
        free(buf);
        free(pos);
        free(tams);
        iniciar();
    }

    void limpar() { tam = 0; }

    const char* obter(int i, int* saidaTam) const {
        *saidaTam = tams[i];
        return buf + pos[i];
    }

    void empurrar(const char* s, int sLen) {
        if(tam == capE) {
            capE = capE ? capE * 2 : 8;
            pos = (int*)realloc(pos,  capE * sizeof(int));
            tams = (int*)realloc(tams, capE * sizeof(int));
        }
        int usadoBuf = tam > 0 ? pos[tam-1] + tams[tam-1] : 0;
        while(usadoBuf + sLen > capB) {
            capB = capB ? capB * 2 : 64;
            buf = (char*)realloc(buf, capB);
        }
        memcpy(buf + usadoBuf, s, sLen);
        pos[tam] = usadoBuf;
        tams[tam] = sLen;
        tam++;
    }

    void empurrar(const char* s) {
        empurrar(s, (int)strlen(s));
    }
};

// mapa hash string->int(endereçamento aberto, carga 0.5)
struct EntradaMapa {
    char* chave;
    int valor;
};

struct MapaStrInt {
    EntradaMapa* slots;
    int capacidade;
    int tamanho;

    void iniciar(int cap = 1024) {
        capacidade = cap;
        tamanho = 0;
        slots = (EntradaMapa*)calloc(cap, sizeof(EntradaMapa));
    }

    void liberar() {
        for(int i = 0; i < capacidade; i++) {
            if(slots[i].chave) free(slots[i].chave);
        }
        free(slots);
        slots = nullptr;
        tamanho = capacidade = 0;
    }

    void _crescer() {
        int novaCap = capacidade * 2;
        EntradaMapa* novos = (EntradaMapa*)calloc(novaCap, sizeof(EntradaMapa));
        for(int i = 0; i < capacidade; i++) {
            if(!slots[i].chave) continue;
            uint32_t h = _hash(slots[i].chave, (int)strlen(slots[i].chave)) % (uint32_t)novaCap;
            while(novos[h].chave) h = (h + 1) % novaCap;
            novos[h] = slots[i];
        }
        free(slots);
        slots = novos;
        capacidade = novaCap;
    }

    int* buscar(const char* chave, int tam = -1) const {
        if(tam < 0) tam = (int)strlen(chave);
        uint32_t h = _hash(chave, tam) % (uint32_t)capacidade;
        while(slots[h].chave) {
            if(strncmp(slots[h].chave, chave, tam) == 0 && slots[h].chave[tam] == '\0') {
                return &slots[h].valor;
            }
            h = (h + 1) % capacidade;
        }
        return nullptr;
    }

    int* inserir(const char* chave, int tam, int valor) {
        if(tamanho * 2 >= capacidade) _crescer();
        uint32_t h = _hash(chave, tam) % (uint32_t)capacidade;
        while(slots[h].chave) {
            if(strncmp(slots[h].chave, chave, tam) == 0 && slots[h].chave[tam] == '\0') {
                slots[h].valor = valor;
                return &slots[h].valor;
            }
            h = (h + 1) % capacidade;
        }
        slots[h].chave = (char*)malloc(tam + 1);
        memcpy(slots[h].chave, chave, tam);
        slots[h].chave[tam] = '\0';
        slots[h].valor = valor;
        tamanho++;
        return &slots[h].valor;
    }

    int* inserir(const char* chave, int valor) {
        return inserir(chave, (int)strlen(chave), valor);
    }
};

// hash de string simples(FNV-1a)
static inline uint32_t _hash(const char* s, int tam) {
    uint32_t h = 2166136261u;
    for(int i = 0; i < tam; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h;
}

// utilitarios UTF-8
static inline int _tamUTF8(unsigned char c) {
    if((c & 0x80) == 0) return 1;
    if((c & 0xE0) == 0xC0) return 2;
    if((c & 0xF0) == 0xE0) return 3;
    if((c & 0xF8) == 0xF0) return 4;
    return 1;
}