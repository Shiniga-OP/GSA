// teste_bpe.cpp
#include "biblis/tokes/bpe.h"

void testeTreinador() {
    printf("\n=== TESTE TREINADOR BPE 2 ===\n\n");

    std::vector<std::string> corpus = {
        "olá mundo",
        "olá brasil",
        "mundo novo",
        "novo mundo",
        "teste de tokenização",
        "tokenização de texto",
        "texto novo",
    };
    TreinadorBPE treinador;
    treinador.treinar(corpus, 50);

    const auto& merges = treinador.merges;
    printf("\nTotal de merges aprendidos: %d\n", (int)merges.size());

    // integração com TokenizadorBPE
    TokenizadorBPE tok(merges);
    tok.construirVocab(corpus);
    printf("Tamanho do vocab: %d\n", tok.vocabTam());

    std::string frase = "olá mundo novo";
    std::vector<int> cod = tok.codificar(frase);
    std::string dec = tok.decodificar(cod);
    printf("\nTexto: %s\n", frase.c_str());
    printf("Codificado: ");
    for(int id : cod) printf("%d ", id);
    printf("\nDecodificado: %s\n", dec.c_str());
    if(dec == frase) printf("Frase OK\n");
    else printf("FALHA: esperado '%s'\n", frase.c_str());
}

int main() {
    printf("\n=== TESTES TOKENIZADOR BPE ===\n\n");
    TokenizadorBPE t({});
    std::vector<std::string> textos = { "olá mundo", "teste de tokenização" };
    t.construirVocab(textos);

    std::string frase = "olá mundo";
    std::vector<int> cod = t.codificar(frase);
    std::string dec = t.decodificar(cod);
    printf("Texto: %s\n", frase.c_str());
    printf("Codificado: ");
    for(int id : cod) printf("%d ", id);
    printf("\nDecodificado: %s\n", dec.c_str());

    // verifica se tá correto
    if(dec == frase) printf("Frase certa OK\n");
    else printf("FALHA na frase: esperado '%s', obtido '%s'\n", frase.c_str(), dec.c_str());
    
    std::vector<std::string> corpus = {
        "o gato comeu o rato",
        "o rato fugiu do gato",
        "o gato meteu a bicuda no rato",
        "o rato saiu voando da bicuda do gato",
        "o gato gritou RONALDINHO SOCCER",
        "e o gato comeu o rato amassado :3",
        "fim da historia do gato, agora a historia do tijolinho",
        "o peninha perguntou pra mãe dele: mãe, por que meu nome é peninha?",
        "a mãe dele respondeu: é porque caiu uma peninha na sua cabeça",
        "ai o tijolinho perguntou, ababahabababan",
        "fim da historia do tijolinho",
        "agora, você sabia que três pratos de trigo não alimentam três tigris tristes?"
        "porque eles são carnívoros, e não comem trigo",
        "e se o rato roeou a roupa do rei de roma eu já não sei",
        "inclusive, você sabia que fazer merges manuais assim dá bastante trabalho?",
        "pois é, vou ter que apagar tudo isso pra colocar em um arquivo ou mais de textos gigantes mais tarde",
        "é complicado ser programador",
        "enfim, bom dia, vou tomar café da manhã",
        "pão francês com hamburger, mussarela, ovo, e um copo de café com leite",
        "bom pra reforçar, já que comi dois pães de queijo mais cedo",
        "até mais tarde"
    };
    TreinadorBPE treinador;
    treinador.treinar(corpus, 100);
    treinador.salvar("merges.txt");
    
    TokenizadorBPE tok(treinador.merges);
    tok.construirVocab(corpus);
    
    testeTreinador();
    
    return 0;
}