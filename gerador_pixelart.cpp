#include <iostream>
#include <vector>
#include <memory>
#include <ctime>
#include "biblis/camadas.h"

using namespace std;

int main() {
    srand(time(NULL));
    // gerador(10 -> 64 -> 64)
    auto gerador = make_unique<Modelo>("Gerador");
    gerador->add(make_unique<Densa>(10, 64, "relu", true, "G_Oculta1"));
    gerador->add(make_unique<Densa>(64, 64, "sigmoid", true, "G_Saida"));

    // discriminador
    auto discriminador = make_unique<Modelo>("Discriminador");
    discriminador->add(make_unique<Densa>(64, 32, "leakyrelu", true, "D_Oculta"));
    discriminador->add(make_unique<Densa>(32, 1, "sigmoid", true, "D_Saida"));

    // dados com cruz e quadrado
    vector<vector<float>> dadosReais;
    
    vector<float> cruz(64, 0.0f);
    for(int i=0; i<8; i++) {
        cruz[i*8+3]=1.0f;
        cruz[i*8+4]=1.0f;
        cruz[3*8+i]=1.0f;
        cruz[4*8+i]=1.0f;
    }
    
    vector<float> quadrado(64, 0.0f);
    for(int i=2; i<6; i++) {
        for(int j=2; j<6; j++) {
            quadrado[i*8+j]=1.0f;
        }
    }
    dadosReais.push_back(cruz);
    dadosReais.push_back(quadrado);

    // taxas de aprendizado assimetricas pra equilibrar o jogo
    float taxaG = 0.005f; 
    float taxaD = 0.001f; // discriminador aprende mais devagar
    int epocas = 15000;
    cout << "Modelo gerador criado com " << gerador->numParametros() << " parametros" << endl;
    cout << "Modelo discriminador criado com " << discriminador->numParametros() << " parametros" << endl;
    cout << "Treinando com equilíbrio de forças..." << endl;

    for(int e = 0; e <= epocas; ++e) {
        // === treino do discriminador ===
        vector<float>& real = dadosReais[rand() % dadosReais.size()];
        discriminador->treinar(real, {1.0f}, erroQuadradoEsperado, taxaD);
        
        vector<float> falsa = gerador->prop(vetor(10, 1.0f));
        discriminador->treinar(falsa, {0.0f}, erroQuadradoEsperado, taxaD);
        
        // === treino do gerador ===
        for(int i = 0; i < 2; i++) {
            vector<float> ruidoG = vetor(10, 1.0f);
            vector<float> gerada = gerador->prop(ruidoG);
            vector<float> dSaida = discriminador->prop(gerada);
            
            // o D tem que dizer que é 1.0
            vector<float> gradErro = { dSaida[0] - 1.0f };
            vector<float> gradParaG = discriminador->retroprop(gradErro);
            
            gerador->retroprop(gradParaG);
            gerador->att(taxaG);
            
            gerador->zerarGradientes();
            discriminador->zerarGradientes();
        }
        if(e % 2000 == 0) {
            vector<float> teste = gerador->prop(vetor(10, 1.0f));
            cout << "Epoca: " << e << "/" << epocas << endl;
            gravarImg(teste);
        }
    }
    cout << "\nResultado em varios estilos:" << endl;
    for(int i=0; i<4; i++) {
        vector<float> img = gerador->prop(vetor(10, 1.0f));
        gravarImg(img);
        criarBMP24(img, "imagem_" + to_string(i) + ".bmp");
    }
    return 0;
}