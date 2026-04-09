class Perceptron {
    constructor() {
        this.pesos = [Math.random()*2-1, Math.random()*2-1];
        this.bias = Math.random()*2-1;
    }
    
    ativacao(x) {
        return x > 0 ? 1 : 0;
    }
    
    prever(entrada) {
        let soma = this.bias;
        for(let i = 0; i < entrada.length; i++) {
            soma += this.pesos[i] * entrada[i];
        }
        return this.ativacao(soma);
    }
    
    treinar(entrada, alvo, taxa=0.01) {
        const saida = this.prever(entrada);
        let erro = alvo - saida;
        
        for(let i = 0; i < this.pesos.length; i++) {
            this.pesos[i] += entrada[i] * erro * taxa;
        }
        this.bias += erro * taxa;
    }
}

const p = new Perceptron();

const entradas = [
    [0, 1],
    [1, 0],
    [0, 0],
    [1, 1]
];

const alvos = [
    0,
    0,
    0,
    1
];

for(let epoca = 0; epoca < 100; epoca++) {
    for(let i = 0; i < entradas.length; i++) {
        p.treinar(entradas[i], alvos[i]);
    }
}

console.log(p.prever([0, 1]));
console.log(p.prever([1, 1]));