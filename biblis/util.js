// biblioteca de utilitários 100% português PT-br e JS
// ativações:
function degrau(x) {
  return x >= 0 ? 1 : 0;
}

function sigmoid(x) {
  return 1/(1+Math.exp(-x));
}
function derivadaSigmoid(x) {
  return x*(1-x);
}

function tanh(x) {
  return Math.tanh(x);
}
function derivadaTanh(x) {
  return 1-x*x;  
}

function ReLU(x) {
  return Math.max(0, x);
}
function derivadaReLU(x) {
  return x>0 ? 1 : 0;
}

function leakyReLU(x) {
  return x>0 ? x : 0.01*x;
}
function derivadaLeakyReLU(x) {
  return x>0 ? 1 : 0.01;
}

function softsign(x) {
  return x/(1+Math.abs(x));
}
function derivadaSoftsign(x) {
  let denom = 1+Math.abs(x);
  return 1/(denom*denom);
}

function softplus(x) {
  return Math.log(1+Math.exp(x));
}

function swish(x) {
  return x*sigmoid(x);
}
function derivadaSwish(x) {
  const sigmoidX = sigmoid(x);
  return sigmoidX+x*sigmoidX*(1-sigmoidX);
}

function GELU(x) {
  return 0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x, 3))));
}

function ELU(x, alfa=1.0) {
  return x >= 0 ? x : alfa*(Math.exp(x)-1);
}

function derivadaELU(x, alfa=1.0) {
  return x >= 0 ? 1 : ELU(x, alfa)+alfa;
}

function SELU(x, alfa=1.67326, escala=1.0507) {
  return escala*(x >= 0 ? x : alfa*(Math.exp(x)-1));
}

function derivadaSELU(x, alfa=1.67326, escala=1.0507) {
  return escala*(x >= 0 ? 1 : alfa*Math.exp(x));
}

function mish(x) {
  return x*Math.tanh(Math.log(1+Math.exp(x)));
}
function derivadaMish(x) {
  const omega = 4*(x+1)+4*Math.exp(2*x)+Math.exp(3*x)+Math.exp(x)*(4*x+6);
  const delta = 2*Math.exp(x)+Math.exp(2*x)+2;
  return Math.exp(x)*omega/(delta*delta);
}

// funções de saída:
function softmax(arr, temperatura=1) {
  if(!Array.isArray(arr)) console.error("[softmax]: valor passado não é um array");
  if(!isFinite(temperatura) || temperatura <= 0) temperatura=1e-8;
  const max = Math.max(...arr.map(v => isFinite(v) ? v : 0));
  const exps = arr.map(v => {
    const t = (v-max)/temperatura;
    return isFinite(t) ? Math.exp(t) : 0;
  });
  const soma = exps.reduce((a, b) => a+b, 0) || 1e-8;
  return exps.map(e => e/soma);
}

function derivadaSoftmax(arr, gradSaida) {
  if(!Array.isArray(arr)) console.error("a derivada de softmax só pode receber vetores. Não: ", arr);
  const gs = [];
  for(let i=0; i<arr.length; i++) {
    let soma = 0;
    for(let j=0; j<arr.length; j++) {
      const delta = (i==j ? 1 : 0);
      soma += gradSaida[j]*arr[i]*(delta-arr[j]);
    }
    gs[i] = soma;
  }
  return gs;
}

function softmaxLote(matriz, temperatura=1) {
  return matriz.map(linha => softmax(linha, temperatura));
}

function argmax(v) {
  return v.indexOf(Math.max(...v));
}

function addRuido(v, intensidade=0.01) {
  return v.map(x => x+(Math.random()*2-1)*intensidade);
}

// funções de erro:
function erroAbsolutoMedio(saida, esperado) {
  return saida.reduce((s, x, i) => s+Math.abs(x-esperado[i]), 0)/saida.length;
}
function erroQuadradoEsperado(saida, esperado) {
  return saida.reduce((s, x, i) => s+0.5*(x-esperado[i])**2, 0);
}
function derivadaErro(saida, esperado) {
  return saida.map((x, i) => x-esperado[i]);
}

function entropiaCruzada(y, yChapeu) {
  return -y.reduce((s, yi, i) => s+yi*Math.log(yChapeu[i]+1e-12), 0);
}
function derivadaEntropiaCruzada(y, yChapeu) {
  return yChapeu.map((yci, i) => yci-y[i]);
}

function huberPerda(saida, esperado, delta=1.0) {
  const erros = saida.map((x, i) => {
    const diff = x-esperado[i];
    return Math.abs(diff) <= delta ? 0.5*diff*diff : delta*(Math.abs(diff)-0.5*delta);
  });
  return erros.reduce((s, x) => s+x, 0)/saida.length;
}

function derivadaHuber(saida, esperado, delta=1.0) {
  return saida.map((x, i) => {
    const diff = x-esperado[i];
    return Math.abs(diff) <= delta ? diff : delta*Math.sign(diff);
  });
}

function perdaTripleto(ancora, positiva, negativa, margem=1.0) {
  const distPos = ancora.reduce((s, a, i) => s+(a-positiva[i])**2, 0);
  const distNeg = ancora.reduce((s, a, i) => s+(a-negativa[i])**2, 0);
  return Math.max(0, distPos-distNeg+margem);
}

function contrastivaPerda(saida1, saida2, rotulo, margem=1.0) {
  const distancia = saida1.reduce((s, x, i) => s+(x-saida2[i])**2, 0);
  return rotulo==1 ? distancia : Math.max(0, margem-Math.sqrt(distancia));
}

// funções de regularização:
function regularL1(pesos, lambda) {
  return pesos.map(linha => linha.map(p => lambda*Math.sign(p)));
}

function regularL2(pesos, lambda) {
  return pesos.map(linha => linha.map(p => lambda*p));
}

function dropout(vetor, taxa) {
  return vetor.map(val => Math.random()<taxa ? 0 : val/(1-taxa));
}

function clipGrad(grad, maxVal=1.0) {
  return Math.min(Math.max(grad, -maxVal), maxVal);
}

function normEntrada(vetor) {
  const max = Math.max(...vetor);
  const min = Math.min(...vetor);
  const amplitude = max-min || 1e-8;
  return vetor.map(x => (x-min)/amplitude);
}

function normZPonto(v) {
  const media = v.reduce((a,b) => a+b, 0)/v.length;
  const variancia = v.reduce((a,b) => a+(b-media)**2, 0)/v.length;
  const desvio = Math.sqrt(variancia);
  return v.map(x => (x-media)/(desvio+1e-8));
}

function acuracia(saida, esperado) {
  const corretos = saida.reduce((s, x, i) => s+(argmax(x)===argmax(esperado[i]) ? 1 : 0), 0);
  return corretos/saida.length;
}

function precisao(confusao) {
  const tp = confusao[0][0], fp = confusao[0].slice(1).reduce((a,b) => a+b, 0);
  return tp/(tp+fp+1e-8); // evita divisão por zero
}

function recall(confusao) {
  const tp = confusao[0][0], fn = confusao.slice(1).reduce((s, l) => s+l[0], 0);
  return tp/(tp+fn+1e-8);
}

function f1Score(confusao) {
  const p = precisao(confusao), r = recall(confusao);
  return 2*(p*r)/(p+r+1e-8);
}

function mse(saida, esperado) {
  return saida.reduce((s, x, i) => s+(x-esperado[i])**2, 0)/saida.length;
}

function klDivergencia(p, q) {
  return p.reduce(((s, pi, i) => s+(pi*Math.log((pi+1e-12)/(q[i]+1e-12)))), 0);
}

function rocAuc(pontos, rotulos) {
  const pares = pontos.map((s, i) => [s, rotulos[i]]).sort((a,b) => b[0]-a[0]);
  let auc = 0, fp = 0, tp = 0, fpPrev = 0, tpPrev = 0;
  pares.forEach(([s, r]) => {
    if(r===1) tp++; else fp++;
    auc += (fp-fpPrev)*(tp+tpPrev)/2;
    fpPrev = fp;
    tpPrev = tp;
  });
  return auc/(tp*fp);
}

// funções de pesos:
function iniPesosXavier(linhas, cols) {
  if(linhas <= 0 || cols <= 0) console.error("[iniPesosXavier]: valor negativo passado como parâmetro");
  let m = [];
  let limite = Math.sqrt(6/(linhas+cols));
  for(let i=0; i<linhas; i++) {
    m[i] = [];
    for(let j=0; j<cols; j++) {
      m[i][j] = (Math.random()*2-1)*limite;
    }
  }
  return m;
}

function iniPesosHe(linhas, cols) {
  if(linhas <= 0 || cols <= 0) console.error("[iniPesosHe]: valor negativo passado como parâmetro");
  let m = [];
  let limite = Math.sqrt(2 / linhas);
  for(let i=0; i<linhas; i++) {
    m[i] = [];
    for(let j=0; j<cols; j++) {
      m[i][j] = (Math.random()*2-1)*limite;
    }
  }
  return m;
}

function attPesos(pesos, gradientes, taxa) {
  return pesos.map((linha, i) =>
    linha.map((p, j) => p-taxa*gradientes[i][j])
  );
}

function attPesosMomentum(pesos, gradientes, taxa, momento, velocidade) {
  return pesos.map((linha, i) =>
    linha.map((p, j) => {
      velocidade[i][j] = momento*velocidade[i][j]+gradientes[i][j];
      return p-taxa*velocidade[i][j];
    })
  );
}

function attPesosAdam(pesos, gradientes, m, v, taxa, beta1=0.9, beta2=0.999, epsilon=1e-8, iteracao) {
  const mCorrigido = m.map((linha, i) => 
    linha.map((val, j) => beta1*val+(1-beta1)*gradientes[i][j])
  );
  const vCorrigido = v.map((linha, i) => 
    linha.map((val, j) => beta2*val+(1-beta2)*gradientes[i][j]**2)
  );
  const mHat = mCorrigido.map(linha => 
    linha.map(val => val/(1-Math.pow(beta1, iteracao)))
  );
  const vHat = vCorrigido.map(linha => 
    linha.map(val => val/(1-Math.pow(beta2, iteracao)))
  );
  return pesos.map((linha, i) =>
    linha.map((p, j) => p-taxa*mHat[i][j]/(Math.sqrt(vHat[i][j])+epsilon))
  );
}

function attPesosRMSprop(pesos, gradientes, cache, taxa=0.001, decadencia=0.9, epsilon=1e-8) {
  cache = cache.map((linha, i) => 
    linha.map((val, j) => decadencia*val+(1-decadencia)*gradientes[i][j]**2)
  );
  return pesos.map((linha, i) =>
    linha.map((p, j) => p-taxa*gradientes[i][j]/(Math.sqrt(cache[i][j])+epsilon))
  );
}

function attPesosAdagrad(pesos, gradientes, cache, taxa=0.01, epsilon=1e-8) {
  cache = cache.map((linha, i) => 
    linha.map((val, j) => val+gradientes[i][j]**2)
  );
  return pesos.map((linha, i) =>
    linha.map((p, j) => p-taxa*gradientes[i][j]/(Math.sqrt(cache[i][j])+epsilon))
  );
}

function iniPesosUniforme(linhas, cols, limiteInferior=-0.05, limiteSuperior=0.05) {
  let m = [];
  for(let i=0; i<linhas; i++) {
    m[i] = [];
    for(let j=0; j<cols; j++) {
      m[i][j] = Math.random()*(limiteSuperior-limiteInferior)+limiteInferior;
    }
  }
  return m;
}

// matrizes 3D:
function tensor3D(profundidade, linhas, colunas, escala=0.1) {
  let t = [];
  for(let d=0; d<profundidade; d++) {
    t[d] = matriz(linhas, colunas, escala);
  }
  return t;
}

function zeros3D(p, l, c) {
  return Array.from({length: p}, () => matrizZeros(l, c));
}

function mapear3D(tensor, fn) {
  return tensor.map(m => m.map(linha => linha.map(fn)));
}

function somar3D(a, b) {
  return a.map((mat, i) => somarMatriz(mat, b[i]));
}

function mult3DporEscalar(tensor, escalar) {
  return tensor.map(m => multMatriz(m, escalar));
}

// matrizes 2D:
function matriz(linhas, cols, escala=0.1) {
  let m = [];
  for(let i=0; i<linhas; i++) {
    m[i] = [];
    for(let j=0; j<cols; j++) {
      m[i][j] = (Math.random()*2-1)*escala;
    }
  }
  return m;
}

function matrizConfusao(saidas, esperados) {
  const classes = saidas[0].length;
  const matriz = Array.from({ length: classes }, ()=> Array(classes).fill(0));
  saidas.forEach((s, i) => {
    const pred = argmax(s);
    const real = argmax(esperados[i]);
    matriz[real][pred]++;
  });
  return matriz;
}

function exterior(a, b) {
  let res = [];
  for(let i=0; i<a.length; i++) {
    res[i] = [];
    for(let j=0; j<b.length; j++) {
      res[i][j] = a[i]*b[j];
    }
  }
  return res;
}

function clonarMatriz(m) {
  const copia = [];
  for(let i=0; i<m.length; i++) {
    copia[i] = m[i].slice();
  }
  return copia;
}

function somarMatriz(a, b) {
  if(a.length != a.length) console.error("[somarMatrizes]: tamanho incompatível");
  return a.map((linha, i) => linha.map((v, j) => v+b[i][j]));
}

function subMatriz(a, b) {
  if(a.length != a.length) console.error("[subMatriz]: tamanho incompatível");
  return a.map((linha, i) => linha.map((v, j) => v-b[i][j]));
}

function multMatriz(m, s) {
  if(a.length != a.length) console.error("[multMatriz]: tamanho incompatível");
  return m.map(linha => linha.map(v => v*s));
}

function multMatrizes(a, b) {
  if(a.length != a.length) console.error("[multMatrizes]: tamanho incompatível");
  return a.map(linhaA => b[0].map((_, j) => linhaA.reduce((soma, valA, k) => soma+valA*(b[k] ? b[k][j] : 0), 0)));
}

function multElementos(a, b) {
  if(a.length != a.length) console.error("[multElementos]: tamanho incompatível");
  return a.map((val, i) => val*b[i]);
}

function aplicarMatriz(m, v) {
  if(!Array.isArray(v)) console.error("[aplicarMatriz]: o segundo parâmetro não é um vetor");
  return m.map(linha => escalarDot(linha, v));
}

function transpor(m) {
  return [...m[0]].map((_, j) => m.map(linha => linha[j]));
}

function matrizZeros(linhas, cols) {
  return Array.from({length: linhas}, () => Array(cols).fill(0));
}

function identidade(n) {
  return Array.from({length: n}, (_, i) =>
    Array.from({length: n}, (_, j) => i===j ? 1 : 0)
  );
}

function matrizVetor(m, v) {
  if(!Array.isArray(v)) console.error("[matrizVetor]: o segundo parâmetro não é um vetor");
    return m.map(linha => {
        return linha.reduce((soma, val, j) => soma+val*v[j], 0);
    });
}

function multVetorMatriz(v, c) {
  if(!Array.isArray(v)) console.error("[multVetorMatriz]: o primeiro parâmetro não é um vetor");
  return v.map((x,i) => x*c[i]);
}

function multMatrizVetor(m, v) {
  if(!Array.isArray(v)) console.error("[multVetorMatriz]: o primeiro parâmetro não é um vetor");
  return m.map(linha =>
    linha.reduce((soma, val, j) => soma+val*v[j], 0)
  );
}

function clipMatriz(m, limite) {
  return m.map(linha => linha.map(v => Math.max(-limite, Math.min(v, limite))));
}

// vetores
function vetor(n, escala=0.1) {
  return Array(n).fill(0).map(() => (Math.random()*2-1)*escala);
}

function somarVetores(a, b) {
  if(!a || !b || a.length !== b.length)
    throw new Error(`[somarVetores]: tamanho incompatível a=${a.length}, b=${b.length}`);
  return a.map((x, i) => x+b[i]);
}

function subVetores(a, b) {
  if(a.length != a.length) console.error("[subVetores]: tamanho incompatível");
  return a.map((x, i) => x-b[i]);
}

function multVetores(a, b) {
  if(a.length != a.length) console.error("[multVetores]: tamanho incompatível");
  return a.map((x, i) => x*b[i]);
}

function escalarDot(v, w) {
  if(!Array.isArray(v) || !Array.isArray(w)) console.error("[escalarDot]: um dos parâmetros não é array");
  return v.reduce((s, x, i) => s+x*w[i], 0);
}

function normalizar(v) {
  if(!Array.isArray(v)) console.error("[normalizar]: o parâmetro não é um vetor");
  const mag = Math.sqrt(v.reduce((s, x) => s+x*x, 0));
  return mag==0 ? v : v.map(x => x/mag);
}

function zeros(n) {
  return Array(n).fill(0);
}

function uns(n) {
  return Array(n).fill(1);
}

function aplicarFuncaoVetor(v, fn) {
  if(!Array.isArray(v)) console.errror("[aplicarFuncaoVetor]: o primeiro parâmetro não é um vetor");
  return v.map(x => fn(x));
}

function embaralhar(pares) {
  for(let i=pares.length-1; i>0; i--) {
    const j = Math.floor(Math.random()*(i+1));
    [pares[i], pares[j]] = [pares[j], pares[i]];
  }
  return pares;
}

function dividirDados(pares, proporcao=0.8) {
  const nTreino = Math.floor(pares.length*proporcao);
  const treino = pares.slice(0, nTreino);
  const teste = pares.slice(nTreino);
  return [treino, teste];
}

function clip(grad, limite=1.0) {
  return grad.map(v => Math.max(-limite, Math.min(limite, v)));
}
function clipVetor(v, limite) {
  if(!Array.isArray(v)) console.error("[clipVetor]: o parâmetro não é um vetor");
  return v.map(x => Math.max(-limite, Math.min(x, limite)));
}

// debug:
function arraysIguais(a, b) {
  return JSON.stringify(a)===JSON.stringify(b);
}

// vetorização:
function oneHot(indice, tamanho) {
  const v = Array(tamanho).fill(0);
  v[indice] = 1;
  return v;
}

function oneHotLote(indices, tamanho) {
  return indices.map(i => oneHot(i, tamanho));
}

// texto:
function buscaFeixe(modelo, inicio, fim, maxComprimento, raioTam, temperatura=1.0, topoK=null) {
    let feixe = [{
        sequencia: [inicio],
        pontuacao: 0.0,
        finalizada: false
    }];
    const completas = [];
    for(let passo=0; passo<maxComprimento; passo++) {
        const candidatos = [];
        for(const hipotese of feixe) {
            if(hipotese.finalizada) {
                candidatos.push(hipotese);
                continue;
            }
            const seqAtual = hipotese.sequencia;
            
            const saida = modelo.propagar(seqAtual);
            const logits = saida[saida.length-1];
            
            const probs = softmax(logits, temperatura);
            
            let opcoes = probs
            .map((prob, token) => ({token, prob}))
            .filter(opcao => opcao.prob>0);
            
            if(topoK != null && topoK>0) {
                opcoes.sort((a, b) => b.prob-a.prob);
                opcoes = opcoes.slice(0, topoK);
            }
            for(const opcao of opcoes) {
                const novoToken = opcao.token;
                const novaSeq = [...seqAtual, novoToken];
                const novaPontuacao = hipotese.pontuacao+Math.log(opcao.prob+1e-10);
                
                const finalizada = (fim != undefined && novoToken==fim);
                
                candidatos.push({
                    sequencia: novaSeq,
                    pontuacao: novaPontuacao,
                    finalizada: finalizada
                });
            }
        }
        candidatos.sort((a, b) => b.pontuacao-a.pontuacao);
        
        feixe = [];
        for(const cand of candidatos) {
            if(feixe.length<raioTam) {
                feixe.push(cand);
            } else {
                if(cand.finalizada) {
                   completas.push(cand);
                }
            }
        }
        if(feixe.every(h => h.finalizada)) {
            completas.push(...feixe);
            break;
        }
    }
    completas.push(...feixe.filter(h => !h.finalizada));
    if(completas.length>0) {
        completas.sort((a, b) => b.pontuacao-a.pontuacao);
        return completas[0].sequencia;
    }
    feixe.sort((a, b) => b.sequencia.length-a.sequencia.length);
    return feixe[0].sequencia;
}

function minMax(m) {
  let min=Infinity, max=-Infinity;
  for(let i=0;i<m.length;i++) {
    for(let j=0;j<m[i].length;j++) {
      if(m[i][j]<min) min=m[i][j];
      if(m[i][j]>max) max=m[i][j];
      if(isNaN(m[i][j])) return "NaN detectado";
    }
  }
  return `min: ${min}, max: ${max}`;
}