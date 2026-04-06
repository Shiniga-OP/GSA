function degrau(x){return x>=0?1:0;}
function sigmoid(x){return 1/(1+Math.exp(-x));}
function derivadaSigmoid(x){return x*(1-x);}
function hardSigmoid(x){return Math.max(0,Math.min(1,0.2*x+0.5));}
function derivadaHardSigmoid(x){return (x>-2.5&& x<2.5)?0.2:0;}
function tanh(x){return Math.tanh(x);}
function derivadaTanh(x){return 1-x*x;}
function ReLU(x){return Math.max(0,x);}
function derivadaReLU(x){return x>0?1:0;}
function leakyReLU(x){return x>0?x:0.01*x;}
function derivadaLeakyReLU(x){return x>0?1:0.01;}
function softsign(x){return x/(1+Math.abs(x));}
function derivadaSoftsign(x){const denom=1+Math.abs(x);return 1/(denom*denom);}
function softplus(x){return Math.log(1+Math.exp(x));}
function swish(x){return x*sigmoid(x);}
function derivadaSwish(x){const sigmoidX=sigmoid(x);return sigmoidX+x*sigmoidX*(1-sigmoidX);}
function hardSwish(x){return x*Math.max(0,Math.min(1,(x+3)/6));}
function derivadaHardSwish(x){return x<=-3?0:x>=3?1:(x+3)/6+x/6;}
function GELU(x){return 0.5*x*(1+tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x,3))));}
function ELU(x,alfa=1.0){return x>=0?x:alfa*(Math.exp(x)-1);}
function derivadaELU(x,alfa=1.0){return x>=0?1:ELU(x,alfa)+alfa;}
function SELU(x,alfa=1.67326,escala=1.0507){return escala*(x>=0?x:alfa*(Math.exp(x)-1));}
function derivadaSELU(x,alfa=1.67326,escala=1.0507){return escala*(x>=0?1:alfa*Math.exp(x));}
function SiLU(x){return x*sigmoid(x);}
function mish(x){return x*tanh(Math.log(1+Math.exp(x)));}
function derivadaMish(x){const omega=4*(x+1)+4*Math.exp(2*x)+Math.exp(3*x)+Math.exp(x)*(4*x+6);const delta=2*Math.exp(x)+Math.exp(2*x)+2;return Math.exp(x)*omega/(delta*delta);}
function bentIdentity(x){return(Math.sqrt(x*x+1)-1)/2+x;}
function derivadaBentIdentity(x){return x/(2*Math.sqrt(x*x+1))+1;}
function gaussian(x){return Math.exp(-x*x);}
function derivadaGaussian(x){return-2*x*Math.exp(-x*x);}
// funções de saída:
function softmax(arr, temperatura=1) {
  if(!Array.isArray(arr)) throw new Error("[softmax]: valor passado não é um array");
  if(!isFinite(temperatura) || temperatura <= 0) temperatura=1e-8;
  const max = Math.max(...arr.map((v, i)=>{
    if(!isFinite(v)) throw new Error(`[softmax] arr[${i}] é NaN`)
    return v;
  }));
  const exps = arr.map(v=>{
    const t = (v-max)/temperatura;
    if(!isFinite(t)) throw new Error("[softmax]: t é NaN");
    return Math.exp(t);
  });
  const soma = exps.reduce((a, b)=>a+b, 0) || 1e-8;
  return exps.map(e=>e/soma);
}
function derivadaSoftmax(arr, gradSaida) {
  if(!Array.isArray(arr)) throw new Error("a derivada de softmax só pode receber vetores. Não: ", arr);
  let soma = 0;
  for(let j=0; j<gradSaida.length; j++) {
    const termo = gradSaida[j]*arr[j];
    if(!isFinite(termo)) throw new Error(`NaN em derivadaSoftmax: grad[${j}]=${gradSaida[j]}, arr[${j}]=${arr[j]}`);
    soma += termo;
  }
  return arr.map((s, i) => {
    const v = s*(gradSaida[i]-soma);
    if(!isFinite(v)) throw new Error(`NaN na saída da derivadaSoftmax[${i}]: s=${s}, grad=${gradSaida[i]}, soma=${soma}`);
    return v;
  });
}
function softmaxLote(matriz,temperatura=1){return matriz.map(linha=>softmax(linha,temperatura));}
function argmax(v){return v.indexOf(Math.max(...v));}
function addRuido(v,intenso=0.01){return v.map(x=>x+(Math.random()*2-1)*intenso);}
// funções de erro:
function erroAbsolutoMedio(saida,esperado){return saida.reduce((s,x,i)=>s+Math.abs(x-esperado[i]),0)/saida.length;}
function erroQuadradoEsperado(saida,esperado){return saida.reduce((s,x,i)=>s+0.5*(x-esperado[i])**2,0);}
function derivadaErro(saida,esperado){return saida.map((x,i)=>x-esperado[i]);}
function entropiaCruzada(y,yChapeu){return-y.reduce((s,yi,i)=>s+yi*Math.log(yChapeu[i]+1e-12),0);}
function derivadaEntropiaCruzada(y, yChapeu) {
  if(!Array.isArray(y) || !Array.isArray(yChapeu) || y.length != yChapeu.length) throw new Error("[derivadaEntropiaCruzada]: vetor inválido");
  return yChapeu.map((yci, i) => {
    const yi = y[i];
    if(!isFinite(yi) || !isFinite(yci)) throw new Error(`[derivadaEntropiaCruzada]: valor inválido em i=${i}: y=${yi}, y^=${yci}`);
    return yci-yi;
  });
}
function huberPerda(saida, esperado, delta=1.0) {
  const erros = saida.map((x, i)=>{
    const diff = x-esperado[i];
    return Math.abs(diff) <= delta ? 0.5*diff*diff : delta*(Math.abs(diff)-0.5*delta);
  });
  return erros.reduce((s, x)=>s+x, 0)/saida.length;
}
function derivadaHuber(saida, esperado, delta=1.0) {
  return saida.map((x, i)=>{
    const diff = x-esperado[i];
    return Math.abs(diff) <= delta ? diff : delta*Math.sign(diff);
  });
}
function perdaTripleto(ancora, positiva, negativa, margem=1.0) {
  const distPos = ancora.reduce((s, a, i)=>s+(a-positiva[i])**2, 0);
  const distNeg = ancora.reduce((s, a, i)=>s+(a-negativa[i])**2, 0);
  return Math.max(0, distPos-distNeg+margem);
}
function contrastivaPerda(saida1, saida2, rotulo, margem=1.0) {
  const distancia = saida1.reduce((s, x, i)=>s+(x-saida2[i])**2, 0);
  return rotulo==1 ? distancia : Math.max(0, margem-Math.sqrt(distancia));
}
// funções de regularização:
function regularL1(pesos,lambda) {return pesos.map(linha=>linha.map(p=>lambda*Math.sign(p)));}
function regularL2(pesos,lambda){return pesos.map(linha=>linha.map(p=>lambda*p));}
function dropout(tensor, taxa) {
  if(Array.isArray(tensor)) return tensor.map(sub=>dropout(sub, taxa));
  else return Math.random()<taxa?0:tensor/(1-taxa);
}
function clipGrad(grad,maxVal=1.0){return Math.min(Math.max(grad,-maxVal),maxVal);}
function normEntrada(vetor) {
  const max = Math.max(...vetor);
  const min = Math.min(...vetor);
  const amplitude = max-min || 1e-8;
  return vetor.map(x=>(x-min)/amplitude);
}
function normZPonto(v) {
  const media = v.reduce((a,b)=>a+b, 0)/v.length;
  const variancia=v.reduce((a,b)=>a+(b-media)**2,0)/v.length;
  const desvio = Math.sqrt(variancia+1e-8);
  return v.map(x=>(x-media)/desvio);
}
function acuracia(saida, esperado) {
  const corretos=saida.reduce((s,x,i)=>s+(argmax(x)===argmax(esperado[i])?1:0),0);
  return corretos/saida.length;
}
function precisao(confusao) {
  const tp = confusao[0][0], fp = confusao[0].slice(1).reduce((a,b)=>a+b, 0);
  return tp/(tp+fp+1e-8);
}
function recall(confusao) {
  const tp = confusao[0][0], fn = confusao.slice(1).reduce((s, l)=>s+l[0], 0);
  return tp/(tp+fn+1e-8);
}
function f1Score(confusao) {
  const p = precisao(confusao), r = recall(confusao);
  return 2*(p*r)/(p+r+1e-8);
}
function mse(saida,esperado){return saida.reduce((s,x,i)=>s+(x-esperado[i])**2,0)/saida.length;}
function klDivergencia(p,q){return p.reduce(((s,pi,i)=>s+(pi*Math.log((pi+1e-12)/(q[i]+1e-12)))),0);}
function rocAuc(pontos, rotulos) {
  const pares = pontos.map((s, i)=>[s, rotulos[i]]).sort((a,b)=>b[0]-a[0]);
  let auc = 0, fp = 0, tp = 0, fpPrev = 0, tpPrev = 0;
  pares.forEach(([s, r])=>{
    if(r===1) tp++;
    else fp++;
    auc += (fp-fpPrev)*(tp+tpPrev)/2;
    fpPrev = fp;
    tpPrev = tp;
  });
  return auc/(tp*fp);
}
// funções de pesos:
function iniPesosXavier(l, cols) {
  if(l <= 0 || cols <= 0) console.error("[iniPesosXavier]: valor negativo passado como parâmetro");
  const limite = Math.sqrt(6/(l+cols));
  return Array(l).fill().map(()=>Array(cols).fill().map(()=>(Math.random()*2-1)*limite));
}
function iniPesosHe(l, cols) {
  if(l <= 0 || cols <= 0) console.error("[iniPesosHe]: valor negativo passado como parâmetro");
  const limite = Math.sqrt(2/l);
  return Array(l).fill().map(()=>Array(cols).fill().map(()=>(Math.random()*2-1)*limite));
}
function iniPesosUniforme(l, cols) {
  const limiteInferior = -Math.sqrt(3/(l+cols));
  const limiteSuperior = Math.sqrt(3/(l+cols));
  return Array(l).fill().map(()=>Array(cols).fill().map(()=>(Math.random()*(limiteSuperior-limiteInferior)+limiteInferior)));
}
// atualização de pesos
function attPesos(pesos,grad,taxa,lambda=1e-3){return pesos.map((l,i)=>l.map((p,j)=>p-taxa*grad[i][j]-lambda*p));}
function attPesosMomentum(pesos,grad,taxa,momento,velocidade){return pesos.map((l,i)=>l.map((p,j)=>{velocidade[i][j]=momento*velocidade[i][j]+grad[i][j];return p-taxa*velocidade[i][j];}));}
function attPesosAdam(pesos, grad, m, v, taxa, beta1=0.9, beta2=0.999, eps=1e-8, iteracao, lambda=0) {
  const fator1 = 1-Math.pow(beta1, iteracao);
  const fator2 = 1-Math.pow(beta2, iteracao);
  return pesos.map((l, i)=>l.map((p, j)=>{
    const g = grad[i][j]+lambda*pesos[i][j];
    m[i][j] = beta1*m[i][j]+(1-beta1)*g;
    v[i][j] = beta2*v[i][j]+(1-beta2)*g*g;
    const mChapeu = m[i][j]/fator1;
    const vChapeu = v[i][j]/fator2;
    return p-taxa*mChapeu/(Math.sqrt(vChapeu)+eps);}));
}
function attPesosAdam1D(p, grad, m, v, taxa, beta1, beta2, eps, t) {
  for(let i=0; i<p.length; i++) {
    m[i] = beta1*m[i]+(1-beta1)*grad[i];
    v[i] = beta2*v[i]+(1-beta2)*grad[i]*grad[i];
    const mChapeu = m[i]/(1-Math.pow(beta1, t));
    const vChapeu= v[i]/(1-Math.pow(beta2, t));
    p[i] = p[i]-taxa*mChapeu/(Math.sqrt(vChapeu)+eps);
  }
  return p;
}
function attPesosRMSprop(pesos, grad, cache, taxa=0.001, decadencia=0.9, eps=1e-8) {
  cache = cache.map((l,i)=>l.map((val,j)=>decadencia*val+(1-decadencia)*grad[i][j]**2));
  return pesos.map((l,i)=>l.map((p,j)=>p-taxa*grad[i][j]/(Math.sqrt(cache[i][j])+eps)));
}
function attPesosAdagrad(pesos, grad, cache, taxa=0.01, eps=1e-8) {
  cache = cache.map((l,i)=>l.map((val,j)=>val+grad[i][j]**2));
  return pesos.map((l,i)=>l.map((p,j)=>p-taxa*grad[i][j]/(Math.sqrt(cache[i][j])+eps)));
}
// matrizes 3D:
function tensor3D(p,l,c,escala){return Array(p).fill().map(()=>matriz(l,c,escala));}
function zeros3D(p,l,c){return Array.from({length:p},()=>matrizZeros(l,c));}
function mapear3D(t,fn){return t.map(m=>m.map(l=>l.map(fn)));}
function somar3D(a,b){return a.map((m,i)=>somarMatriz(m,b[i]));}
function mult3DporEscalar(t,escalar){return t.map(m>multMatriz(m,escalar));}
// matrizes 2D:
function matriz(l,c,escala=0.1){return Array(l).fill().map(()=>vetor(c, escala));}
function exterior(a,b){return a.map(ai=>b.map(bj=>ai*bj));}
function clonarMatriz(m){return new Float32Array(m.buffer,m.byteOffset,m.length);}
function somarMatriz(a, b) {
  if(a.length != b.length) console.error("[somarMatrizes]: tamanho incompatível");
  return a.map((l, i)=>l.map((v, j)=>v+b[i][j]));
}
function subMatriz(a, b) {
  if(a.length != b.length) console.error("[subMatriz]: tamanho incompatível");
  return a.map((l, i)=>l.map((v, j)=>v-b[i][j]));
}
function multMatriz(m, s) {
  if(a.length != b.length) console.error("[multMatriz]: tamanho incompatível");
  return m.map(l=>l.map(v=>v*s));
}
function multMatrizes(a, b) {
  if(a.length != b.length) console.error("[multMatrizes]: tamanho incompatível");
  return a.map(l=>b[0].map((_,j)=>l.reduce((soma,valA,k)=>soma+valA*(b[k]?b[k][j]:0),0)));
}
function aplicarMatriz(m, v) {
  if(!Array.isArray(v)) console.error("[aplicarMatriz]: o segundo parâmetro não é um vetor");
  return m.map(l=>escalarDot(l, v));
}
function transpor(m){return[...m[0]].map((_,j)=>m.map(l=>l[j]));}
function matrizZeros(l,c){return Array.from({length:l}, ()=>zeros(c))}
function identidade(n) {return Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?1:0));}
function matrizVetor(m, v) {
  if(!Array.isArray(v))console.error("[matrizVetor]: o segundo parâmetro não é um vetor");
  return m.map(l=>{return l.reduce((soma,v,j)=>soma+v*v[j],0);});
}
function multVetorMatriz(v, c) {
  if(!Array.isArray(v))console.error("[multVetorMatriz]: o primeiro parâmetro não é um vetor");
  return v.map((x,i)=>x*c[i]);
}
function multMatrizVetor(m, v) {
  if(!Array.isArray(v)) console.error("[multVetorMatriz]: o primeiro parâmetro não é um vetor");
  return m.map(l=>l.reduce((soma, val, j)=>soma+val*v[j], 0));
}
function clipMatriz(m,limite){return m.map(l=>l.map(v=>Math.max(-limite,Math.min(v,limite))));}
// vetores
function vetor(n,escala=0.1){const v=new Float32Array(n);for(let i=0;i<n;i++){v[i]=(Math.random()*2-1)*escala;}return v;}
function somarVetores(a, b) {
  if(!a || !b || a.length !== b.length) throw new Error(`[somarVetores]: tamanho incompatível a=${a.length}, b=${b.length}`);
  return a.map((x, i)=>x+b[i]);
}
function subVetores(a, b) {
  if(a.length != a.length) console.error("[subVetores]: tamanho incompatível");
  return a.map((x, i)=>x-b[i]);
}
function multVetores(a, b) {
  if(a.length != a.length) console.error("[multVetores]: tamanho incompatível");
  return a.map((x, i)=>x*b[i]);
}
function escalarDot(v, w) {
  const eVet = Array.isArray(v) || v instanceof Float32Array;
  const eW = Array.isArray(w) || w instanceof Float32Array;
  if(!eVet || !eW) throw new Error("[escalarDot] Um dos vetores não é um array ou TypedArray");
  if(v.length != w.length) throw new Error(`[escalarDot]: tamanhos diferentes v=${v.length}, w=${w.length}`);
  let soma = 0;
  for(let i=0; i<v.length; i++) {
    const prod = v[i]*w[i];
    if(!isFinite(prod))throw new Error(`[escalarDot]: produto inválido em [${i}]:${v[i]}*${w[i]}=${prod}`);
    soma += prod;
  }
  return soma;
}
function multElementos(a, b) {
  if(a.length != b.length) console.error("[multElementos]: tamanho incompatível");
  return a.map((e, i)=>e*b[i]);
}
function normalizar(v) {
  if(!Array.isArray(v)) console.error("[normalizar]: o parâmetro não é um vetor");
  const mag = Math.sqrt(v.reduce((s, x)=>s+x*x, 0));
  return mag==0 ? v : v.map(x=>x/mag);
}
function zeros(n){return Array(n).fill(0);}
function uns(n){return Array(n).fill(1);}
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
function clipVetor(v, limite) {
  if(!Array.isArray(v)) console.error("[clipVetor]: o parâmetro não é um vetor");
  return v.map(x=>Math.max(-limite, Math.min(x, limite)));
}
// debug:
function heapUsado(){console.log(`Heap: ${process.memoryUsage().heapUsed>>20}MB`);}
function arraysIguais(a,b){return JSON.stringify(a)===JSON.stringify(b);}
function limparGrad(grad){return grad.map(l=>l.map(v=>isFinite(v)?v:0));}
function eNaN(tensor, nome, classe, parar=true) {
  for(let i=0; i<tensor.length; i++) {
    if(parar && tensor[0][0]==undefined) {
        if(isNaN(tensor[i])) throw new Error(`[${classe}] ${nome} contém NaN em [${i}]`);
        else if(!isFinite(tensor[i])) throw new Error(`[${classe}] ${nome} contém Infinity em [${i}][${j}]`);
        else if(tensor[i]==undefined) throw new Error(`[${classe}] ${nome} contém Undefined em [${i}]`);
      } else if(tensor[0][0]==undefined) {
        if(isNaN(tensor[i])) console.error(`[${classe}] ${nome} contém NaN em [${i}]`);
        else if(!isFinite(tensor[i])) console.error(`[${classe}] ${nome} contém Infinity em [${i}]`);
        else if(tensor[i]==undefined) console.error(`[${classe}] ${nome} contém Undefined em [${i}]`);
      }
    for(let j=0; j<tensor[i].length; j++) {
      if(parar) {
        if(isNaN(tensor[i][j])) throw new Error(`[${classe}] ${nome} contém NaN em [${i}][${j}]`);
        else if(!isFinite(tensor[i][j])) throw new Error(`[${classe}] ${nome} contém Infinity em [${i}][${j}]`);
        else if(tensor[i][j]==undefined) throw new Error(`[${classe}] ${nome} contém Undefined em [${i}][${j}]`);
      } else {
        if(isNaN(tensor[i][j])) console.error(`[${classe}] ${nome} contém NaN em [${i}][${j}]`);
        else if(!isFinite(tensor[i][j])) console.error(`[${classe}] ${nome} contém Infinity em [${i}][${j}]`);
        else if(tensor[i][j]==undefined) console.error(`[${classe}] ${nome} contém Undefined em [${i}][${j}]`);
      }
    }
  }
}
// vetorização:
function oneHot(i, tam) {
  const v = Array(tam).fill(0);
  if(i<0 || i >= tam)throw new Error(`[oneHot]: índice fora do vocabulário: ${i}`);
  v[i] = 1;
  return v;
}
function oneHotLote(idcs,tam){return idcs.map(i=>oneHot(i,tam));}
// probabilidades:
function amostrarTopP(probs, p=0.9) {
  const probsComIndice = probs.map((prob, indice)=>({indice, prob}));
  probsComIndice.sort((a, b)=>b.prob-a.prob);
  let somaCumulativa = 0;
  let i = 0;
  for(; i<probsComIndice.length; i++) {
    somaCumulativa += probsComIndice[i].prob;
    if(somaCumulativa >= p) break;
  }
  const nucleus = probsComIndice.slice(0, i+1);
  const somaNucleus = nucleus.reduce((s, item)=>s+item.prob, 0);
  const probsNormalizadas = nucleus.map(item=>item.prob/somaNucleus);
  const amostra = Math.random();
  let soma = 0;
  for(let j=0; j<nucleus.length; j++) {
    soma += probsNormalizadas[j];
    if(amostra <= soma) return nucleus[j].indice;
  }
  return nucleus[0].indice;
}