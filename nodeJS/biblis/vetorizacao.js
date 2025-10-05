class CodificadorPosicional {
  constructor(dimModelo, seqMaxima=5000) {
    this.dimModelo = dimModelo;
    this.seqMaxima = seqMaxima;
    this.codificacao = this.gerarCodificacao();
  }
  gerarCodificacao() {
    const codificacao = [];
    for(let pos=0; pos<this.seqMaxima; pos++) {
      const v = [];
      for(let i=0; i<this.dimModelo; i++) {
        const expoente = i/this.dimModelo;
        const valor = pos/Math.pow(10000, expoente);
        v.push(i%2==0 ? Math.sin(valor) : Math.cos(valor));
      }
      codificacao.push(v);
    }
    return codificacao;
  }
  aplicar(x) {
    return x.map((seq, pos)=>somarVetores(seq, this.codificacao[pos]));
  }
}
class TokenizadorBPE {
  constructor(merges=[]) {
    this.vocab = {};
    this.bpeRanks = {};
    this.cache = new Map();
    // inicializa os merges
    for(let i=0; i<merges.length; i++) {
      const chave = merges[i].join(' ');
      this.bpeRanks[chave] = i;
    }
    // tokens especiais
    this.tokenPraId = new Map([['<ALMO>', 0], ['<DES>', 1], ['<FIM>', 2]]);
    this.idPraToken = new Map([[0, '<ALMO>'], [1, '<DES>'], [2, '<FIM>']]);
    this.proximoId = 3;
  }
  construirVocabulario(textos) {
    const todosTokens = new Set();
    const todosCaracteres = new Set();
    // add tokens especiais
    todosTokens.add('<ALMO>');
    todosTokens.add('<DES>');
    todosTokens.add('<FIM>');
    // coleta todos os caracteres únicos primeiro
    for(const texto of textos)for(const carac of texto)if(carac.trim() !== '')todosCaracteres.add(carac);
    for(const carac of todosCaracteres)todosTokens.add(carac);
    // processa todos os textos pra tokens BPE
    for(const texto of textos) {
      const tokens = this.encode(texto);
      tokens.forEach(token=>todosTokens.add(token));
    }
    // mapeia tokens para IDs
    let id = 3;
    for(const token of todosTokens) {
      if(!this.tokenPraId.has(token)) {
        this.tokenPraId.set(token, id);
        this.idPraToken.set(id, token);
        id++;
      }
    }
    this.proximoId = id;
    console.log(`Vocabulário construído: ${this.proximoId} tokens`);
  }
  codificar(texto) {
    const tokensBPE = this.encode(texto);
    return tokensBPE.map(token=>{
      let id = this.tokenPraId.get(token);
      if(id !== undefined) return id;
      // se token não existe, tenta quebrar em caracteres individuais
      const caracs = [...token];
      if(caracs.length===1) {
        // é um caractere único, adiciona ao vocabulário
        id = this.proximoId++;
        this.tokenPraId.set(token, id);
        this.idPraToken.set(id, token);
        return id;
      }
      // token composto desconhecido, usa <DES>
      return 1;
    });
  }
  decodificar(ids) {
    const tokens = ids.map(id=>{
      if(id===2) return ''; // <FIM>
      const token = this.idPraToken.get(id);
      return token !== undefined ? token : '<DES>';
    }).filter(token => token !== ''); // remove tokens vazios
    return this.decode(tokens);
  }
  get vocabTam(){return this.proximoId;}
  obterPares(palavra) {
    const pares = new Set();
    for(let i=0; i<palavra.length-1; ++i) pares.add(palavra[i]+" "+palavra[i+1]);
    return pares;
  }
  bpe(token) {
    if(this.cache.has(token)) return this.cache.get(token);
    let palavra = token.split('');
    let pares = this.obterPares(palavra);
    if(!pares.size) return [token];
    while(true) {
      let minRank = Infinity;
      let melhorPar = null;
      for(const par of pares) {
        const rank = this.bpeRanks[par];
        if(rank !== undefined && rank<minRank) {
          minRank = rank;
          melhorPar = par;
        }
      }
      if(melhorPar===null) break;
      const [primeiro, segundo] = melhorPar.split(' ');
      const novaPalavra = [];
      let i = 0;
      while(i<palavra.length) {
        let j = palavra.indexOf(primeiro, i);
        if(j===-1) {
          novaPalavra.push(...palavra.slice(i));
          break;
        }
        novaPalavra.push(...palavra.slice(i, j));
        if(j<palavra.length-1 && palavra[j+1]===segundo) {
          novaPalavra.push(primeiro+segundo);
          i = j+2;
        } else {
          novaPalavra.push(palavra[j]);
          i = j+1;
        }
      }
      palavra = novaPalavra;
      pares = this.obterPares(palavra);
    }
    this.cache.set(token, palavra);
    return palavra;
  }
  encode(texto) {
    // divide em palavras mantendo espaços
    const palavras = texto.split(/(\s+)/);
    const tokens = [];
    for(const palavra of palavras) {
      if(palavra.trim()=='') {
        // é um espaço - tratar como token especial
        if(palavra==' ') tokens.push('Ġ');
        else tokens.push('Ġ');
      } else {
        // é uma palavra real - tentar BPE primeiro
        const bpeTokens = this.bpe(palavra);
        // se BPE falhou(retornou a palavra original)
        if(bpeTokens.length===1 && bpeTokens[0]===palavra&&!this.tokenPraId.has(palavra))for(const carac of palavra)tokens.push(carac);
        else tokens.push(...bpeTokens);
      }
    }
    return tokens;
  }
  decode(tokens) {
    let texto = '';
    for(const token of tokens) {
      if(token=='Ġ') texto += ' ';
      else if(token.startsWith('Ġ'))texto += ' '+token.slice(1);
      else texto += token;
    }
    return texto;
  }
}
class TreinadorBPE {
  constructor(numMerges=500, logIntervalo=0.1) {
    this.numMerges = numMerges;
    this.logIntervalo = logIntervalo;
    this.vocab = {};
    this.merges = [];
  }
  treinar(textos) {
    console.log("Iniciando treinamento BPE");
    console.log(`Textos: ${textos.length} | Merges: ${this.numMerges}`);
    this.construirVocab(textos);
    console.log(`Tokens iniciais: ${Object.keys(this.vocab).length}`);
    const pontosLog = [];
    for(let i=1; i <= 10; i++) pontosLog.push(Math.floor(this.numMerges*i*this.logIntervalo));
    for(let i=0; i<this.numMerges; i++) {
      if(pontosLog.includes(i)) console.log(`Progresso: ${Math.round((i/this.numMerges)*100)}% (${i}/${this.numMerges})`);
      const pares = this.contarPares();
      if(Object.keys(pares).length==0) {
        console.log("Sem pares disponíveis - parando antecipadamente");
        break;
      }
      const [melhorPar, freq] = Object.entries(pares).reduce((melhor, [par, f])=>f>melhor[1] ? [par, f] : melhor, ['', 0]);
      this.aplicarMerge(melhorPar);
      this.merges.push(melhorPar.split(' '));
    }
    console.log("Treinamento concluído");
    console.log(`Merges realizados: ${this.merges.length}`);
    return this.merges;
  }
  construirVocab(textos) {
    this.vocab = {};
    textos.forEach(txt => {
      const palavras = txt.split(/(\s+)/);
      palavras.forEach(palavra=>{
        if(palavra.trim()==='') {
          // é um espaço
          const tokens = ['Ġ', '</w>'];
          const chave = tokens.join(' ');
          this.vocab[chave] = (this.vocab[chave] || 0)+1;
        } else {
          // é uma palavra real
          const tokens = [...palavra, '</w>'];
          const chave = tokens.join(' ');
          this.vocab[chave] = (this.vocab[chave] || 0)+1;
        }
      });
    });
  }
  contarPares() {
    const pares = {};
    for(const [seq, freq] of Object.entries(this.vocab)) {
      const tokens = seq.split(' ');
      for(let i=0; i<tokens.length-1; i++) {
        const par = tokens[i]+' '+tokens[i+1];
        pares[par] = (pares[par] || 0)+freq;
      }
    }
    return pares;
  }
  aplicarMerge(par) {
    const [a, b] = par.split(' ');
    const novoToken = a+b;
    const regex = new RegExp(`${this.escapeRegex(a)} ${this.escapeRegex(b)}`, 'g');
    const novoVocab = {};
    for(const [seq, freq] of Object.entries(this.vocab)) {
      novoVocab[seq.replace(regex, novoToken)] = freq;
    }
    this.vocab = novoVocab;
  }
  escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
}
class TokenizadorSimples {
  constructor() {
    this.tokenPraId = new Map();
    this.idPraToken = new Map();
    this.proximoId = 0;
  }
  construir(texto) {
    const tokens = texto.toLowerCase().split(/\s+/);
    
    for(const token of tokens) {
      if(!this.tokenPraId.has(token)) {
        this.tokenPraId.set(token, this.proximoId);
        this.idPraToken.set(this.proximoId, token);
        this.proximoId++;
      }
    }
    this.tokenPraId.set('<ALMO>', this.proximoId);
    this.idPraToken.set(this.proximoId, '<ALMO>');
    this.proximoId++;
    
    this.tokenPraId.set('<DES>', this.proximoId);
    this.idPraToken.set(this.proximoId, '<DES>');
    this.proximoId++;
  }
  codificar(texto) {
    const tokens = texto.toLowerCase().split(/\s+/);
    return tokens.map(token => 
      this.tokenPraId.get(token) || this.tokenPraId.get('<DES>')
    );
  }
  decodificar(ids) {
    return ids.map(id => this.idPraToken.get(id) || '<DES>').join(' ');
  }
  get vocabTam() {
    return this.proximoId;
  }
}
