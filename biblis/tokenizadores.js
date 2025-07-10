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
