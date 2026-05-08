
// Construtor do modelo

function modelo = ModeloPropagacaoInformacao(beta, alpha , k )
    
    modelo = struct();
    
    modelo.beta = beta;
    modelo.alpha = alpha;
    modelo.k = k;
    
endfunction


function [novo_I, novo_S, novo_R] = passo(modelo,I,S,R)
    
    novo_I = I - (modelo.beta)*(modelo.k)*S*I;
    novo_S = S + (modelo.beta)*(modelo.k)*S*I - modelo.alpha*S*(S+R);
    novo_R = R + (modelo.alpha)*S*(S+R);
    
endfunction
