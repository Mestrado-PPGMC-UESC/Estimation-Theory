
function simulador = Simulador(modelo,numero_passos)
    
    simulador = struct();
    simulador.modelo = modelo;
    simulador.numero_passos = numero_passos;
    
endfunction


function [I,S,R] = executar(simulador,I0,S0,R0)
    
    I=zeros(1,simulador.numero_passos +1);
    S=zeros(1,simulador.numero_passos +1);
    R=zeros(1,simulador.numero_passos +1);
    
    I(1) = I0;
    S(1) = S0;
    R(1) = R0;
    
    for i=1:simulador.numero_passos
        
        [I(i+1),S(i+1),R(i+1)] = passo(simulador.modelo,I(i),S(i),R(i));
        
    end
    
endfunction
