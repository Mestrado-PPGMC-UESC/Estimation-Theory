function estimador = EstimadorGaussNewton(dados_observados,numero_passos,max_iter,chutes_iniciais,tolerancia,epsilon)
    
    estimador = struct();
    estimador.chutes_iniciais = chutes_iniciais;
    estimador.dados_observados=dados_observados;
    estimador.numero_passos = numero_passos;
    estimador.max_iter=max_iter;
    estimador.tolerancia = tolerancia;
    estimador.epsilon = epsilon;
    estimador.I0 = dados_observados(1,1);
    estimador.S0 = dados_observados(1,2);
    estimador.R0 = dados_observados(1,3);
        
endfunction

function [parametros,historico_beta,historico_alpha,historico_k,historico_erro,iteracoes] = estimar(estimador)

    parametros = estimador.chutes_iniciais;
    
    historico_beta = [];
    historico_alpha = [];
    historico_k = [];
    historico_erro =[];
    
    for iteracao=1:max_iter
        
        residuo = calcular_residuo(estimador,parametros);
        
        J = calcular_jacobiana_numerica(estimador, parametros, residuo);

        erro = norm(residuo);

        historico_beta($+1) = parametros(1);
        historico_alpha($+1) = parametros(2);
        historico_k($+1) = parametros(3);
        historico_erro($+1) = erro;

        // Resolve: (J'J) delta = J'r
        delta = (J' * J) \ (J' * residuo);

        [parametros, parar] = atualizar_parametros_com_protecao(estimador, parametros, delta);

        if parar then
            break;
        end

    end

    iteracoes = iteracao;

endfunction


function residuo = calcular_residuo(estimador,parametros)
    
    simulacao = simular_com_parametros(estimador,parametros);
    residuo_matriz = estimador.dados_observados - simulacao;
    residuo = matrix(residuo_matriz,-1,1);
    
endfunction

function simulacao = simular_com_parametros(estimador,parametros)
    
    beta = parametros(1);
    alpha = parametros(2);
    k = parametros(3);
    
    modelo = ModeloPropagacaoInformacao(beta,alpha,k);
    
    simulador = Simulador(modelo);
    
    [I,S,R] = executar(simulador,estimador.I0,estimador.S0,estimador.R0);
    
    simulacao = [I'S'R'];
    
endfunction


function J = calcular_jacobiana_numerica(estimador, parametros, residuo_base)

    numero_residuos = length(residuo_base);
    numero_parametros = length(parametros);

    J = zeros(numero_residuos, numero_parametros);

    for j = 1:numero_parametros

        parametros_perturbados = parametros;

        parametros_perturbados(j) = parametros_perturbados(j) + estimador.epsilon;

        residuo_perturbado = calcular_residuo(estimador, parametros_perturbados);

        J(:, j) = (residuo_perturbado - residuo_base) / estimador.epsilon;

    end

endfunction


function [parametros_novos, parar] = atualizar_parametros_com_protecao(estimador, parametros, delta)

    parametros_novos = parametros - delta;

    parametros_novos = max(parametros_novos, 1e-8 * ones(parametros_novos));

    if or(isnan(parametros_novos)) | or(isinf(parametros_novos)) then
        parametros_novos = parametros;
        parar = %t;
        return;
    end

    if norm(delta) < estimador.tolerancia then
        parar = %t;
    else
        parar = %f;
    end

endfunction



