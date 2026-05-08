function estimador = EstimadorLevenbergMarquardt(numero_passos, I0, S0, R0, dados_observados, max_iter, tolerancia, epsilon, mu, parametros_iniciais)

    estimador = struct();

    estimador.numero_passos = numero_passos;
    estimador.I0 = I0;
    estimador.S0 = S0;
    estimador.R0 = R0;

    estimador.dados_observados = dados_observados;

    estimador.max_iter = max_iter;
    estimador.tolerancia = tolerancia;
    estimador.epsilon = epsilon;

    // Amortecimento inicial
    estimador.mu = mu;

    // Chute inicial
    estimador.parametros_iniciais = parametros_iniciais;

endfunction
