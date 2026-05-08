clc;
clear;

// Carrega parâmetros

exec("simulators/simulador.sci",-1);
exec("visualization/plotador.sci",-1);
exec("models/modelo_propagacao.sci",-1);
exec("estimators/estimador_gauss_newton.sci",-1);


//DADOS

// Parâmetros do modelo
beta = 0.04;
alpha = 0.03;
k = 0.8;

// Condições iniciais
I0 = 0.9;
S0 = 0.1;
R0 = 0.0;

// Número de passos
numero_passos = 500;

// Parâmetros do Gauss-Newton
max_iter = 500;
tolerancia = 1e-8;
epsilon = 1e-6;

// Chute inicial
chutes_iniciais = [0.035; 0.025; 3.0]; //beta,alpha,k


////////////////////////
//1. Simulação do modelo
////////////////////////

// Cria modelo
modelo = ModeloPropagacaoInformacao(beta, alpha, k);

// Constroi simulador
simulador = Simulador(modelo,numero_passos);

// Executa
[I, S, R] = executar(simulador,I0,S0,R0);

//Cria plotador
plotador = Plotador();

//Plota sistema
plotar_sistema(plotador,I,S,R,"Propagação da Informação");


////////////////////////
//2. Estimação por Gauss-Newton
////////////////////////

estimador = EstimadorGaussNewton([I' S' R'],numero_passos,max_iter,chutes_iniciais,tolerancia,epsilon);

[parametros, hist_beta, hist_alpha, hist_k, hist_erro, iteracoes] = estimar(estimador);

disp("Parametros estimados:");
disp(parametros);
disp(iteracoes);

////////////////////////
//3. Resimulação com parâmetros estimados
////////////////////////

beta_estimado = parametros(1);
alpha_estimado = parametros(2);
k_estimado = parametros(3);

// Cria modelo estimado
modelo_estimado = ModeloPropagacaoInformacao(beta_estimado, alpha_estimado, k_estimado);

// Cria simulador estimado
simulador_estimado = Simulador(modelo_estimado, numero_passos);

// Executa simulação estimada
[I_est, S_est, R_est] = executar(simulador_estimado, I0, S0, R0);

// Plota resultado estimado
plotar_sistema(plotador, I_est, S_est, R_est,"Propagação da Informação - Estimado Gauss-Newton");

////////////////////////
// 4. Convergência
////////////////////////

plotar_convergencia(plotador, hist_beta, hist_alpha, hist_k, hist_erro, "Gauss-Newton");





