import numpy as np
import random
from sklearn.linear_model import LinearRegression
from collections import deque
import pickle
import datetime
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim

def vif_selection(
    x: np.ndarray,
    cols: list,
    max_vif: float = 4,
    deleted_cols=[],
    savepath=None,
    verbose=False,
    initial_pass=False,
):

    min_tol = 1 / max_vif
    idx = 0

    def quick_tolerance_min_idx(x, idx=0):
        tolerance = np.zeros(x.shape[1])

        # to improve speed of finding the intial zeros
        gen = deque(range(x.shape[1]))
        gen.rotate(-idx)
        gen = list(gen)

        for i in gen:
            X, y = np.delete(x, i, 1), x[:, i]
            # https://stackoverflow.com/questions/36573046/difference-between-numpy-linalg-lstsq-and-sklearn-linear-model-linearregression
            r_squared = LinearRegression().fit(X, y).score(X, y)
            tol = 1 - r_squared
            if tol == 0:
                return i, tol
            else:
                tolerance[i] = tol

        idx = np.argmin(tolerance)
        return idx, tolerance[idx]

    if verbose:
        print(len(cols), datetime.datetime.now())

    def check(tol):
        if initial_pass:
            return tol == 0
        else:
            return tol < min_tol

    while True:
        idx, tol = quick_tolerance_min_idx(x, idx)
        if check(tol):
            x = np.delete(x, idx, 1)
            poped_col = cols.pop(idx)
            deleted_cols.extend([poped_col])
            if savepath:
                pickle.dump(deleted_cols, open(savepath, "wb"))

            if verbose:
                print(
                    len(cols), datetime.datetime.now(), f"{tol:.3g}", poped_col
                )
        else:
            break

    return x, cols, deleted_cols

class MLP(nn.Module):
    """
    Classe que define uma Rede Neural Multicamadas (MLP).

    Esta classe implementa uma MLP com camadas totalmente conectadas (lineares) 
    e funções de ativação configuráveis.

    Args:
        - num_dados_entrada (int): O número de unidades na camada de entrada da MLP.
        - individuo (list): Uma lista que define a arquitetura da MLP. Deve ter 4 elementos:
            - O tipo de função de ativação para as camadas ocultas (['relu', 'tanh', 'sigmoid']).
            - O número de unidades na primeira camada oculta.
            - O número de unidades na segunda camada oculta.
            - O número de unidades na camada de saída.
        - num_targets (int): O número de unidades na camada de saída da MLP.

    Métodos:
        - forward(x): Realiza a passagem direta (forward pass) dos dados através da MLP.

    Returns:
        - torch.Tensor: Os dados de saída da MLP após a passagem direta.
    """
    
    def __init__(self, num_dados_entrada, individuo, num_targets):
        super().__init__()

        ativacoes = {
          'relu': nn.ReLU(),
          'tanh': nn.Tanh(),
          'sigmoid': nn.Sigmoid(),
        }

        self.camadas = nn.Sequential(
            nn.Linear(num_dados_entrada, individuo[2]),
            ativacoes[individuo[0]],
            nn.Linear(individuo[2], individuo[3]),
            ativacoes[individuo[0]],
            nn.Linear(individuo[3], num_targets),
        )

    def forward(self, x):
        x = self.camadas(x)
        return x

def gene_neuronios():
    """Sorteia um valor para a quantidade de neurônios, variando de 1 a 20"""
    valores_possiveis = list(range(1, 21))
    gene = random.choice(valores_possiveis)
    return gene

def gene_ativacao():
    """Sorteia uma função de ativação, dentro de 4 opções definidas"""
    valores_possiveis = ['sigmoid', 'tanh', 'relu']
    gene = random.choice(valores_possiveis)
    return gene

def gene_otimizador():
    """Sorteia um otimizador, dentro de 3 opções definidas"""
    valores_possiveis = ['lbfgs','sgd', 'adam']
    gene = random.choice(valores_possiveis)
    return gene

def cria_candidato():
    """Cria uma lista com uma função de ativação, um otimizador, neuronios
    na camada 1, neuronios na camada 2, nesta exata ordem."""
    candidato = []
    camadas = 2
    otimizador = gene_otimizador()
    func_ativacao = gene_ativacao()
    candidato.append(func_ativacao)
    candidato.append(otimizador)

    for _ in range(camadas):
      n_neuronios = gene_neuronios()
      candidato.append(n_neuronios)
    return candidato

def populacao(tamanho):
    """Cria uma população para o problema.

    Args:
      tamanho: tamanho da população

    """
    populacao = []
    for _ in range(tamanho):
      candidato = cria_candidato()
      populacao.append(candidato)
    return populacao

def funcao_objetivo(populacao, redes, otimizadores, NUM_DADOS_DE_ENTRADA, 
                    NUM_DADOS_DE_SAIDA, NUM_EPOCAS, x, y, fn_perda, TAXA_DE_APRENDIZADO):
    """
    Calcula a fitness de cada indivíduo na população utilizando uma rede neural MLP (Multi-Layer Perceptron).
    
    Args:
        - populacao (list): Uma lista de indivíduos representando diferentes arquiteturas de redes neurais MLP.
        - redes (list): Uma lista de dicionários contendo informações sobre as redes neurais já treinadas.
        - otimizadores: Não utilizado nesta versão da função.
        - NUM_DADOS_DE_ENTRADA (int): O número de dados de entrada para a MLP.
        - NUM_DADOS_DE_SAIDA (int): O número de dados de saída para a MLP.
        - NUM_EPOCAS (int): O número de épocas de treinamento da MLP.
        - x (torch.Tensor): Os dados de entrada.
        - y (torch.Tensor): Os dados de saída.
        - fn_perda: A função de perda utilizada para treinamento da MLP.
        - TAXA_DE_APRENDIZADO (float): A taxa de aprendizado utilizada pelo otimizador durante o treinamento.

    Returns:
        - fitness (list): Uma lista contendo a fitness de cada indivíduo na população.
        - redes (list): A lista atualizada de dicionários contendo informações sobre as redes neurais já treinadas.
    """
    
    fitness = []      #Lista para armazenar a fitness de cada indivíduo
    for a in range(len(populacao)):
        individuo = populacao[a]
        individuo_mse = float('inf')

        for b, rede in enumerate(redes):     #Mecanismo de memória
            if individuo == rede['individuo']:
                individuo_mse = rede['mse']
                break
        else:
            minha_mlp = MLP(NUM_DADOS_DE_ENTRADA, individuo, NUM_DADOS_DE_SAIDA)      #Criando a instância da classe da MLP
            
            otimizador = optim.Adam(minha_mlp.parameters(), lr=TAXA_DE_APRENDIZADO)
            minha_mlp.train()  # Definindo a ação para treino da MLP

            for epoca in range(NUM_EPOCAS):  # Treinando a MLP, passando por cada passo
                y_pred = minha_mlp(x)  # Forward pass
                otimizador.zero_grad()  # Zero grad
                loss = fn_perda(y, y_pred)  # Loss
                loss.backward()  # Backpropagation
                otimizador.step()  # Atualiza parâmetros

                if loss.item() < individuo_mse:      #Salvando o MSE
                    individuo_mse = loss.item()

            redes.append({'individuo': individuo, 'mse': individuo_mse})      #Salvando na memória a nova rede testada
        
        individuo_rmse = individuo_mse**(1/2)      #Calulando o RMSE 
        fitness.append(individuo_rmse)      # Adiciona o RMSE do indivíduo (seja encontrado ou treinado) à lista de fitness

    return fitness, redes




def cruzamento_uniforme(pai, mae, chance_de_cruzamento):
    """Realiza cruzamento uniforme

    Args:
      pai: lista representando um individuo
      mae: lista representando um individuo
      chance_de_cruzamento: float entre 0 e 1 representando a chance de cruzamento

    """
    if random.random() < chance_de_cruzamento:
        filho1 = []
        filho2 = []

        for gene_pai, gene_mae in zip(pai, mae):
            if random.choice([True, False]):
                filho1.append(gene_pai)
                filho2.append(gene_mae)
            else:
                filho1.append(gene_mae)
                filho2.append(gene_pai)

        return filho1, filho2
    else:
        return pai, mae

def selecao_torneio_min(populacao, fitness, tamanho_torneio):
    """
    Faz a seleção de uma população usando torneio.

    Nota: da forma que está implementada, só funciona em problemas de
    minimização.

    Args:
      populacao: lista contendo os indivíduos do problema
      fitness: lista contendo os valores computados da função objetivo
      tamanho_torneio: quantidade de indivíduos que batalham entre si

    Returns:
      lista de indivíduos selecionados
    """
    selecionados = []

    for _ in range(len(populacao)):
        # Sorteia índices de 'tamanho_torneio' indivíduos da população
        indices_sorteados = random.sample(range(len(populacao)), tamanho_torneio)
        
        # Encontra o índice do indivíduo com menor fitness entre os sorteados
        fitness_sorteados = [fitness[i] for i in indices_sorteados]
        indice_min_fitness = indices_sorteados[fitness_sorteados.index(min(fitness_sorteados))]
        
        # Adiciona o indivíduo selecionado à lista de selecionados
        selecionados.append(populacao[indice_min_fitness])

    return selecionados



def mutacao_simples(populacao, chance_de_mutacao):
    """Realiza mutação simples

    Args:
      populacao: lista contendo os indivíduos do problema
      chance_de_mutacao: float entre 0 e 1 representando a chance de mutação
      valores_possiveis: lista com todos os valores possíveis dos genes

    """
    valores_possiveis = [
        ['sigmoid', 'tanh', 'relu'],
        ['lbfgs', 'sgd', 'adam'],
        list(range(1, 21))
    ]
    
    for individuo in populacao:
        if random.random() < chance_de_mutacao:
            gene = random.randint(0, len(individuo) - 1)
            valor_gene = individuo[gene]

            if gene == 0:
                valores_sorteio = list(set(valores_possiveis[0]) - set([valor_gene]))
                individuo[gene] = random.choice(valores_sorteio)
            elif gene == 1:
                valores_sorteio = list(set(valores_possiveis[1]) - set([valor_gene]))
                individuo[gene] = random.choice(valores_sorteio)
            else:
                valores_sorteio = list(set(valores_possiveis[2]) - set([valor_gene]))
                individuo[gene] = random.choice(valores_sorteio)

    return populacao