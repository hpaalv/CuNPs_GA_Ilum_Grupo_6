<div align="center">
  <h1>Ilum - Escola de Ciência</h1>
</div>

<div align="center">
  <h2> Aplicação de Algoritmos Genéticos na Otimização de Hiperparâmetros em Modelos de Deep Learning para Predição de Energia de Total e de Formação de Nanopartículas de Cobre </h2>
</div>

<div align="center">
  <h3> Grupo 6 </h3>
</div>

<div align="center">
  <h4>  
  Anna Karen Pinto;
  Beatriz Borges;
  Paulo Henrique dos Santos
  </h4>
</div>

<div align="center">
  <h5> Campinas/2024 </h5>
</div>


# Resumo  

Este trabalho visa explorar e propor um modelo de Algoritmo Genético destinado à otimização dos hiperparâmetros de uma rede neural, identificando quais os melhores hiperparâmetros para serem usados na rede neural que resultarão em uma boa predição. Neste caso será utilizado como exemplo a rede neural construída anteriormente para predição da energia total e de formação de nanopartículas de cobre [13].

Destaca-se que este projeto é desenvolvido como produto da disciplina de Redes Neurais, integrante do curso de Bacharelado em Ciência e Tecnologia, oferecido pela Universidade Ilum - Escola de Ciência, instituição acadêmica vinculada ao CNPEM (Centro Nacional de Pesquisa em Energia e Materiais). 

# Glossário

Dataset: Um conjunto de dados organizados em uma tabela.

Mean Squared Error (MSE): É uma métrica utilizada para avaliar a precisão de um modelo de regressão.

Root Mean Squared Error (RMSE): É uma métrica utilizada para avaliar a precisão de um modelo de regressão. Resulta em um valor que está na mesma unidade dos valores do target.

Variância de Inflação (VIF): Uma medida que ajuda a entender se as informações em um conjunto de dados são redundantes.

Logaritmização: Uma transformação que ajusta os valores dos dados para uma escala diferente, útil para tornar os dados mais fáceis de trabalhar.

Normalização: Uma técnica que coloca os valores dos dados em uma faixa específica, para facilitar o treinamento de modelos.

Função objetivo: Uma função que recebe um indivíduo e retorna o seu valor de aptidão.

Minimização: Processo de encontrar o valor mínimo de uma função objetivo.

Indivíduo: Um candidato para a solução do problema.

Gene: Um parâmetro que pertence a um indivíduo.

População: Um conjunto de candidatos para a solução do problema.

Geração: Cada população em busca genética. A primeira geração é aleatória, e as outras são formadas por seleção, cruzamento e mutação da geração anterior.

Seleção: Processo onde utiliza-se o valor de aptidão dos indivíduos para selecionar quais irão passar seus genes para a geração seguinte.

Cruzamento: Processo onde o material genético de indivíduos selecionados é misturado.

Mutação: Processo onde os genes dos indivíduos selecionados têm uma chance de alterar seu valor.

Hall da Fama: Conjunto dos n indivíduos que obtiveram os melhores valores de aptidão durante o processo de busca.



# Importando Dados 

* Baixe o arquivo intitulado "mlp_final.ipynb" desse github.
* Acesse o link: <https://data.csiro.au/collection/csiro:42598>.[1].
* Baixe o arquivo com o dataset e coloque-o na mesma pasta que o arquivo "mlp_final.ipynb".
* Não é necessário renomear o arquivo de dados.

OBS.: O dataset utilizado e a lista com todos os atributos presentes no dataset estão armazenados neste repositório em arquivos intitulados "mlp_final.ipynb" e "Cu_nanoparticle_headerlist.pdf", respectivamente.



# Introdução

Materiais em escala nanométrica exibem características distintas em comparação com materiais em escalas macrométricas, devido a uma série de fatores, incluindo a superfície exposta dos materiais. Em escalas nanométricas, a relação entre área de superfície e volume é amplificada, tornando a superfície de contato proporcionalmente maior em relação ao volume do material. Essa proporção aumentada da superfície confere propriedades aos materiais nanoestruturados. [2]

Os nanomateriais possuem uma variedade de aplicações, incluindo catálise, imagiologia por ressonância magnética e liberação controlada de fármacos. Além disso, processos de modificação superficial podem ser empregados para mitigar os efeitos citotóxicos associados a certos materiais. Essas modificações podem incluir revestimentos ou funcionalizações que tornam a interação com o ambiente biológico mais favorável, reduzindo assim os efeitos adversos. São classificadas como nanométricas partículas com dimensões tipicamente entre 1-100 nm [2][3][4]. Portanto, é fundamental entender como a energia total e de formação influencia no produto final e nas características para a qual a nanopartícula será designada, permitindo a implementação de medidas preventivas e o planejamento adequado, incentivando um investimento tecnológico e científico maior nesta área.

Com essas questões, foi feita uma rede neural anteriormente para a predição dos atributos de saída (valor da energia total e de formação da nanopartícula) [13]. Entretanto, observou-se que a métrica RMSE (Root Mean Square Error) utilizada nessa rede neural não era o ideal, informando que a performance deste modelo de redes neurais não está bom/preciso o suficiente.

Uma forma de melhorar estes parâmetros é fazendo uma otimização dos hiperparâmetros do modelo. A ciência que pode ser utilizada para esta problemática é os Algoritmos Genéticos. Essa técnica de busca e otimização baseia-se nos princípios da seleção natural e evolução biológica, sendo uma ótima ferramenta para solucionar problemas de otimização.

A rede neural recebe os dados na camada de entrada com seus respectivos pesos. Cada neurônio possui uma função de ativação e um viés ao qual realizará cálculos. Durante o processo de aprendizado, os pesos de conexão na rede são ajustados após o processamento de cada dado com base na quantidade de erro na saída em comparação com o resultado esperado. O qual permite que um sistema aprenda e melhore de forma autônoma, sem ser programado explicitamente, alimentando-o com grandes quantidades de dados. [5] 

O Algoritmo Genético funciona nos seguintes passos [15]:
1. Criação da população inicial (que será aleatória, de acordo com os dados do dataset);
2. Cálculo da função objetivo para todos os membros da população inicial e atualização do Hall da Fama;
3. Seleção dos indivíduos (que seguem para a geração seguinte);
4. Cruzamento dos indivíduos selecionados (gerando troca de material genético entre os indivíduos);
5. Mutação dos indivíduos da população recém-criada (possibilidade de trazer informação nova ao sistema);
6. Cálculo da função objetivo para todos os membros da população recém-criada e atualização do Hall da Fama;
7. Checagem de critérios de parada (caso não tenham sido atendidos, retorna-se ao passo 3);
8. Retorno ao usuário do Hall da Fama.

Para automatizar processos na interface do usuário, foi implementada uma ferramenta poderosa: os scripts. Eles podem realizar todas as ações que seriam feitas com o mouse ou teclado, tornando-se ideais para automatizar tarefas altamente repetitivas ou demoradas que, de outra forma, exigiriam muito tempo e esforço manual. [14]



# Metodologia

Inicialmente, procedeu-se com a importação das bibliotecas necessárias e dos dados, os quais foram baixados na referência [1], foram carregados em um Dataframe da biblioteca Pandas e aplicado o método "dropna" - Responsável por remover as linhas que contêm valores ausentes (NaN) do DataFrame [7] . Essa etapa visava preparar o terreno para uma análise e predição, com dados relevantes. Além disso um documento contendo os significados e as unidades de cada atributo está presente neste diretório para aqueles que desejam entender melhor os dados aplicados no projeto. 

Durante a análise detalhada, identificou-se que o conjunto continha muitos dados e grande parte deles eram valores nulos, por isso a implementação do método "dropna" foi necessária. Os dados restantes foram divididos entre features e targets, cujos targets são os valores de energia total e energia de formação das nanopartículas de cobre, e após em treino e teste. As porcentagens usadas como parâmetros para tal atividade foram definidas como 90% para treino e 10% para teste, com a semente aleatória sendo 10. A semente aleatória é um número utilizado para inicializar o gerador de números aleatórios garantindo que os resultados de operações que envolvem aleatoriedade possam ser reproduzíveis.[8]

Durante uma sessão de instrução em sala de aula, o docente sugeriu a aplicação de normalização e logaritmização nos dados, para a redução da dimensionalidade. Após a execução do procedimento dropna, onde as linhas contendo valores NaN foram eliminadas, procedeu-se à segunda etapa de logaritmização. Contudo, foi constatado que muitos dos dados continham valores nulos. O logaritmo de 0 resulta em uma indefinição matemática. Nas bibliotecas utilizadas para o cálculo do logaritmo, esse resultado é representado por um valor NaN. Consequentemente, surgiram desafios durante o treinamento da rede, devido à disparidade na quantidade de dados entre o conjunto X e Y, resultando como opção para o grupo não realizar a logaritmização.

Porém a normalização foi feita sem problema algum. A escolhida foi a normalização pelo máximo absoluto, que consiste em um método de pré-processamento de dados ao qual cada valor presente no conjunto é submetido a uma divisão pelo valor máximo encontrado. Isso resulta em uma escala onde o valor máximo é 1 e os demais valores são proporcionais a esse máximo.

O conjunto foi submetido a uma análise de multicolinearidade (Seleção VIF - Variance Inflation Factor). A multicolinearidade existe no momento em que duas ou mais variáveis independentes em um modelo de regressão múltipla apresentam alta correlação entre si. Quando algumas características são muito correlacionadas, pode-se ter dificuldade em diferenciar entre seus efeitos individuais sobre a variável dependente, ou seja, quando há multicolinearidade significativa entre as variáveis independentes em um modelo, isso pode introduzir viés nos coeficientes de regressão e afetar a interpretabilidade do modelo. O VIF funciona calculando a multicolinearidade de cada variável independente (coluna) em relação às outras variáveis independentes e com isso ele retorna um valor, quanto maior o valor do VIF para uma variável independente, maior é a multicolinearidade dessa variável com as outras variáveis independentes. [9]

O VIF implementado no código presente neste repositório foi disponibilizado pelo docente Daniel Roberto Cassar, que ministra a Disciplina de Redes Neurais e Algoritmos Genéticos da Universidade Ilum - Escola de Ciência. O seu funcionamento procede da seguinte maneira:

O algoritmo realiza uma seleção de variáveis com base no VIF. Ele recebe um array NumPy e uma lista contendo os dados e os nomes das variáveis independentes, respectivamente. Possui também um limite máximo para o VIF. Variáveis com VIF maior que esse valor serão removidas e armazenadas numa lista. O processo segue até que todas as variáveis tenham um VIF abaixo do limite especificado. Por fim, a função retorna os dados das variáveis independentes atualizadas, a lista atualizada de nomes de variáveis e a lista de variáveis removidas. O código dele se encontra no script.

Em seguida foi realizada a MLP [5]. O Multilayer Perceptron (MLP) é um tipo de rede neural artificial composta por várias camadas, incluindo uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída. Cada camada é formada por neurônios interconectados, cada um com sua própria função de ativação e viés.[10] 

Como o desejado era fazer uma comparação entre diferentes arquiteturas desse modelo de predição para encontrar o qual possui os melhores valores de hiperparâmetro, logo melhor valor de métrica, foi variado no modelo o tipo de função de ativação (Sigmoid, Tanh ou ReLU), o tipo de otimizador (L-BFGS, SGD ou Adam), quantidade de neurônios (de 1 a 20) em 2 camadas fixas. OBS1: Se encontra no script as funções que geram aleatoriamente o número de neurônios, o tipo de otimizador e tipo de função de ativação. OBS2: Diferencial sobre cada tipo de função e otimizador mais detalhado no Notebook Jupyter.

Com os hiperparâmetros definidos para a Rede Neural, agora inicia-se o código para a implementação do Algoritmo Genético de forma a solucionar nosso problema de otimização. Para isso, criou-se quatro variáveis, uma para armazenar todos os hiperparâmetros (a rede sem si) obtidos, outra que armazena todos os valores do MSE obtidos [12] (métrica adotada pelo grupo), uma para o Hall da Fama exibindo a melhor Rede Neural predita e outra para o Hall da Fama exibindo o melhor RMSE obtido (Hall do RMSE). Assim, sempre que uma nova rede fosse criada, o melhor valor do RMSE dessa rede é comparado com o armazenado na variável fixa, caso esse valor fosse menor que o armazenado, ele o substitui. Assim, a rede com o valor menor também substitui a rede já colocada na variável que armazena os hiperparâmetros.

As funções relacionadas ao Algoritmo Genético estão no script. 
1. Primeiro, tem-se a função "cria_candidato" para a criação de candidatos aleatórios para a solução de nosso problema, sendo os indivíduos da nossa primeira geração. Cada indivíduo terá 4 genes: o tipo de função de ativação, o tipo de otimizador, número de neurônios na primeira camada e número de neurônios na segunda camada, respectivamente.
2. Tem-se a função "populacao" que vai gerar uma lista contendo a população com n indivíduos desejados.
3. A função "funcao_objetivo" é a função que calculará o fitness de cada indivíduo na população junto com a Rede Neural MLP, sendo onde a MLP será rodada e já retornará a lista contendo as redes e seus valores de fitness.
4. Utilizando a função "selecao_torneio_min", será sorteado indivíduos para avaliar qual possui o menor valor de fitness e adicioná-los a uma lista.
5. Com a função "cruzamento_uniforme", tem o código relacionado ao cruzamento uniforme aplicado neste Algoritmo Genético, onde é pego genes aleatórios da mãe e do pai para a formação dos indivíduos filhos.
6. Função "mutacao_simples" que, com uma semente aleatória, irá mutar aleatoriamente um dos genes do indivíduo.

Por fim, a rede com melhor valor da métrica RMSE é selecionada, atualizando as duas listas do Hall da Fama de redes e de métricas.

# Resultados e Discussões

Após o treinamento e teste de diferentes arquiteturas da Rede Neural MLP para modelagem de nanomateriais, aplicando o Algoritmo Genético para otimização de seus hiperparâmetros, observou-se resultados promissores em termos de melhores valores da métrica de RMSE. Foi utilizado uma variedade de arquiteturas, ajustando o tipo de função de ativação, tipo de otimizador e número de neurônios em cada uma das duas camadas para encontrar a configuração mais adequada.

Esses resultados validam a eficácia da abordagem de Algoritmos Genéticos para problemas de otimização e sugerem que esta técnica pode ser aplicada com sucesso em uma variedade de problemas relacionados a modelos de predição de materiais em escala nanométrica.

# Conclusão

Neste projeto, foi explorado a importância de analisar a precisão das predições utilizando cada tipo de otimizador, analisando qual seria a melhor otimização para nosso problema: melhoria dos hiperparâmetros para minimização da métrica RMSE.

Após obtidos os dados de métrica RMSE da Rede Neural MLP utilizada anteriormente, destacou-se o uso de Algoritmos Genéticos como uma ferramenta de otimização para minimizar essa métrica. Ao aplicar Algoritmos Genéticos para ajustar os hiperparâmetros da rede neural, pode-se explorar de forma eficiente o espaço de busca e encontrar boas configurações que resultem em previsões mais precisas e uma melhor compreensão dos fatores que influenciam as propriedades dos nanomateriais na rede neural. Essa abordagem permite não apenas melhorar o desempenho da rede neural, mas também otimizar o design dos nanomateriais para aplicações específicas, maximizando assim o seu potencial em diversas áreas da Ciência e da Tecnologia.

Em suma, a utilização de Algoritmos Genéticos para minimizar a métrica RMSE de uma rede neural aplicada a nanomateriais, em conjunto com a manipulação de datasets estruturados, representa uma abordagem poderosa e interdisciplinar para avançar nosso entendimento e aplicação desses materiais em escala nanométrica. Espera-se que este trabalho inspire mais pesquisas e investimentos na área, levando a avanços significativos e inovações em Nanotecnologia e Ciência dos Materiais.

# Referências

[1] Copper Nanoparticle Data Set. Disponível em: <https://data.csiro.au/collection/csiro:42598>. Acesso em: 09 abr. 2024.

[2] OS NANOMATERIAIS E A DESCOBERTA DE NOVOS MUNDOS NA BANCADA DO QUÍMICO  |  Manuel A. Martins e Tito Trindade - Quim. Nova, Vol. 35, No. 7, 1434-1446, 2012. Disponível em: <https://www.scielo.br/j/qn/a/P8tgywDnt7nS6tGyHdQ3BCF/>. Acesso em: 02 mai. 2024.

[3] Ojha, N. K.; Zyryanov, G. V.; Majee, A.; Charushin, V. N.; Chupakhin, O. N.; Santra, S. Copper nanoparticles as inexpensive and efficient catalyst: A valuable contribution inorganic synthesis. Coordination Chemistry Reviews 2017, 353, 1–57.11.

‌
[4] Ssekatawa K, Byarugaba DK, Angwe MK, Wampande EM, Ejobi F, Nxumalo E, Maaza M, Sackey J, Kirabira JB. Phyto-Mediated Copper Oxide Nanoparticles for Antibacterial, Antioxidant and Photocatalytic Performances. Front Bioeng Biotechnol. 2022 Feb 16;10:820218. doi: 10.3389/fbioe.2022.820218. PMID: 35252130; PMCID: PMC8889028.

‌
[5] Multilayer perceptron | Wikipedia, the free encyclopedia. Disponível em <https://en.wikipedia.org/wiki/Multilayer_perceptron#:~:text=A%20multilayer%20perceptron%20(MLP)%20is,that%20is%20not%20linearly%20separable.>. Acesso em: 29 abr. 2024.

[6] Dados estruturados de conjunto de dados | Central da Pesquisa Google | Documentação. Disponível em: <https://developers.google.com/search/docs/appearance/structured-data/dataset?hl=pt-br>. Acesso em: 11 nov. 2023.
‌
[7] Pandas. DataFrame.dropna | Pandas | Documentação. Disponível em: <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html>. Acesso em: 02 mai. 2024.

[8] Python Random seed() Method | w3 schools. Disponível em: <https://www.w3schools.com/python/ref_random_seed.asp>. Acesso em: 02 mai. 2024.

[9] Detecting Multicollinearity with VIF – Python | geeks for geeks. Disponível em: <https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/>. Acesso em: 02 mai. 2024.

[10] Perceptron Multi-Camadas (MLP) | icmc usp. Disponível em: <https://sites.icmc.usp.br/andre/research/neural/MLP.htm>. Acesso em: 03 mai. 2024.

[11] Relu | Função de ativação:
https://www.deeplearningbook.com.br/funcao-deativacao/#:~:text=ReLU%20é%20a%20função%20de,neurônios%20ativados%20pela%20função%20ReLU. Acesso em: 03 mai. 2024.

[12] Métricas para Regressão: Entendendo as métricas R², MAE, MAPE, MSE e RMSE | medium. Disponível em: <https://medium.com/data-hackers/prevendo-n%C3%BAmeros-entendendo-m%C3%A9tricas-de-regress%C3%A3o-35545e011e70>. Acesso em: 03 mai. 2024.

[13] HPAALV. hpaalv/CuNPs_MLP_Ilum_Grupo_6. Disponível em: <https://github.com/hpaalv/CuNPs_MLP_Ilum_Grupo_6>. Acesso em: 23 maio. 2024.

[14] Visão geral de script. Disponível em: <https://www.ibm.com/docs/pt-br/spss-modeler/18.4.0?topic=language-scripting-overview>. Acesso em: 27 maio. 2024.

‌[15] Cassar, D. (2024). ATP-303 GA 2.3 - Notebook algoritmo genético [Notebook Jupyter]. Arquivo pessoal.
