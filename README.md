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

Este trabalho visa explorar e propor um modelo de algoritmo genético destinado à otimização dos hiperparâmetros de uma rede neural, identificando quais os melhores hiperparâmetros para serem usados na rede neural que resultarão em uma boa predição. Neste caso será utilizado como exemplo a rede neural construída anteriormente para predição da energia total e de formação de nanopartículas de cobre [13].

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

Os nanomateriais possuem uma variedade de aplicações, incluindo catálise, imagiologia por ressonância magnética e liberação controlada de fármacos. Além disso, processos de modificação superficial podem ser empregados para mitigar os efeitos citotóxicos associados a certos materiais. Essas modificações podem incluir revestimentos ou funcionalizações que tornam a interação com o ambiente biológico mais favorável, reduzindo assim os efeitos adversos. São classificadas como nanométricas partículas com dimensões tipicamente entre 1-100 nm [2][3][4]. Portanto, é fundamental entendermos como a energia total e de formação influencia no produto final e nas características para a qual a nanopartícula será designada, permitindo a implementação de medidas preventivas e o planejamento adequado, incentivando um investimento tecnológico e científico maior nesta área.

Com essas questões, foi feita uma rede neural anteriormente para a predição dos atributos de saída (valor da energia total e de formação da nanopartícula) [13]. Entretanto, observou-se que a métrica RMSE (Root Mean Square Error) utilizada nessa rede neural não era o ideal, informando que a performance deste modelo de redes neurais não está bom/preciso o suficiente.

Uma forma de melhorar estes parâmetros é fazendo uma otimização dos hiperparâmetros do modelo. A ciência que pode ser utilizada para esta problemática é os Algorítmos Genéticos. Essa técnica de busca e otimização basea-se nos princípios da seleção natural e evolução biológica, sendo uma ótima ferramenta para solucionar problemas de otimização.

]]]]]]]]A rede recebe os dados na camada de entrada com seus respectivos pesos. Cada neurônio possui uma função de ativação e um viez ao qual realizará cáculos. Durante o processo de aprendizado, os pesos de conexão na rede são ajustados após o processamento de cada dado com base na quantidade de erro na saída em comparação com o resultado esperado. O qual permite que um sistema aprenda e melhore de forma autônoma, sem ser programado explicitamente, alimentando-o com grandes quantidades de dados. [5] 

O algoritmo genético funciona nos seguintes passos:
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

Durante uma sessão de instrução em sala de aula, o docente sugeriu a aplicação de normalização e logaritmização nos dados, para a redução da dimensionalidade. Após a execução do procedimento dropna, onde as linhas contendo valores NaN foram eliminadas, procedeu-se à segunda etapa de logaritmização. Contudo, foi constatado que muitos dos dados continham valores nulos. O logaritmo de 0 resulta em uma indefinição matemática. Nas bibliotecas utilizadas para o cálculo do logaritmo, esse resultado é representado por um valor NaN. Consequentemente, surgiram desafios durante o treinamento da rede, devido à disparidade na quantidade de dados entre o conjunto X e Y resultando como opçõa para o grupo não realizar a logarimização.

Porém a normalização foi feita sem problema algum. A escolhida foi a normalização pelo máximo absoluto, que consiste em um método de pré-processamento de dados ao qual cada valor presente no conjunto é submetido a uma divisão pelo valor máximo encontrado. Isso resulta em uma escala onde o valor máximo é 1 e os demais valores são proporcionais a esse máximo.

O conjunto foi submetido a uma análise de multicolinearidade (Seleção VIF - Variance Inflation Factor). A multicolinearidade existe no momento em que duas ou mais variáveis independentes em um modelo de regressão múltipla apresentam alta correlação entre si. Quando algumas características são muito correlacionadas, pode-se ter dificuldade em diferenciar entre seus efeitos individuais sobre a variável dependente, ou seja, quando há multicolinearidade significativa entre as variáveis independentes em um modelo, isso pode introduzir viés nos coeficientes de regressão e afetar a interpretabilidade do modelo. O VIF funciona calculando a multicolinearidade de cada variável independente (coluna) em relação às outras variáveis independentes e com isso ele retorna um valor, quanto maior o valor do VIF para uma variável independente, maior é a multicolinearidade dessa variável com as outras variáveis independentes. [9]

O VIF implementado no código presente neste repositório foi disponibilizado pelo docente Daniel Roberto Cassar, que ministra a Disciplina de Redes Neurais e Algoritmos Genéticos da Universidade Ilum - Escola de Ciência. O seu funcionamento procede da seguinte maneira:

O algoritmo realiza uma seleção de variáveis com base no VIF. Ele recebe um array NumPy  e uma lista contendo os dados e os nomes das variáveis independentes, respectivamente. Possui também um limite máximo para o VIF. Variáveis com VIF maior que esse valor serão removidas e armazenadas numa lista. O processo segue até que todas as variáveis tenham um VIF abaixo do limite especificado. Por fim, a função retorna os dados das variáveis independentes atualizadas, a lista atualizada de nomes de variáveis e a lista de variáveis removidas.

Em seguida foi realizada a MLP [5]. O Multilayer Perceptron (MLP) é um tipo de rede neural artificial composta por várias camadas, incluindo uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída. Cada camada é formada por neurônios interconectados, cada um com sua própria função de ativação e viés.[10]

O processo de funcionamento do MLP envolve a propagação da informação, que possui pesos, através das camadas, começando pela entrada, onde os neurônios processam os dados com base em seus viés e funções de ativação. Essa informação é então transmitida para as camadas ocultas, onde o processo é repetido, podendo possuir mais de uma camada oculta, até chegar à camada de saída, que produz os resultados finais da rede neural.[10]

Um detalhe importante a se mencionar é que nossa função de ativação não é a Sigmoide, utilziada por padrão. Decidimos, por meio dos testes, utilizar a Relu, pois esta mostrou-se mais eficaz em promover a convergência mais rápida durante o treinamento, além de ajudar a mitigar problemas de desvanecimento de gradiente. A função de ativação ReLU (Rectified Linear Unit) é conhecida por sua simplicidade e eficácia, pois ativa os neurônios quando o valor de entrada é positivo e os desativa quando é negativo, facilitando o aprendizado de representações mais discriminativas nos dados. Isso pode resultar em um treinamento mais rápido e em melhores desempenhos em muitos casos, tornando-a uma escolha popular em arquiteturas de redes neurais profundas.[11]

Durante o treinamento do MLP, utiliza-se o método de backpropagation para ajustar os pesos das conexões entre os neurônios. Se a saída da rede não corresponde à esperada, é calculado um erro, que é então retropropagado da camada de saída até a camada de entrada através de derivadas parciais. Os pesos das conexões são modificados de acordo com o erro propagado.[12]

O treinamento supervisionado do MLP ocorre em dois passos. Primeiro, um padrão é apresentado à camada de entrada, e a resposta é calculada até a camada de saída. Em seguida, o erro é propagado de volta para ajustar os pesos das conexões, repetindo esse processo até que o erro seja minimizado e a rede neural produza resultados precisos.[12].

Para a escolha dos hiperparâmetros, foi pensada a necessidade de uma rede que variasse a quantidade de neurônios e camadas ocultas, para que a melhor configuração possível dos dados fosse encontrada. Os hiperparâmetros foram definidos em intervalos para que a MLP pudesse variar em diferentes arquiteturas a sua estrutura, porém sempre com os mesmos dados de entrada e saída, sem alterar a quantidade de neurônios nessas camadas nas diversas conformações.

Com os hiperparâmetros definidos, um loop que inteirou de forma a criar e testar várias redes foi inserido no código, vairando a quantidade de camadas e neurônios como proposto. Para isso, criou-se duas variáveis, uma para armazenar os hiperparâmetros (a rede sem si) e outra que armazena o valor do MSE [12] (métrica adotada pelo grupo). Assim, sempre que uma nova rede fosse criada, o melhor valor do RMSE dessa rede é comparado com o armazenado na variável fixa, caso esse valor fosse menor que o armazeado, ele o substitui. Assim, a rede com o valor menor também substitui a rede já colocada na variável que armazena os hiperparâmetros.

Por fim, a rede com melhor arquitetura é selecionada e, treinada, novamente por uma quantidae maior de eras buscando minimizar ainda mais o RMSE.

# Resultados e Discussões

(corrigir em prol dos ga) Após o treinamento e teste de diferentes arquiteturas de redes neurais MLP para modelagem de nanomateriais, observamos resultados promissores em termos de desempenho. Utilizamos uma variedade de arquiteturas, ajustando o número de camadas e neurônios em cada camada para encontrar a configuração mais adequada.

Esses resultados validam a eficácia da abordagem de algoritmos genéticos para problemas de otimização e sugerem que esta técnica pode ser aplicada com sucesso em uma variedade de problemas relacionados a modelos de predição de materiais em escala nanométrica.

# Conclusão

Nesta revisão, exploramos a importância dos nanomateriais e sua vasta gama de aplicações em diversas áreas, desde a catalisação até a medicina. A manipulação precisa das propriedades dos nanomateriais tem sido uma área de pesquisa em crescimento devido ao seu potencial para revolucionar tecnologias existentes e criar novas soluções para desafios atuais.

Após obter o dado de métrica RMSE da Rede Neural MLP utilizada anteriormente, destacou-se o uso de Algoritmos Genéticos como uma ferramenta de otimização para minimizar essa métrica. Ao aplicar algoritmos genéticos para ajustar os hiperparâmetros da rede neural, pode-se explorar de forma eficiente o espaço de busca e encontrar boas configurações que resultem em previsões mais precisas e uma melhor compreensão dos fatores que influenciam as propriedades dos nanomateriais na rede neural. Essa abordagem nos permite não apenas melhorar o desempenho da rede neural, mas também otimizar o design dos nanomateriais para aplicações específicas, maximizando assim o seu potencial em diversas áreas da Ciência e da Tecnologia.

Em suma, a utilização de algoritmos genéticos para minimizar a métrica RMSE de uma rede neural aplicada a nanomateriais, em conjunto com a manipulação de datasets estruturados, representa uma abordagem poderosa e interdisciplinar para avançar nosso entendimento e aplicação desses materiais em escala nanométrica. Esperamos que este trabalho inspire mais pesquisas e investimentos na área, levando a avanços significativos e inovações em nanotecnologia e ciência dos materiais.

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
