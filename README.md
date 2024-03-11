# laboratorioai900
Desafio de projeto Azure Machine Learning


<h1>
  Modelo de previsão Machine Learning para desafio de projeto</span>  
</h1>

Esse modelo de previsão foi criado e treinado com um conjunto de dados de alugueis de bicicletas que prevê um número de alugueis para um determinado dia.


<span>As orientações a seguir estão em </span> https://microsoftlearning.github.io/mslearn-ai-fundamentals/Instructions/Labs/01-machine-learning.html
   
<h2> Criar espaço de trabalho:</h2>
Após acessar o portal https://portal.azure.com  com suas credenciais é necessário criar um espaço de trabalho.

<h3> Selecione  CRIAR UM RECURSO</h3>
Procure por Azure Machine Learning
Preencha os campos com as configurações:</br></br>

**Assinatura**: nome da assinatura sugerido no campo.
</br>**Grupo de recursos** : Selecione ou crie um novo
</br>**Nome** : dê um nome para o grupo de recursos.
</br>**Região** : escolha a região mais próxima.
</br>**Conta de armazenamento** : esse campo é preenchido automaticamente, pode deixar.
</br>**Cofre de chaves** : esse campo é preenchido automaticamente, pode deixar.
</br>**Insights de aplicativo** : esse campo é preenchido automaticamente, pode deixar.
</br>**Registro de conteiner** : _nenhum_.
<h4> Selecione  Revisar + criar e depois Criar</h4></br>

Aguarde a criação do espaço, isso leva alguns minutos.
<h4> Selecione  Launch Studio</h4>
<h4> Selecione  o espaço de trabalho recém-criado</h4>
<h2>Treinando um modelo de previsão com machine learning automatizado</h2>
</br>
<p>Expanda o menu superior e selecione ML automatizado</p>

<p>Selecione Criar trabalho ML automatizado</p>

<h2>Configurações básicas</h2>

**Nome do trabalho** : mslearn-bike-automl.
</br>**Novo nome do experimento** :  mslearn-bike-rental
</br>**Descrição** : Aprendizado de máquina automatizado para previsão de aluguel de bicicletas
</br>**Marcadores** : _nenhum_.
</br> Avançar>

<h2>Tipo de tarefa e dados</h2>

**Selecione o tipo de tarefa** : Regressão
</br>**Selecionar conjunto de dados :** : Criar um conjunto de dados
<h2>Tipo de dados</h2>

**Nome** : alugueldebicicletas
</br>**Descrição** : dados históricos de aluguel de bicicletas
</br>**Tipo** : Tabular
<h2>Fonte de dados</h2>

Selecione **Dos arquivos da web** 
<h2>URL da Web</h2>

</br>**URL da Web** : https://aka.ms/bike-rentals
</br>**Ignorar validação de dados** : _não selecionar_
<h2>Configurações :</h2>

**Formato de arquivo** : Delimitado
</br>**Delimitador** : Vírgula
</br>**Codificação** : UTF-8
</br>**Cabeçalhos de coluna** : Somente o primeiro arquivo possui cabeçalhos
</br>**Pular linhas** : _nenhum_ </br>
**O conjunto de dados contém dados multilinhas** : _não selecionar_

<h2>Esquema :</h2>

Deixar todas colunas selecionadas exceto **Path**
</br> Criar </br>
Após a criação selecione o conjunto de dados **alugueldebicicletas** para enviar o trabalho de ML automatizado. 
<h2>Configurações de tarefa :</h2>

**Tipo de tarefa** : Regressão
</br>**Conjunto de dado** :alugueldebicicletas
</br>**Coluna de destino** : Rentals(Integer)
<h2>Configurações adicionais :</h2>

**Métrica primária** : raiz do erro quadrático médio normalizado
</br>**Explique o melhor modelo** : _não selecionar_
</br>**Usar todos os modelos suportados** : _desmarcado_
</br>**Modelos permitidos** : no combo seleciona as opçoes **_LightGBM_**, **_RandomForest_**
<h2>Limites :</h2>

**Máximo de testes** : 3
</br>**Máximo de testes simultâneos** : 3
</br>**Máximo de nós** : 3
</br>**Limite de pontuação da métrica** : 0,085
</br>**Tempo limite** : 15
</br>**Tempo limite de iteração** : 15
</br>**Habilitar rescisão antecipada** : _selecionar_
<h2>Validação e teste  :</h2>

**Tipo de validação** : divisão de validação de treinamento
</br>**Porcentagem de dados de validação** : 10
</br>**Conjunto de dados de teste** : _nenhum_
<h2>Calcular :</h2>

**Selecione o tipo de computação** : sem servidor
</br>**Tipo de máquina virtual ** : CPU
</br>**Camada de máquina virtual** : Dedicada
</br>**Tamanho da máquina virtual** :Standard_DS3_V2* `(Escolha a máquina que melhor te atende, o valor descrito é cobrado por hora em que a instancia do seu modelo estiver disponível)`
</br>**Número de instâncias** : 1 </br>
Enviar o trabalho de treinamento. `Obs.: é inciado automaticamente, demora alguns minutos.`

<h2>Avaliar o melhor modelo :</h2>

Na guia **Visão Geral** voc terá a visão do trabalho de treinamento 
No lado inferior direito vc terá algumas informações, selecione o melhor modelo que esta logo abaixo de **Algorithm name**.
</br>Na guia **Métricas** você poderá analisar os gráficos gerados
<h2>Implantar e testar :</h2>

Na guia **Modelo** no menu lateral estará o melhor modelo de treinamento selecione **Implantar**, opção **serviço Web**
<h2>Configurações:</h2>

**Nome** : prever-aluguéis
</br>**Descrição** :  Prever aluguel de bicicletas
</br>**Tipo de computação** :  Instância de Contêiner do Azure
</br>**Habilitar autenticação** :  selecionado
</br>A implantação leva alguns minutos em quanto isso o **status de implantação** vai estar **Executando** quando filalizado vai estar **sucesso**
<h2>Teste do modelo implantado:</h2>
No menu lateral selecione <img align="center" alt="Pontos de extremidades" src="https://img.shields.io/badge/Pontos%20de%20extremidades-FFFAFA?style=for-the-badge"> 

na aba **Pontos de extremidades em tempo real** e selecione o modelo
</br> Visualize a guia **Teste**
</br> Em **Dados de entrada para testar os pontos de extremidades** substitua os dados de entrada por

```javascript
  {
   "Inputs": { 
     "data": [
       {
         "day": 1,
         "mnth": 1,   
         "year": 2022,
         "season": 2,
         "holiday": 0,
         "weekday": 1,
         "workingday": 1,
         "weathersit": 2, 
         "temp": 0.3, 
         "atemp": 0.3,
         "hum": 0.3,
         "windspeed": 0.3 
       }
     ]    
   },   
   "GlobalParameters": 1.0
 }
 ```
Clique em  Testar </br>
Resultados do teste `Terá um resultado similar a esse`
O Teste retorna, de acordo com as entradas um número previsto de aluguéis.
```
{
  "Results": [
    348.05644071607327
  ]
}
```
