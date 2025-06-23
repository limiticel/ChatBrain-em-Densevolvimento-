from Bibliotecas.Metodos import Word_Configuration
from tensorflow.keras.models import load_model
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ChatConfig import *
from ChatConfig import *




MainConfig=Word_Configuration()

class Brain:
    """
    Classe principal de controle da IA conversacional. 
    Responsável por treinar, prever e gerenciar os modelos neurais principais e secundários.
    """
    def __init__(self):
        """
        Inicializa os componentes do cérebro da IA:
        - Carrega configurações do chat.
        - Inicializa os subsistemas (BruteBrain, Limbid e Cortex).
        - Lê os textos de treinamento.
        - Atualiza os modelos.
        """
        self.ChatConfig=Data()
        self._DataOfBruteBrain=BruteBrain()
        self._LimbidSystem=Limbid_System()
        self._CortexSystem=Cortex_System()
        self.Models={}
        self.SentencesForPratice={}
        self.ResponsesForPratice={}
        self.Read_Texts()
        self.Update_All()

    def Read_Texts(self, userpath="DataAi/UserEntry.json", aipath="DataAi/AiOutput.json"):
        """
        Lê os dados de entrada do usuário e as respostas da IA a partir de arquivos JSON.
        Popula os dicionários internos e atualiza os dados brutos.

        :param userpath: Caminho para o arquivo JSON com entradas do usuário.
        :param aipath: Caminho para o arquivo JSON com respostas da IA.
        """
        import json
        with open(userpath, "r", encoding="utf-8") as f:
            linhas = json.load(f)
        self.SentencesForPratice = {i: linha for i, linha in enumerate(linhas)}

        with open(aipath, "r", encoding="utf-8") as f:
            linhas = json.load(f)
        self.ResponsesForPratice = {i: linha for i, linha in enumerate(linhas)}

        self.Update_Brute_Brain()

    def Update_Brute_Brain(self):
        """
        Atualiza os dados brutos (BruteBrain):
        - Limpa, padroniza e enumera as frases de entrada e saída.
        - Cria a lista de classes com base nos índices dos grupos de frases.
        """
        listofclasses=[]
        temporaryforentry=[]
        temporaryforresponse=[]
        for item in self.SentencesForPratice.keys():
            for sub in self.SentencesForPratice[item]:
                listofclasses.append([item])
        self._DataOfBruteBrain.ListofClasses=listofclasses

        for key in self.SentencesForPratice.values():
            for sentence in key:
                temporaryforentry.append(MainConfig.Simple_Padd(MainConfig.Clear_Words(sentence)))
        for sentence in temporaryforentry:
            self._DataOfBruteBrain.EandCSentences.append(MainConfig.Enumerate_Words(sentence))

        for key in self.ResponsesForPratice.values():
            for sentence in key:
                temporaryforresponse.append(MainConfig.Simple_Padd(MainConfig.Clear_Words(sentence)))
        
        for sentence in temporaryforresponse:
            self._DataOfBruteBrain.EandCResponses.append(MainConfig.Enumerate_Words(sentence))
            
    def Update_All(self):
        """
        Atualiza todos os modelos com base nas configurações e dados carregados:
        - Define o modelo principal (Limbid).
        - Define os modelos por classe (Cortex).
        """
        self._LimbidSystem.Set_Main_Model(self.ChatConfig.ConfigChat.Mnw,len(self.SentencesForPratice.keys()))
        self._CortexSystem.Set_Models(self.ChatConfig.ConfigChat.Mnw,len(self.ResponsesForPratice.keys()),self.ResponsesForPratice)


    def Train_Limbid_System(self, Datax=None, Datay=None, epochs=500,verbose=0):
        """
        Treina o modelo principal (Limbid), responsável por classificar a intenção da frase.

        :param Datax: Dados de entrada para treino (opcional).
        :param Datay: Classes de saída para treino (opcional).
        :param epochs: Número de épocas de treinamento.
        :param verbose: Nível de verbosidade do treinamento.
        """
        Datax=self._DataOfBruteBrain.EandCSentences if not Datax else Datax
        Datay=self._DataOfBruteBrain.ListofClasses if not Datay else Datay

        Datax=np.array(Datax)
        print(Datax)
        Datay=np.array(Datay)

        self._LimbidSystem.MainModel.fit(Datax,Datay, epochs=epochs,verbose=verbose)

    def Limbid_Prediction(self,Entryuser,General=False):
        """
        Realiza uma previsão usando o modelo principal (Limbid) com base na entrada do usuário.

        :param Entryuser: Frase digitada pelo usuário.
        :param General: Se True, retorna a distribuição completa de probabilidade.
        :return: Índice da classe prevista.
        """
        Entryuser=MainConfig.Simple_Padd(MainConfig.Clear_Words(Entryuser))
        Entryuser=MainConfig.Enumerate_Words(Entryuser)
        Entryuser=np.array([Entryuser])
        pred=self._LimbidSystem.MainModel.predict(Entryuser,verbose=0)
        if not General:
            pred=np.argmax(pred, axis=-1)
        else:
            pred=pred
            pred=np.argmax(pred, axis=-1)

        return pred
    
    def Train_Cortex_System(self, Datax=None, Datay=None, epochs=500, verbose=0):
        """
        Treina os modelos secundários (Cortex), um para cada classe de resposta.

        :param Datax: Dados de entrada (opcional, sobrescreve os gerados).
        :param Datay: Dados de saída (opcional).
        :param epochs: Número de épocas de treinamento.
        :param verbose: Verbosidade do treino.
        """
        templistclasses=[]
        templistsentences=[]
        for key in self.ResponsesForPratice.keys():
            templistclasses=[]
            templistsentences=[]
            for i in range(len(self.ResponsesForPratice[key])):
                templistclasses.append([i])
                sentece=MainConfig.Simple_Padd(MainConfig.Clear_Words(self.ResponsesForPratice[key][i]))
                sentece=MainConfig.Enumerate_Words(sentece)
                templistsentences.append(sentece)

            if Datax is None:
                Datax=np.array(templistsentences)

            if Datay is None:
                Datay=np.array(templistclasses)
        

            self._CortexSystem.Models[key].fit(Datax,Datay, epochs=epochs,verbose=verbose)
    
    

    def Cortex_Prediction(self,definedclass, Entryuser, General=False):
        """
        Realiza a previsão da resposta ideal usando o modelo Cortex específico da classe.

        :param definedclass: Índice da classe previamente detectada.
        :param Entryuser: Frase digitada pelo usuário.
        :param General: Se True, retorna a distribuição de probabilidade completa.
        :return: Índice da resposta prevista.
        """
        Entryuser=MainConfig.Simple_Padd(MainConfig.Clear_Words(Entryuser))
        Entryuser=MainConfig.Enumerate_Words(Entryuser)
        Entryuser=np.array([Entryuser])
        pred=self._CortexSystem.Models[definedclass].predict(Entryuser,verbose=0)
        if not General:
            pred=np.argmax(pred, axis=-1)
        else:
            pred=pred

        return pred        

    def Load_Models(self):
        """
        Carrega os modelos treinados previamente do disco:
        - Modelo principal: 'limbid_model.keras'
        - Modelos do córtex: 'cortex_*.keras'
        """
        # Carrega modelo principal
        self._LimbidSystem.MainModel = load_model("Models/limbid_model.keras")

        # Carrega modelos do córtex
        self._CortexSystem.Models = {}
        for file in os.listdir("Models/cortex"):
            if file.endswith(".keras"):
                idx = int(file.split("_")[1].split(".")[0])
                self._CortexSystem.Models[idx] = load_model(os.path.join("Models/cortex", file))




class Limbid_System:
    """
    Classe responsável por definir e armazenar o modelo principal (Límbico).
    Este modelo é usado para classificar a intenção da frase do usuário em uma das categorias disponíveis.
    """
    def __init__(self):
        """
        Inicializa o sistema límbico.
        Atributos:
            - MainModel: armazena a arquitetura e pesos do modelo principal de classificação.
        """
        self.MainModel=None

    def Set_Main_Model(self,SizeofPadd,NumberofCategories):
        """
        Cria e compila o modelo principal (MainModel) com base nos parâmetros fornecidos.

        :param SizeofPadd: Tamanho do padding aplicado nas entradas (usado como shape de entrada).
        :param NumberofCategories: Número de classes de saída (categorias a serem previstas).
        
        Arquitetura:
            - Flatten: transforma a entrada 2D em 1D.
            - Dense(128): camada oculta com 128 neurônios e ativação ReLU.
            - Dropout(0.3): regularização com 30% de taxa.
            - Dense(64): camada oculta com 64 neurônios e ativação ReLU.
            - Dropout(0.3): outra regularização.
            - Dense(NumberofCategories): camada de saída com softmax para classificação.
        """
        self.MainModel= keras.Sequential([
            layers.Flatten(input_shape=(SizeofPadd, )),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(NumberofCategories, activation="softmax")
        ])

        self.MainModel.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

class BruteBrain:
    def __init__(self):
        self.EandCSentences=[]
        self.EandCResponses=[]
        self.ListofClasses=[]

class Cortex_System:
    """
    Sistema responsável por armazenar e configurar os modelos individuais de resposta (córtex).
    Cada classe prevista pelo modelo principal possui um modelo Cortex associado,
    treinado para selecionar a melhor resposta dentro do conjunto de respostas dessa classe.
    """
    def __init__(self):
        """
        Inicializa o sistema do córtex.
        Atributos:
            - Models: dicionário que armazena os modelos por índice de classe (int).
        """
        self.Models={}

    def Set_Models(self,SizeofPadd,NumberofCategories, Dictresponses):
        """
        Cria e compila um modelo separado para cada categoria de saída.

        :param SizeofPadd: Tamanho da entrada (padding) usada nas frases.
        :param NumberofCategories: Quantidade total de categorias/classes detectadas.
        :param Dictresponses: Dicionário com as respostas por categoria. A chave é o índice da classe,
                              e o valor é uma lista com respostas possíveis dessa classe.

        Arquitetura dos modelos:
            - Flatten: achata a entrada.
            - Dense(128): camada densa com ativação ReLU.
            - Dropout(0.3): regularização para evitar overfitting.
            - Dense(N): onde N é a quantidade de respostas possíveis para a categoria, com ativação softmax.
        """
        if not self.Models:
            for i in range(NumberofCategories):
                self.Models[i]=keras.Sequential([
                    layers.Flatten(input_shape=(SizeofPadd,)),
                    layers.Dense(128,activation='relu'),
                    layers.Dropout(0.3),
                    layers.Dense(len(Dictresponses[i]), activation='softmax')
                ])

        
            for obj in self.Models:
                self.Models[obj].compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
    


