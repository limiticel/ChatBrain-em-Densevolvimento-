from Ai_model import *
import os

class App:
    """
    Classe principal da aplicação que gerencia o ciclo de vida da IA conversacional.
    Responsável por:
    - Carregar ou treinar modelos.
    - Executar a interação com o usuário.
    - Salvar modelos treinados.
    """

    def __init__(self):
        """
        Inicializa a aplicação:
        - Instancia o modelo de IA (Brain).
        - Verifica se os modelos treinados já existem.
        - Carrega os modelos se existirem, ou treina-os do zero.
        """
        self.Model = {
            "Brain": Brain(),
            "LimbidSys": Limbid_System()  # Opcional, já incluso em Brain
        }
        self.runnig = True

        if self.Models_Exist():
            self.Model["Brain"].Load_Models()
            print("Modelos carregados com sucesso.")
        else:
            self.Configurate_Ai()

    def Models_Exist(self):
        """
        Verifica se os arquivos dos modelos salvos já existem no diretório apropriado.

        :return: True se os arquivos do modelo límbico e do córtex existirem, False caso contrário.
        """
        return os.path.exists("Models/limbid_model.keras") and os.path.isdir("Models/cortex")

    def Run(self, Details=True):
        """
        Executa o loop principal da aplicação para interação com o usuário via terminal.

        :param Details: Se True, imprime também os detalhes da classe prevista e opções de resposta.
        """
        while self.runnig:
            UserEntry = input("Você: ")
            if UserEntry == 'sair()':
                self.runnig = False
                break

            # Classifica a entrada e seleciona a resposta
            DefinedClass = self.Model["Brain"].Limbid_Prediction(UserEntry, False)
            DefinedResponse = self.Model["Brain"].Cortex_Prediction(int(DefinedClass), UserEntry, False)

            # Mostra detalhes (classe e respostas disponíveis)
            if Details:
                print('Classe:', DefinedClass, self.Model['Brain'].ResponsesForPratice[int(DefinedClass)])
            
            # Mostra a resposta escolhida pelo modelo
            print("Model:", self.Model['Brain'].ResponsesForPratice[int(DefinedClass)][int(DefinedResponse)])

    def Configurate_Ai(self, standardepochs=1500):
        """
        Treina os modelos do sistema límbico e córtex do zero, e os salva no disco para uso posterior.

        :param standardepochs: Número de épocas para treinamento de cada modelo.
        """
        print("Treinando modelos...")
        self.Model['Brain'].Train_Limbid_System(verbose=1, epochs=standardepochs)
        self.Model['Brain'].Train_Cortex_System(verbose=1, epochs=standardepochs)

        # Cria diretório para os modelos se não existir
        os.makedirs("Models/cortex", exist_ok=True)

        # Salva o modelo límbico
        self.Model['Brain']._LimbidSystem.MainModel.save("Models/limbid_model.keras")

        # Salva cada modelo do córtex
        for i, model in self.Model['Brain']._CortexSystem.Models.items():
            model.save(f"Models/cortex/model_{i}.keras")
