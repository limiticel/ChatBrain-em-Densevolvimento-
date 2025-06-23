import random
import string
import unicodedata

class Word_Configuration:
    def __init__(self):
        self.Dicionary={'eu':0,"voce":1,"nos":2}
        self.FillgStandardPadd="<PAD>"
    
    def Simple_Padd(self,sentence, length=8):
        sentence=sentence.split()
        while len(sentence)<length:
                sentence.append(self.FillgStandardPadd)

        return sentence
        

    def Simple_Tokenizer(self,sentence):
        """
        Tokeniza uma frase ou uma lista de frases, separando as palavras por espaços.

        Se a entrada for uma lista de frases (strings), retorna uma lista de listas, 
        onde cada sublista contém as palavras da frase correspondente.
        
        Se a entrada for uma única frase (string), retorna uma lista com as palavras da frase.

        Args:
            sentence (str or list of str): Uma frase ou uma lista de frases a serem tokenizadas.

        Returns:
            list: Lista contendo as palavras separadas por espaços. Se a entrada for uma lista de frases,
                retorna uma lista de listas de palavras.
        
        Example:
            >>> Simple_Tokenizer("gosto de pizza")
            ['gosto', 'de', 'pizza']

            >>> Simple_Tokenizer(["gosto de pizza", "adoro café"])
            [['gosto', 'de', 'pizza'], ['adoro', 'café']]
        """
        new_sentence=[]
        if isinstance(sentence, list):
            for item in sentence:
                value=item.split()
                new_sentence.append(value)
        else:
            new_sentence=''
            new_sentence=sentence.split()

        return new_sentence
    
    def Enumerate_Words(self, list_of_words):
        """
        Atribui valores numéricos únicos e consistentes para cada palavra, usando números decimais aleatórios com duas casas.

        Se a palavra já estiver no dicionário interno (`self.Dicionary`), o valor existente será reutilizado.
        Caso contrário, um novo número aleatório entre 0 e 100 (com duas casas decimais) será gerado e associado à palavra.

        Args:
            list_of_words (str or list of str): Frase (string) ou lista de palavras a serem enumeradas.

        Returns:
            list of float: Lista de números correspondentes às palavras, conforme o dicionário interno.

        Observações:
            - Garante que nenhuma palavra tenha o mesmo valor numérico que outra.
            - Usa `random.uniform(0, 100)` e converte para duas casas decimais com `format`.

        Example:
            >>> self.Dicionary = {}
            >>> Enumerate_Words(["gato", "cachorro"])
            [12.43, 55.29]  # (números gerados aleatoriamente)
            
            >>> Enumerate_Words(["gato"])
            [12.43]  # mesmo número atribuído anteriormente
        """

        if isinstance(list_of_words,str):
            list_of_words=list_of_words.split()
        list_num=[]
        for item in list_of_words:
            if item in self.Dicionary:
                list_num.append(self.Dicionary[item])
            else:
                run=True
                while run:
                    numeral=random.uniform(0,100)
                    numeral=float(format(numeral,".2f"))
                    if numeral not in self.Dicionary.values():
                        self.Dicionary[item]=numeral
                        list_num.append(numeral)
                        run=False
                    else:
                        pass
    
        return list_num 
    
    def Clear_Words(self,sentence):
        """
        Limpa uma frase removendo pontuação, acentos e convertendo todas as letras para minúsculas.

        A função realiza as seguintes etapas:
            1. Converte todo o texto para minúsculas.
            2. Remove acentos e caracteres especiais com `unicodedata`.
            3. Remove pontuações usando `str.translate`.

        Args:
            sentence (str): Frase a ser normalizada e limpa.

        Returns:
            str: Texto limpo, sem acentos, sem pontuação e todo em letras minúsculas.

        Example:
            >>> Clear_Words("Olá, você gosta de Pão?")
            'ola voce gosta de pao'
        """
        sentence=sentence.lower()

        sentence=unicodedata.normalize('NFD', sentence)
        sentence=''.join([c for c in sentence if unicodedata.category(c) != 'Mn'])

        sentence= sentence.translate(str.maketrans('','', string.punctuation))

        return sentence
