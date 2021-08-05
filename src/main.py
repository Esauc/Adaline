import pandas as pd

from activation_function import SignFunction
from adaline import Adaline


print('------------------------------------')

dataset = pd.read_csv('database/dataset-treinamento.csv')

n = 4 #Número de entradas

# dataset.iloc[LINHA INICIAL (Inclusive) : LINHA FINAL (exclusive) , COLUNA INICIAL (inclusive) : COLUNA FINAL (exclusive)]
#Só : significa todas as linhas/colunas
X = dataset.iloc[:, 0:n].values # ENTRADAS
d = dataset.iloc[:, n:].values  #SAÍDAS

a = Adaline(X, d, 0.0025, (10**-6), SignFunction)
#X valores de entrada
#d valores de saída

a.train()


#t = 1 #TREINAMENTO 1
#a.theta = -1.73045059
#a.W = [1.30503106, 1.60317264, -0.38348097, -1.14974569]

#t = 2 #TREINAMENTO 2
#a.theta = -1.72913097
#a.W = [1.30674231,	1.61128472,	-0.38216074,	-1.15508637]



#t = 3 #TREINAMENTO 3
#a.theta = -1.7309785
#a.W = [1.3044548,	1.60146574,	-0.38305422,	-1.14856879]

#t = 4 #TREINAMENTO 4
#a.theta = -1.73498822
#a.W = [1.30496703,	1.60406134,	-0.38265301,	-1.15029989]

#t = 5 #TREINAMENTO 5
#a.theta = -1.72837578
#a.W = [1.3043394,	1.60193683,	-0.38243105,	-1.14886613]

print('')

#print(f'T{t} Amostra 1, saida {a.evaluate([0.9694,0.6909,0.4334,3.4965])}')
#print(f'T{t} Amostra 2, saida {a.evaluate([0.5427,1.3832,0.6390,4.0352])}')
#print(f'T{t} Amostra 3, saida {a.evaluate([0.6081,-0.9196,0.5925,0.1016])}')
#print(f'T{t} Amostra 4, saida {a.evaluate([-0.1618,0.4694,0.2030,3.0117])}')
#print(f'T{t} Amostra 5, saida {a.evaluate([0.1870,-0.2578,0.6124,1.7749])}')
#print(f'T{t} Amostra 6, saida {a.evaluate([0.4891,-0.5276,0.4378,0.6439])}')
#print(f'T{t} Amostra 7, saida {a.evaluate([0.3777,2.0149,0.7423,3.3932])}')
#print(f'T{t} Amostra 8, saida {a.evaluate([1.1498,-0.4067,0.2469,1.5866])}')
#print(f'T{t} Amostra 9, saida {a.evaluate([0.9325,1.0950,1.0359,3.3591])}')
#print(f'T{t} Amostra 10, saida {a.evaluate([0.5060,1.3317,0.9222,3.7174])}')
#print(f'T{t} Amostra 11, saida {a.evaluate([0.0497,-2.0656,0.6124,-0.6585])}')
#print(f'T{t} Amostra 12, saida {a.evaluate([0.4004,3.5369,0.9766,5.3532])}')
#print(f'T{t} Amostra 13, saida {a.evaluate([-0.1874,1.3343,0.5374,3.2189])}')
#print(f'T{t} Amostra 14, saida {a.evaluate([0.5060,1.3317,0.9222,3.7174])}')
#print(f'T{t} Amostra 15, saida {a.evaluate([1.6375,-0.7911,0.7537,0.5515])}')