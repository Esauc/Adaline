import numpy as np

class Adaline:

  def __init__(self, input_values, output_values, learning_rate, precision, activation_function):
    self.input_values = input_values
    self.output_values = output_values
    self.learning_rate = learning_rate #Especifica a taxa de aprendizagem
    self.precision = precision #Especifica a precisão 
    self.activation_function = activation_function
    self.W = np.random.rand(input_values.shape[1]) #Pesos W gerados de forma aleatória
    self.theta = np.random.rand(1)[0] #gerado de forma aleatória

    
  def train(self):
    initialW = [self.W[0], self.W[1], self.W[2], self.W[3]]
    initialTheta = self.theta

    epochs = 0 #Inicia o contador de número de épocas
    eqm = 0 #Iniciar variável Eqm com valor zero
    error = True
    p = len(self.output_values) #Obter a quantidade de padrões de treinamento {p}
    print(f'Initial W: {self.W}')
    print(f'Initial Theta: {self.theta}')
    print(f'[EPOCH {epochs}]')

    while error:
      error = False

      eqmAnterior = eqm #Recebe o valor anterior da variável Eqm

      print(f'Epoch: {epochs}')

      #Associoar a saída desejada para cada amostra
      for x, d in zip(self.input_values, self.output_values):

        u = np.dot(x, self.W) - self.theta
        y = self.activation_function.g(u)

        #Atualiza o vetor dos pesos
        self.W = self.W + self.learning_rate * (d - u) * x
        self.theta = self.theta + self.learning_rate * (d - y) * -1

        eqm = eqm + (d -u) ** 2 #Erro quadrático

        #print(f'Input: {x}, Output: {y}, Expected: {d}')

      eqm = eqm / p #Erro quadrático médio
      epochs += 1
      
      #Recalcula Eqm com os novos pesos W
      e = eqm -eqmAnterior
      
      print(f'Eqm = {eqm}')
      print(f'Error = {e}')

      #Para verificar se é linearmente separável é só trocar por if(e != 0.000000):
      #Se executar eternamente -> Não é linearmente separável
      #Se achar um resultado significa que erro é zero e é linearmente separável
      if (abs(e) > self.precision):
        error = True

      #print('')

    print('TRAINING COMPLETE!')
    print(f'Initial W: {initialW}')
    print(f'Initial Theta: {initialTheta}')
    print(f'Final W: {self.W}')
    print(f'Final Theta: {self.theta}')


  def evaluate(self, input_value):
    u = np.dot(input_value, self.W) -self.theta
    y = self.activation_function.g(u)

    return y
    
