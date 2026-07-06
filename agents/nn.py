try:
    import cupy as np
    _GPU_ATIVA = True
except ImportError:
    import numpy as np
    _GPU_ATIVA = False

class RedeNeuralSupervisionada:
    def __init__(self, tamanho_entrada=86, tamanho_oculto1=256, tamanho_oculto2=128, tamanho_saida=58, taxa_aprendizado=0.01):
        """
        Arquitetura profunda otimizada: 86 -> 256 -> 128 -> 58
        """
        self.lr = taxa_aprendizado
        
        # Camada Oculta 1 (He Initialization)
        self.W1 = np.random.randn(tamanho_oculto1, tamanho_entrada) * np.sqrt(2. / tamanho_entrada)
        self.b1 = np.zeros((tamanho_oculto1, 1))
        
        # Camada Oculta 2 (He Initialization)
        self.W2 = np.random.randn(tamanho_oculto2, tamanho_oculto1) * np.sqrt(2. / tamanho_oculto1)
        self.b2 = np.zeros((tamanho_oculto2, 1))
        
        # Camada de Saída (Xavier Initialization para Softmax)
        self.W3 = np.random.randn(tamanho_saida, tamanho_oculto2) * np.sqrt(1. / tamanho_oculto2)
        self.b3 = np.zeros((tamanho_saida, 1))
        
        self.cache = {}

    def relu(self, Z):
        return np.maximum(0, Z)

    def derivada_relu(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.relu(Z1)
        
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.relu(Z2)
        
        Z3 = np.dot(self.W3, A2) + self.b3
        A3 = self.softmax(Z3)
        
        self.cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
        return A3

    def backward(self, Y_target):
        m = Y_target.shape[1]
        
        A3, A2, A1, X = self.cache["A3"], self.cache["A2"], self.cache["A1"], self.cache["X"]

        # Derivada Combinada (Softmax + Cross-Entropy)
        dZ3 = A3 - Y_target 
        dW3 = (1. / m) * np.dot(dZ3, A2.T)
        db3 = (1. / m) * np.sum(dZ3, axis=1, keepdims=True)

        # Propagação para a Camada Oculta 2
        dA2 = np.dot(self.W3.T, dZ3)
        dZ2 = dA2 * self.derivada_relu(self.cache["Z2"])
        dW2 = (1. / m) * np.dot(dZ2, A1.T)
        db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

        # Propagação para a Camada Oculta 1
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.derivada_relu(self.cache["Z1"])
        dW1 = (1. / m) * np.dot(dZ1, X.T)
        db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Atualização (SGD)
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return - (1. / m) * np.sum(Y_target * np.log(A3 + 1e-8))

    def treinar(self, X_train, Y_train, X_val=None, Y_val=None, epochs=1500, batch_size=128, ao_validar=None):
        """
        Executa o loop de treinamento sobre o dataset particionado em mini-batches,
        avaliando contra um conjunto de validação para detectar overfitting.

        ao_validar (opcional): callback(epoch, custo_val, rede) chamado a cada
        avaliação de validação. Usado por training_loop.py para guardar em
        memória os pesos da época com o menor Val Custo da execução.
        """
        historico_custo = []
        m = X_train.shape[1]
        
        for epoch in range(epochs):
            # Embaralha os dados a cada época
            permutacao = np.random.permutation(m)
            X_shuffled = X_train[:, permutacao]
            Y_shuffled = Y_train[:, permutacao]
            
            custo_epoch = 0
            num_batches = 0
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                Y_batch = Y_shuffled[:, i:i+batch_size]
                
                self.forward(X_batch)
                custo = self.backward(Y_batch)
                custo_epoch += custo
                num_batches += 1
                
            custo_medio = custo_epoch / num_batches
            historico_custo.append(custo_medio)
            
            if epoch % 10 == 0:
                val_str = ""
                # Avaliação do conjunto de validação
                if X_val is not None and Y_val is not None:
                    self.forward(X_val)
                    A3_val = self.cache["A3"]
                    m_val = X_val.shape[1]
                    custo_val = - (1. / m_val) * np.sum(Y_val * np.log(A3_val + 1e-8))
                    val_str = f" | Val Custo: {custo_val:.4f}"

                    if ao_validar is not None:
                        ao_validar(epoch, custo_val, self)

                print(f"Epoch {epoch} | Treino Custo: {custo_medio:.4f}{val_str}")
                
        return historico_custo
    
# criar um modelo para jogar o Agente da rede com o Agente heurístico
# para ver se está melhorando 

#1-  PARTE GRAFICA
#2- IA x Heurístico e Heurístico x IA