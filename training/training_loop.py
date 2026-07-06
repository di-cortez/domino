import json
import os
import numpy as np
from agents.codificador import CodificadorDomino
from agents.nn import RedeNeuralSupervisionada

try:
    import cupy as cp
    USE_GPU = True
    print("CuPy disponível. Treinamento na GPU.")
except ImportError:
    import numpy as cp     # cp becomes a numpy alias
    USE_GPU = False
    print("CuPy não encontrado. Treinamento na CPU.")

EPOCHS = 1000
BATCH_SIZE = 128

def carregar_dataset(caminho_arquivo, codificador):
    """
    Lê o arquivo JSONL e converte as partidas em matrizes matemáticas X e Y.
    """
    X_list = []
    Y_list = []
    
    print(f"Lendo dataset de {caminho_arquivo}...")
    
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        for linha in f:
            if not linha.strip():
                continue
                
            registro = json.loads(linha)
            estado = registro["estado"]
            acao_alvo = registro["acao_alvo"]

            # Bug 1 Fix: Comentário corrigido para R^N, N = CodificadorDomino.TAMANHO_VETOR [cite: 818]
            vetor_x = codificador.encode_estado(estado)
            X_list.append(vetor_x)
            
            # Bug 2 Fix: Desserialização explícita e segura para o JSON None e listas [cite: 830]
            if acao_alvo is None:
                idx_acao = 57
            elif acao_alvo == ["COMPRAR", None] or acao_alvo == ("COMPRAR", None):
                idx_acao = 56
            else:
                if isinstance(acao_alvo[0], list):
                    acao_alvo = (tuple(acao_alvo[0]), acao_alvo[1])
                elif isinstance(acao_alvo, list):
                    acao_alvo = tuple(acao_alvo)
                    
                idx_acao = codificador.acao_para_indice[acao_alvo]
                
            vetor_y = np.zeros((58, 1))
            vetor_y[idx_acao, 0] = 1.0
            Y_list.append(vetor_y)
            
    X = np.hstack(X_list)
    Y = np.hstack(Y_list)
    
    print(f"Dataset carregado! Shape X: {X.shape}, Shape Y: {Y.shape}")
    return X, Y

def main():
    arquivo_dataset = "dataset/dataset_2.jsonl"
    arquivo_pesos = "models/pesos_domino_sl.npz" # Issue 5 Fix: Formato NumPy nativo [cite: 870]
    
    codificador = CodificadorDomino()
    
    # 1. Parse do Dataset
    X_full, Y_full = carregar_dataset(arquivo_dataset, codificador)
    
    # Issue 4 Fix: Divisão de Treino/Validação (85% / 15%) [cite: 860]
    m_total = X_full.shape[1]
    m_train = int(m_total * 0.85)
    
    indices = np.random.permutation(m_total)
    indices_train = indices[:m_train]
    indices_val = indices[m_train:]
    
    X_train = cp.array(X_full[:, indices_train])
    Y_train = cp.array(Y_full[:, indices_train])
    X_val = cp.array(X_full[:, indices_val])
    Y_val = cp.array(Y_full[:, indices_val])

    print(f"Divisão concluída: {X_train.shape[1]} treino | {X_val.shape[1]} validação")

    # Bug 1 Fix: Comentário corrigido para TAMANHO_VETOR->256->128->58 [cite: 818]
    rede = RedeNeuralSupervisionada(
        tamanho_entrada=CodificadorDomino.TAMANHO_VETOR,
        tamanho_oculto1=256, #256
        tamanho_oculto2=128, #128
        tamanho_saida=58,
        taxa_aprendizado=0.005 # Bug 3 Fix: Reduzido de 0.05 para estabilidade
    )

    # Guarda em memória, a cada validação, os pesos da época com o menor Val
    # Custo já visto na execução (checkpoint do melhor modelo, não do último).
    melhor_estado = {"custo_val": float("inf"), "pesos": None}

    def salvar_se_melhor(epoch, custo_val, rede_atual):
        if custo_val < melhor_estado["custo_val"]:
            melhor_estado["custo_val"] = custo_val
            melhor_estado["pesos"] = {
                "W1": rede_atual.W1.copy(), "b1": rede_atual.b1.copy(),
                "W2": rede_atual.W2.copy(), "b2": rede_atual.b2.copy(),
                "W3": rede_atual.W3.copy(), "b3": rede_atual.b3.copy(),
            }
            print(f"  -> Novo melhor Val Custo ({custo_val:.4f}) na epoch {epoch}. Pesos salvos em memória.")

    # 3. Treinamento
    print("\nIniciando treinamento da rede neural...")
    #epoch 1500
    historico = rede.treinar(
        X_train, Y_train, X_val=X_val, Y_val=Y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, ao_validar=salvar_se_melhor,
    )

    def convert_to_np(matriz):
        return cp.asnumpy(matriz) if USE_GPU else matriz

    # Issue 5 Fix: Salvamento nativo e seguro com np.savez [cite: 870]
    print(f"\nTreinamento concluído. Salvando modelo em {arquivo_pesos}...")

    # Garante que a pasta de destino (ex.: models/) exista antes de salvar.
    pasta_pesos = os.path.dirname(arquivo_pesos)
    if pasta_pesos:
        os.makedirs(pasta_pesos, exist_ok=True)

    # Salva os pesos da melhor época (menor Val Custo), não os da última,
    # caindo de volta para os pesos finais se nunca houve validação.
    pesos_para_salvar = melhor_estado["pesos"] or {
        "W1": rede.W1, "b1": rede.b1,
        "W2": rede.W2, "b2": rede.b2,
        "W3": rede.W3, "b3": rede.b3,
    }

    np.savez(
        arquivo_pesos,
        W1=convert_to_np(pesos_para_salvar["W1"]), b1=convert_to_np(pesos_para_salvar["b1"]),
        W2=convert_to_np(pesos_para_salvar["W2"]), b2=convert_to_np(pesos_para_salvar["b2"]),
        W3=convert_to_np(pesos_para_salvar["W3"]), b3=convert_to_np(pesos_para_salvar["b3"])
    )

    print(f"Modelo salvo com sucesso (melhor Val Custo: {melhor_estado['custo_val']:.4f}). Pronto para ser usado pelo Agente Neural!")

if __name__ == "__main__":
    main()