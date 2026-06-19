import json
import os
import time

# Certifique-se de importar suas classes dos arquivos corretos
from middleware.motor_domino import MotorDomino
from agents.heuristic_agent import AgenteEstrategico
from middleware.middleware import GerenciadorPartida

def gerar_dataset(num_partidas, arquivo_saida):
    """
    Simula milhares de partidas usando agentes especialistas e extrai
    cada turno como um par de (Estado, Ação) para o treinamento da Rede Neural.
    """
    print(f"Iniciando geração de {num_partidas} partidas...")
    inicio = time.time()

    total_turnos_salvos = 0

    # Garante que a pasta de destino (ex.: dataset/) exista antes de escrever.
    pasta_saida = os.path.dirname(arquivo_saida)
    if pasta_saida:
        os.makedirs(pasta_saida, exist_ok=True)

    # Abrimos o arquivo em modo 'w' para iniciar um dataset limpo
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        for i in range(num_partidas):
            motor = MotorDomino(num_jogadores=2)
            
            # Ambos os jogadores são especialistas. 
            # Isso garante que 100% dos turnos gerados sejam labels de alta qualidade.
            agentes = [AgenteEstrategico(), AgenteEstrategico()] 
            gerenciador = GerenciadorPartida(motor, agentes)
            
            # Joga a partida até o fim silenciosamente
            info, historico_partida = gerenciador.jogar_partida_completa()
            
            # Grava cada ponto de decisão individualmente no formato JSON Lines
            for turno in historico_partida:
                f.write(json.dumps(turno) + '\n')
                total_turnos_salvos += 1
                
            if (i + 1) % 5000 == 0:
                print(f"{i + 1} partidas simuladas... ({total_turnos_salvos} exemplos extraídos)")
                
    tempo_total = time.time() - inicio
    print("-" * 40)
    print("GERAÇÃO CONCLUÍDA!")
    print(f"Total de pares (Estado, Ação): {total_turnos_salvos}")
    print(f"Arquivo gerado: {arquivo_saida}")
    print(f"Tempo de execução: {tempo_total:.2f} segundos")

if __name__ == "__main__":
    # 10.000 partidas costumam gerar entre 150.000 e 200.000 turnos de dominó,
    # o que é um tamanho excelente para o primeiro treinamento da rede profunda.
    gerar_dataset(num_partidas=10000, arquivo_saida="dataset/dataset_2.jsonl")