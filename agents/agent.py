import random

# ==========================================
# 1. MIDDLEWARE / INTERFACES
# ==========================================
# Fonte única da verdade: as definições vivem em middleware/middleware.py e
# são reexportadas aqui para manter os imports existentes
# (`from agents.agent import Agente`) funcionando sem duplicar a lógica.
from middleware.middleware import Agente, GerenciadorPartida  # noqa: F401

# ==========================================
# 2. AGENTES HEURÍSTICOS
# ==========================================
class AgenteAleatorio(Agente):
    """Escolhe uma jogada válida de forma puramente aleatória."""
    def escolher_jogada(self, estado, jogadas_legais):
        # Se a única ação for passar (None) ou comprar, random.choice lidará com isso
        return random.choice(jogadas_legais)

class AgenteGuloso(Agente):
    """Escolhe a peça com a maior soma de pontos (pips) para esvaziar a mão pesada."""
    def escolher_jogada(self, estado, jogadas_legais):
        # Se não há jogadas de colocação de peça, retorna a ação obrigatória
        if jogadas_legais == [None] or jogadas_legais == [("COMPRAR", None)]:
            return jogadas_legais[0]
        
        melhor_jogada = None
        maior_pontuacao = -1
        
        for jogada in jogadas_legais:
            peca = jogada[0]
            pontuacao = peca[0] + peca[1]
            if pontuacao > maior_pontuacao:
                maior_pontuacao = pontuacao
                melhor_jogada = jogada
                
        return melhor_jogada

# ==========================================
# 3. EXECUÇÃO DA PARTIDA
# ==========================================
if __name__ == "__main__":
    from middleware.motor_domino import MotorDomino
    
    print("Iniciando MotorDomino...")
    motor = MotorDomino(num_jogadores=2)
    
    print("Instanciando Agentes (Guloso vs Aleatório)...")
    agente0 = AgenteGuloso()
    agente1 = AgenteAleatorio()
    
    print("Configurando Gerenciador de Partida (Middleware)...")
    gerenciador = GerenciadorPartida(motor, [agente0, agente1])
    
    print("Executando partida completa de forma automatizada...")
    gerenciador.jogar_partida_completa()
    
    # Extraindo o dicionário final (agora seguro e serializável)
    estado_final = motor.to_dict()
    
    print("\n--- Resultado da Partida ---")
    if estado_final["vencedor"] == -1:
        print("Resultado: Empate (Jogo Trancado resolutamente empatado)")
    else:
        print(f"Resultado: Jogador {estado_final['vencedor']} Venceu!")
        
    print(f"Total de turnos processados: {estado_final['turno']}")
    print(f"Peças restantes no monte: {len(estado_final['monte'])}")
    print(f"Tamanho final da mão do Jogador 0: {len(estado_final['maos'][0])}")
    print(f"Tamanho final da mão do Jogador 1: {len(estado_final['maos'][1])}")