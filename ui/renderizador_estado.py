"""
Estado auxiliar do renderizador da mesa.

O motor entrega a cadeia em ordem lógica. Para a animação ficar estável, este
objeto guarda qual peça foi escolhida como pivô e tenta reencontrá-la nos
snapshots seguintes.
"""

class RenderizadorEstado:
    def __init__(self):
        self._chave_pivo = None

    def _chave_peca(self, info):
        if "id" in info:
            return ("id", info["id"])

        if "id_peca" in info:
            return ("id_peca", info["id_peca"])

        a, b = info["peca"]
        return ("peca", min(a, b), max(a, b))

    def obter_indice_pivo(self, cadeia_visual):
        if not cadeia_visual:
            return None

        if self._chave_pivo is not None:
            for i, info in enumerate(cadeia_visual):
                if self._chave_peca(info) == self._chave_pivo:
                    return i

        self._chave_pivo = self._chave_peca(cadeia_visual[0])
        return 0
