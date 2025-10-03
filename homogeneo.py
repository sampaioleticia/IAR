import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
from typing import List, Tuple, Optional
from numba import jit

class Item:
    """Representa um item que pode ser manipulado pelas formigas."""
    
    def __init__(self, item_id: int, item_type: str):
        self.item_id = item_id
        self.item_type = item_type
    
    def __repr__(self):
        return f"Item({self.item_id}, {self.item_type})"


class Formiga:
    """Representa um agente formiga que se move pelo grid e manipula itens."""
    
    def __init__(self, formiga_id: int, x: int, y: int):
        self.formiga_id = formiga_id
        self.x = x
        self.y = y
        self.carregando = False
        self.item_carregado: Optional[Item] = None
    
    def mover(self, novo_x: int, novo_y: int):
        """Move a formiga para uma nova posição."""
        self.x = novo_x
        self.y = novo_y
    
    def pegar_item(self, item: Item):
        """Pega um item do grid."""
        self.carregando = True
        self.item_carregado = item
    
    def largar_item(self) -> Optional[Item]:
        """Larga o item carregado."""
        item = self.item_carregado
        self.carregando = False
        self.item_carregado = None
        return item


# Funções JIT compiladas para cálculos críticos
@jit(nopython=True, cache=True)
# Densidade Local
def calcular_densidade_numba(grid_itens, x, y, raio, altura, largura):
    """
    Calcula a densidade local: razão entre itens presentes e células totais na vizinhança.
    
    Args:
        grid_itens: Array numpy com IDs dos itens
        x, y: Coordenadas da posição central
        raio: Raio de visão
        altura, largura: Dimensões do grid
    
    Returns:
        float: Densidade normalizada entre 0 e 1
    """
    count_itens = 0 # ← Numerador: soma do indicador 𝟙[G[x',y'] ≥ 0]
    count_total = 0 # ← Denominador: |𝒩ᵣ(x,y)|
    
    # Percorre a vizinhança de Moore
    for dx in range(-raio, raio + 1): # |x' - x| ≤ r
        for dy in range(-raio, raio + 1): # |y' - y| ≤ r
            if dx == 0 and dy == 0: # Pula a célula central
                continue
            
            nx, ny = x + dx, y + dy
            if 0 <= nx < altura and 0 <= ny < largura:
                count_total += 1
                if grid_itens[nx, ny] >= 0:
                    count_itens += 1
    
    if count_total == 0:
        return 0.0
    
    return count_itens / count_total


@jit(nopython=True, cache=True)
def probabilidade_pegar_numba(f_xi, k1, alpha):
    """
    Calcula a probabilidade de pegar um item baseada na densidade local.
    Probabilidade maior em áreas com baixa densidade.
    """
    return (k1 / (k1 + f_xi)) ** (2 * alpha)


@jit(nopython=True, cache=True)
def probabilidade_largar_numba(f_xi, k2, alpha):
    """
    Calcula a probabilidade de largar um item baseada na densidade local.
    Probabilidade maior em áreas com alta densidade.
    """
    return (f_xi / (k2 + f_xi)) ** (2 * alpha)


class Grid:
    """Grid bidimensional para armazenar itens e formigas."""
    
    def __init__(self, largura: int, altura: int):
        self.largura = largura
        self.altura = altura
        # Array numpy: -1 = vazio, >= 0 = item_id
        self.itens_array = np.full((altura, largura), -1, dtype=np.int32)
        self.itens_obj = {}  # Mapeia item_id -> Item object
        self.formigas = [[[] for _ in range(largura)] for _ in range(altura)]
    
    def adicionar_item(self, item: Item, x: int, y: int) -> bool:
        """Adiciona um item ao grid na posição especificada."""
        if 0 <= x < self.altura and 0 <= y < self.largura and self.itens_array[x, y] == -1:
            self.itens_array[x, y] = item.item_id
            self.itens_obj[item.item_id] = item
            return True
        return False
    
    def remover_item(self, x: int, y: int) -> Optional[Item]:
        """Remove e retorna o item da posição especificada."""
        if 0 <= x < self.altura and 0 <= y < self.largura:
            item_id = self.itens_array[x, y]
            if item_id >= 0:
                self.itens_array[x, y] = -1
                return self.itens_obj[item_id]
        return None
    
    def adicionar_formiga(self, formiga: Formiga, x: int, y: int) -> bool:
        """Adiciona uma formiga ao grid na posição especificada."""
        if 0 <= x < self.altura and 0 <= y < self.largura:
            self.formigas[x][y].append(formiga)
            formiga.mover(x, y)
            return True
        return False
    
    def mover_formiga(self, formiga: Formiga, novo_x: int, novo_y: int) -> bool:
        """Move uma formiga para uma nova posição."""
        if 0 <= novo_x < self.altura and 0 <= novo_y < self.largura:
            self.formigas[formiga.x][formiga.y].remove(formiga)
            self.formigas[novo_x][novo_y].append(formiga)
            formiga.mover(novo_x, novo_y)
            return True
        return False
    
    def obter_posicoes_vizinhas(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Retorna lista de posições vizinhas válidas (4-vizinhança)."""
        posicoes = []
        direcoes = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dx, dy in direcoes:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.altura and 0 <= ny < self.largura:
                posicoes.append((nx, ny))
        
        return posicoes
    
    def copiar_estado(self):
        """Cria uma cópia do estado atual do grid."""
        copia = Grid(self.largura, self.altura)
        copia.itens_array = self.itens_array.copy()
        copia.itens_obj = self.itens_obj.copy()
        return copia


class AntClusteringFase1:
    """
    Implementação do algoritmo de clustering baseado em formigas.
    Fase 1: Itens homogêneos (todos do mesmo tipo).
    """
    
    def __init__(
        self, 
        largura_grid: int = 50, 
        altura_grid: int = 50, 
        num_formigas: int = 20, 
        k1: float = 0.1, 
        k2: float = 0.3,
        raio_visao: int = 1, 
        alpha: float = 1.0
    ):
        """
        Args:
            largura_grid: Largura do grid
            altura_grid: Altura do grid
            num_formigas: Número de agentes formigas
            k1: Constante para probabilidade de pegar
            k2: Constante para probabilidade de largar
            raio_visao: Raio de percepção local
            alpha: Parâmetro de sensibilidade
        """
        self.grid = Grid(largura_grid, altura_grid)
        self.formigas: List[Formiga] = []
        self.k1 = k1
        self.k2 = k2
        self.raio_visao = raio_visao
        self.alpha = alpha
        self.iteracao = 0
        self.estados_salvos = {}
        self.historico_desvio = []
        
        # Inicializar formigas em posições aleatórias
        for i in range(num_formigas):
            formiga = Formiga(i, 0, 0)
            self.formigas.append(formiga)
            x = random.randint(0, self.grid.altura - 1)
            y = random.randint(0, self.grid.largura - 1)
            self.grid.adicionar_formiga(formiga, x, y)
    
    def adicionar_itens_aleatorios(self, quantidade: int, tipo: str = "A"):
        """Adiciona itens em posições aleatórias do grid."""
        for item_id in range(quantidade):
            item = Item(item_id, tipo)
            tentativas = 1000
            
            while tentativas > 0:
                x = random.randint(0, self.grid.altura - 1)
                y = random.randint(0, self.grid.largura - 1)
                if self.grid.adicionar_item(item, x, y):
                    break
                tentativas -= 1
    
    def calcular_f_homogeneo(self, x: int, y: int) -> float:
        """Calcula a densidade local normalizada."""
        return calcular_densidade_numba(
            self.grid.itens_array, 
            x, y, 
            self.raio_visao, 
            self.grid.altura, 
            self.grid.largura
        )
    
    def executar_passo_formiga(self, formiga: Formiga):
        """
        Executa um passo completo de uma formiga:
        1. Movimento aleatório
        2. Decisão de pegar item (se livre)
        3. Decisão de largar item (se carregando)
        """
        # 1. MOVIMENTO: Move-se aleatoriamente para uma posição vizinha
        posicoes_vizinhas = self.grid.obter_posicoes_vizinhas(formiga.x, formiga.y)
        if posicoes_vizinhas:
            nova_pos = random.choice(posicoes_vizinhas)
            self.grid.mover_formiga(formiga, nova_pos[0], nova_pos[1])
        
        # 2. AÇÃO DE PEGAR: Se livre, há item e densidade baixa
        if not formiga.carregando:
            if self.grid.itens_array[formiga.x, formiga.y] >= 0:
                f_xi = self.calcular_f_homogeneo(formiga.x, formiga.y)
                prob_pegar = probabilidade_pegar_numba(f_xi, self.k1, self.alpha)
                
                if random.random() < prob_pegar:
                    item = self.grid.remover_item(formiga.x, formiga.y)
                    formiga.pegar_item(item)
        
        # 3. AÇÃO DE LARGAR: Se carregando, posição vazia e densidade alta
        else:
            if self.grid.itens_array[formiga.x, formiga.y] == -1:
                f_xi = self.calcular_f_homogeneo(formiga.x, formiga.y)
                prob_largar = probabilidade_largar_numba(f_xi, self.k2, self.alpha)
                
                if random.random() < prob_largar:
                    item = formiga.largar_item()
                    self.grid.adicionar_item(item, formiga.x, formiga.y)
    
    def executar_iteracao(self):
        """Executa uma iteração completa (todas as formigas se movem)."""
        formigas_embaralhadas = self.formigas.copy()
        random.shuffle(formigas_embaralhadas)
        
        for formiga in formigas_embaralhadas:
            self.executar_passo_formiga(formiga)
        
        self.iteracao += 1
    
    def diagnostico(self):
        """Retorna estatísticas do estado atual da simulação."""
        total_carregando = sum(1 for f in self.formigas if f.carregando)
        total_itens_grid = np.sum(self.grid.itens_array >= 0)
        
        # Calcular desvio padrão das posições dos itens
        posicoes = np.argwhere(self.grid.itens_array >= 0)
        if len(posicoes) > 1:
            std_x = np.std(posicoes[:, 0])
            std_y = np.std(posicoes[:, 1])
            desvio_padrao = (std_x + std_y) / 2  
        else:
            desvio_padrao = 0.0
        
        return {
            'formigas_carregando': total_carregando, 
            'itens_no_grid': int(total_itens_grid),
            'desvio_padrao': desvio_padrao
        }
    
    def simular(
        self, 
        num_iteracoes: int, 
        verbose: bool = False, 
        intervalo_log: int = 10000,
        iteracoes_para_plotar: List[int] = None, 
        pasta_saida: str = '1_graficos',
        intervalo_desvio: int = 10000
    ):
        """
        Executa a simulação completa.
        
        Args:
            num_iteracoes: Número total de iterações
            verbose: Se True, exibe logs de progresso
            intervalo_log: Intervalo entre logs
            iteracoes_para_plotar: Lista de iterações para salvar gráficos
            pasta_saida: Pasta onde salvar os gráficos
            intervalo_desvio: Intervalo para calcular desvio padrão
        
        Returns:
            float: Tempo total de execução em segundos
        """
        # Criar pasta de saída
        if not os.path.exists(pasta_saida):
            os.makedirs(pasta_saida)
        
        if iteracoes_para_plotar is None:
            iteracoes_para_plotar = []
        
        tempo_inicio = time.time()
        
        # Loop principal da simulação
        for i in range(num_iteracoes):
            self.executar_iteracao()
            
            # Salvar estado para plotagem
            if (i + 1) in iteracoes_para_plotar:
                self.estados_salvos[i + 1] = self.grid.copiar_estado()
            
            # Calcular e salvar desvio padrão periodicamente
            if (i + 1) % intervalo_desvio == 0:
                diag = self.diagnostico()
                self.historico_desvio.append((i + 1, diag['desvio_padrao']))
            
            # Log de progresso
            if verbose and (i + 1) % intervalo_log == 0:
                tempo_decorrido = time.time() - tempo_inicio
                diag = self.diagnostico()
                print(
                    f"  [{i + 1:>8}/{num_iteracoes}] "
                    f"Tempo: {tempo_decorrido:>6.1f}s | "
                    f"Grid: {diag['itens_no_grid']:>3} | "
                    f"Carregando: {diag['formigas_carregando']:>2} | "
                    f"Desvio: {diag['desvio_padrao']:>5.2f}"
                )
        
        tempo_total = time.time() - tempo_inicio
        
        # Gerar gráficos
        self._gerar_graficos(iteracoes_para_plotar, pasta_saida)
        
        return tempo_total
    
    def _gerar_graficos(self, iteracoes_para_plotar: List[int], pasta_saida: str):
        """Gera e salva todos os gráficos da simulação."""
        print("\nGerando gráficos...")
        
        # Gráficos individuais para cada iteração
        for iteracao in iteracoes_para_plotar:
            if iteracao in self.estados_salvos:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                self._plotar(ax, f"Iteração {iteracao:,}", self.estados_salvos[iteracao])
                plt.tight_layout()
                plt.savefig(f'{pasta_saida}/iteracao_{iteracao}.png', dpi=300)
                plt.close(fig)
                print(f"  Salvo: iteracao_{iteracao}.png")
        
        # Gráfico comparativo de evolução
        if iteracoes_para_plotar:
            todas = [0] + iteracoes_para_plotar
            fig, axes = plt.subplots(1, len(todas), figsize=(6 * len(todas), 8))
            
            if len(todas) == 1:
                axes = [axes]
            
            for idx, it in enumerate(todas):
                if it in self.estados_salvos:
                    self._plotar(axes[idx], f"Iter {it:,}", self.estados_salvos[it])
            
            plt.tight_layout()
            plt.savefig(f'{pasta_saida}/clustering_evolucao.png', dpi=300)
            plt.close(fig)
            print(f"  Salvo: clustering_evolucao.png")
        
        # Gráfico de evolução do desvio padrão
        if self.historico_desvio:
            self._plotar_evolucao_desvio(pasta_saida)
    
    def _plotar_evolucao_desvio(self, pasta_saida: str):
        """Plota a evolução do desvio padrão ao longo das iterações."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        iteracoes_hist = [x[0] for x in self.historico_desvio]
        desvios_hist = [x[1] for x in self.historico_desvio]
        
        ax.plot(
            iteracoes_hist, 
            desvios_hist, 
            linewidth=2, 
            color='#2C3E50', 
            alpha=0.8
        )
        ax.fill_between(
            iteracoes_hist, 
            desvios_hist, 
            alpha=0.3, 
            color='#3498DB'
        )
        
        ax.set_xlabel('Iterações', fontsize=12, fontweight='600', color='#2C3E50')
        ax.set_ylabel('Desvio Padrão', fontsize=12, fontweight='600', color='#2C3E50')
        ax.set_title(
            'Evolução do Desvio Padrão ao Longo das Iterações', 
            fontsize=14, 
            fontweight='700', 
            pad=15, 
            color='#1A252F'
        )
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#95A5A6')
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        plt.savefig(f'{pasta_saida}/evolucao_desvio.png', dpi=300)
        plt.close(fig)
        print(f"  Salvo: evolucao_desvio.png")
    
    def _plotar(self, ax, titulo: str, grid_estado=None):
        """Plota o estado do grid em um eixo matplotlib."""
        if grid_estado is None:
            grid_estado = self.grid
        
        ax.set_facecolor('#FAFAFA')
        
        # Plotar posições dos itens
        posicoes = np.argwhere(grid_estado.itens_array >= 0)
        for pos in posicoes:
            i, j = pos
            ax.plot(
                j, i, 'o', 
                color='#2C3E50', 
                markersize=8, 
                markeredgecolor='#34495E', 
                markeredgewidth=0.8, 
                alpha=0.85
            )
        
        # Configurar eixos
        ax.set_xlim(-1.5, grid_estado.largura + 0.5)
        ax.set_ylim(-1.5, grid_estado.altura + 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, color='#BDC3C7')
        ax.set_axisbelow(True)
        
        # Labels e título
        ax.set_xlabel('Posição X', fontsize=11, fontweight='600', color='#2C3E50')
        ax.set_ylabel('Posição Y', fontsize=11, fontweight='600', color='#2C3E50')
        ax.set_title(titulo, fontsize=13, fontweight='700', pad=15, color='#1A252F')
        
        # Estilizar bordas
        for spine in ax.spines.values():
            spine.set_edgecolor('#95A5A6')
            spine.set_linewidth(1.2)
        
        ax.tick_params(axis='both', which='major', labelsize=9, colors='#34495E')


def main():
    """Função principal para executar a simulação."""
    print("=" * 75)
    print(" " * 8 + "ANT CLUSTERING - ITENS HOMOGÊNEOS")
    print("=" * 75)

    # Parâmetros da simulação
    GRID_SIZE = 40        
    NUM_AGENTES = 100       
    NUM_ITENS = 500      
    K1 = 0.25            
    K2 = 0.06          
    ALPHA = 3.0        
    RAIO_VISAO = 1  
    NUM_ITERACOES = 15000000 
    
    # Exibir configuração
    print(f"\n  Grid:                {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Agentes:             {NUM_AGENTES}")
    print(f"  Itens:               {NUM_ITENS}")
    print(f"  k1:                  {K1}")
    print(f"  k2:                  {K2}")
    print(f"  alpha:               {ALPHA}")
    print(f"  Raio de visão:       {RAIO_VISAO}")
    print(f"  Total de iterações:  {NUM_ITERACOES:,}")
    print(f"  Ocupação inicial:    {100 * NUM_ITENS / (GRID_SIZE ** 2):.1f}%")
    print("=" * 75)
    
    # Inicializar clustering
    clustering = AntClusteringFase1(
        largura_grid=GRID_SIZE, 
        altura_grid=GRID_SIZE, 
        num_formigas=NUM_AGENTES,
        k1=K1, 
        k2=K2, 
        raio_visao=RAIO_VISAO, 
        alpha=ALPHA
    )
    clustering.adicionar_itens_aleatorios(quantidade=NUM_ITENS, tipo="A")
    
    # Preparar pasta de saída
    os.makedirs('1_graficos', exist_ok=True)
    clustering.estados_salvos[0] = clustering.grid.copiar_estado()
    
    # Executar simulação
    print("\nExecutando simulação...\n")
    tempo = clustering.simular(
        num_iteracoes=NUM_ITERACOES, 
        verbose=True, 
        intervalo_log=1000000,
        iteracoes_para_plotar=[
            0, 10000, 50000, 100000, 250000, 500000, 1000000, 
            2000000, 4000000, 6000000, 8000000, 10000000, 12000000, 15000000
        ],
        intervalo_desvio=10000
    )
    
    # Resultados finais
    print("\n" + "=" * 75)
    print("SIMULAÇÃO CONCLUÍDA!")
    print("=" * 75)
    
    diag = clustering.diagnostico()
    print(f"\nTempo total: {tempo:.1f}s ({tempo / 60:.1f}min)")
    print(f"Estado final: Grid={diag['itens_no_grid']}, Carregando={diag['formigas_carregando']}")
    print(f"\nParâmetros: k1={K1}, k2={K2}, alpha={ALPHA}, raio={RAIO_VISAO}")
    print("=" * 75)


if __name__ == "__main__":
    main()