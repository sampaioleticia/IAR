# Fase 1 
#   1. Carrega os dados
#   2. Normaliza e distribui no grid aleatóriamente
#   3. Cria e distribui formigas aleatóriamente 
# Fase 2
#   1. Cada iteração todas as formigas agem uma vez
#   2. Calcula o movimento inteligete 
#   3. Formiga decide se pega ou não o item
#      3.1. Calcula f_xi = quão bem o item esta posicionado 
#      - Olha o vizinho ao redor, calcula a distância para cada vizinho, converte a distanca em similaridade (f_xi = média das similaridades)
#      3.2. Calcula a probabilidaed de pegar (prob = (k1 / (k1 + f_xi))^expoente)
#      3.3. Sorteia: random < prob? Se sim pega, senão, não pega
#   4. Formiga decide se larga ou não um item
#      4.1. Calcula f_xi
#      - Simula largar o item, olha seus vizinhos, calcula a similaride média
#      4.2. Calcula a probabilidade de largar (prob = (f_xi / (k2 + f_xi))^expoente)
#      4.3. Sorteia: random < prob? Se sim larga, senão, continua carregando

# Similaridade Local
# 1. Identifica os vizinhos
# 2. Calcula a distância euclidiana para cada vizinho
# 3. Converte distância em similaridade
# 4. Calcula a média

import numpy as np
import matplotlib.pyplot as plt
import random
import time
from typing import List, Tuple, Optional
from numba import jit

class Item:
    def __init__(self, item_id: int, dados: np.ndarray, cor: int):
        self.item_id = item_id
        self.dados = dados  
        self.cor = cor  # Cor/grupo original (apenas para visualização)
    
    def __repr__(self):
        return f"Item({self.item_id}, dados={self.dados}, cor={self.cor})"

class Formiga:
    def __init__(self, formiga_id: int, x: int, y: int):
        self.formiga_id = formiga_id
        self.x = x
        self.y = y
        self.carregando = False 
        self.item_carregado: Optional[Item] = None
    
    def mover(self, novo_x: int, novo_y: int):
        self.x = novo_x
        self.y = novo_y
    
    def pegar_item(self, item: Item):
        self.carregando = True
        self.item_carregado = item
    
    def largar_item(self) -> Optional[Item]:
        item = self.item_carregado
        self.carregando = False
        self.item_carregado = None
        return item

@jit(nopython=True, cache=True)
def distancia_euclidiana(vetor1, vetor2):
    """Calcula distância euclidiana entre dois vetores"""
    return np.sqrt(np.sum((vetor1 - vetor2) ** 2))

@jit(nopython=True, cache=True)
def calcular_similaridade(dist, alpha):
    """Calcula similaridade baseada na distância"""
    if dist < alpha:
        return 1.0 - (dist / alpha) # Quanto menor a distância, maior a similaridade 
    return 0.0

@jit(nopython=True, cache=True)
def probabilidade_pegar_numba(f_xi, k1, alpha_prob):
    """Calcula probabilidade de pegar"""
    return (k1 / (k1 + f_xi)) ** (2 * alpha_prob) # Se f_xi é alto a probabilidade é baixa

@jit(nopython=True, cache=True)
def probabilidade_largar_numba(f_xi, k2, alpha_prob):
    """Calcula probabilidade de largar"""
    return (f_xi / (k2 + f_xi)) ** (2 * alpha_prob) # Se f_xi é alto a probabilidade é alta

class Grid:
    def __init__(self, largura: int, altura: int):
        self.largura = largura
        self.altura = altura
        self.itens_array = np.full((altura, largura), -1, dtype=np.int32)
        self.itens_obj = {}  # Mapeia item_id -> Item object
        self.formigas = [[[] for _ in range(largura)] for _ in range(altura)]
    
    def adicionar_item(self, item: Item, x: int, y: int) -> bool:
        # Verifica se a posição é valida e está vazia
        if 0 <= x < self.altura and 0 <= y < self.largura and self.itens_array[x, y] == -1:
            self.itens_array[x, y] = item.item_id
            self.itens_obj[item.item_id] = item
            return True
        return False
    
    def remover_item(self, x: int, y: int) -> Optional[Item]:
        if 0 <= x < self.altura and 0 <= y < self.largura:
            item_id = self.itens_array[x, y]
            if item_id >= 0:
                self.itens_array[x, y] = -1
                return self.itens_obj[item_id]
        return None
    
    def adicionar_formiga(self, formiga: Formiga, x: int, y: int) -> bool:
        if 0 <= x < self.altura and 0 <= y < self.largura:
            self.formigas[x][y].append(formiga)
            formiga.mover(x, y)
            return True
        return False
    
    def mover_formiga(self, formiga: Formiga, novo_x: int, novo_y: int) -> bool:
        if 0 <= novo_x < self.altura and 0 <= novo_y < self.largura:
            self.formigas[formiga.x][formiga.y].remove(formiga)
            self.formigas[novo_x][novo_y].append(formiga)
            formiga.mover(novo_x, novo_y)
            return True
        return False
    
    def obter_posicoes_vizinhas(self, x: int, y: int) -> List[Tuple[int, int]]:
        # Retorna lista com as 4 posições vizinhas (não inclui diagonais)
        posicoes = []
        direcoes = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in direcoes:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.altura and 0 <= ny < self.largura:
                posicoes.append((nx, ny))
        return posicoes
    
    def copiar_estado(self):
        copia = Grid(self.largura, self.altura)
        copia.itens_array = self.itens_array.copy()
        copia.itens_obj = self.itens_obj.copy()
        return copia

class AntClusteringHeterogêneo:
    def __init__(self, largura_grid: int = 50, altura_grid: int = 50, 
                 num_formigas: int = 15, k1: float = 0.5, k2: float = 0.025,
                 raio_visao: int = 1, alpha: float = 0.35, temperatura_movimento: float = 2.0):
        self.grid = Grid(largura_grid, altura_grid)
        self.formigas: List[Formiga] = []
        self.k1 = k1
        self.k2 = k2
        self.raio_visao = raio_visao
        self.alpha = alpha
        self.temperatura_movimento = temperatura_movimento  # Controla o quão "guloso" é o movimento
        self.iteracao = 0
        self.estados_salvos = {}
        self.historico_metricas = []
        
        for i in range(num_formigas):
            formiga = Formiga(i, 0, 0)
            self.formigas.append(formiga)
            x = random.randint(0, self.grid.altura - 1)
            y = random.randint(0, self.grid.largura - 1)
            self.grid.adicionar_formiga(formiga, x, y)
    
    def carregar_dados(self, arquivo: str, normalizar: bool = True):
        """Carrega dados do arquivo e cria itens"""
        dados = []
        cores = []
        
        with open(arquivo, 'r') as f:
            for linha in f:
                linha = linha.strip()
                if not linha or linha.startswith('#'):
                    continue
                partes = linha.split('\t')
                if len(partes) >= 3:
                    x = float(partes[0].replace(',', '.'))
                    y = float(partes[1].replace(',', '.'))
                    cor = int(partes[2])
                    dados.append([x, y])
                    cores.append(cor)
        
        dados = np.array(dados)
        
        # Normalização
        if normalizar:
            dados_min = dados.min(axis=0)
            dados_max = dados.max(axis=0)
            dados = (dados - dados_min) / (dados_max - dados_min)
        
        # Criar itens
        for idx, (vetor_dados, cor) in enumerate(zip(dados, cores)):
            item = Item(idx, vetor_dados, cor)
            tentativas = 1000
            while tentativas > 0:
                x = random.randint(0, self.grid.altura - 1)
                y = random.randint(0, self.grid.largura - 1)
                if self.grid.adicionar_item(item, x, y):
                    break
                tentativas -= 1
        
        print(f"Carregados {len(dados)} itens de {len(set(cores))} grupos diferentes")
    
    def calcular_f_heterogeneo(self, item: Item, x: int, y: int) -> float:
        """Calcula função de similaridade f(xi) baseada em distância euclidiana"""
        # 1. Olha para todos os vizinhos ao redor (dentro do raio)
        # 2. Para todos, calcula a distância euclidiana 
        # 3. Converte a distância em similaridade
        # 4. Retorna a média das similaridades
        soma_similaridade = 0.0
        count = 0
        
        for dx in range(-self.raio_visao, self.raio_visao + 1):
            for dy in range(-self.raio_visao, self.raio_visao + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid.altura and 0 <= ny < self.grid.largura:
                    item_id_vizinho = self.grid.itens_array[nx, ny]
                    if item_id_vizinho >= 0:
                        item_vizinho = self.grid.itens_obj[item_id_vizinho]
                        dist = distancia_euclidiana(item.dados, item_vizinho.dados)
                        similaridade = calcular_similaridade(dist, self.alpha)
                        soma_similaridade += similaridade
                        count += 1
        
        if count == 0:
            return 0.0
        
        return soma_similaridade / count
    
    def movimento_inteligente(self, formiga: Formiga, posicoes_vizinhas: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Implementa movimento direcionado baseado no estado da formiga:
        - Se carregando: prefere regiões de alta similaridade
        - Se vazia: prefere regiões de baixa similaridade (para encontrar itens deslocados)
        """
        if not posicoes_vizinhas:
            return (formiga.x, formiga.y)
        
        # Se a formiga está carregando um item
        if formiga.carregando:
            scores = []
            for pos in posicoes_vizinhas:
                # Calcula a similaridade do item carregado naquela posição
                f_vizinho = self.calcular_f_heterogeneo(formiga.item_carregado, pos[0], pos[1])
                scores.append(f_vizinho)
            
            # Converte scores em probabilidades usando softmax
            scores_array = np.array(scores)
            if np.sum(scores_array) > 0:
                # Softmax com temperatura
                exp_scores = np.exp(scores_array * self.temperatura_movimento)
                probabilidades = exp_scores / np.sum(exp_scores)
                indice_escolhido = np.random.choice(len(posicoes_vizinhas), p=probabilidades)
                return posicoes_vizinhas[indice_escolhido]
        
        # Se a formiga está vazia (prefere áreas de baixa similaridade)
        else:
            scores = []
            for pos in posicoes_vizinhas:
                # Verifica se há um item naquela posição
                if self.grid.itens_array[pos[0], pos[1]] >= 0:
                    item_id = self.grid.itens_array[pos[0], pos[1]]
                    item = self.grid.itens_obj[item_id]
                    # Calcula similaridade do item (baixa similaridade = alta probabilidade)
                    f_vizinho = self.calcular_f_heterogeneo(item, pos[0], pos[1])
                    # Inverte o score (queremos baixa similaridade)
                    scores.append(1.0 - f_vizinho)
                else:
                    # Posição vazia tem score neutro
                    scores.append(0.5)
            
            # Converte scores em probabilidades
            scores_array = np.array(scores)
            if np.sum(scores_array) > 0:
                exp_scores = np.exp(scores_array * self.temperatura_movimento)
                probabilidades = exp_scores / np.sum(exp_scores)
                indice_escolhido = np.random.choice(len(posicoes_vizinhas), p=probabilidades)
                return posicoes_vizinhas[indice_escolhido]
        
        # Fallback: movimento aleatório
        return random.choice(posicoes_vizinhas)
    
    def executar_passo_formiga(self, formiga: Formiga):
        # 1. MOVIMENTO INTELIGENTE
        posicoes_vizinhas = self.grid.obter_posicoes_vizinhas(formiga.x, formiga.y)
        if posicoes_vizinhas:
            nova_pos = self.movimento_inteligente(formiga, posicoes_vizinhas)
            self.grid.mover_formiga(formiga, nova_pos[0], nova_pos[1])
        
        # 2. AÇÃO DE PEGAR
        if not formiga.carregando:
            if self.grid.itens_array[formiga.x, formiga.y] >= 0:
                item = self.grid.itens_obj[self.grid.itens_array[formiga.x, formiga.y]]
                f_xi = self.calcular_f_heterogeneo(item, formiga.x, formiga.y)
                prob_pegar = probabilidade_pegar_numba(f_xi, self.k1, self.alpha)
                if random.random() < prob_pegar:
                    item_removido = self.grid.remover_item(formiga.x, formiga.y)
                    formiga.pegar_item(item_removido)
        
        # 3. AÇÃO DE LARGAR
        else:
            if self.grid.itens_array[formiga.x, formiga.y] == -1:
                f_xi = self.calcular_f_heterogeneo(formiga.item_carregado, formiga.x, formiga.y)
                prob_largar = probabilidade_largar_numba(f_xi, self.k2, self.alpha)
                if random.random() < prob_largar:
                    item = formiga.largar_item()
                    self.grid.adicionar_item(item, formiga.x, formiga.y)
    
    def executar_iteracao(self):
        formigas_embaralhadas = self.formigas.copy()
        random.shuffle(formigas_embaralhadas)
        for formiga in formigas_embaralhadas:
            self.executar_passo_formiga(formiga)
        self.iteracao += 1
    
    def calcular_metricas(self):
        """Calcula métricas de qualidade do clustering"""
        total_carregando = sum(1 for f in self.formigas if f.carregando)
        total_itens_grid = np.sum(self.grid.itens_array >= 0)
        
        posicoes = np.argwhere(self.grid.itens_array >= 0)
        if len(posicoes) > 0:
            soma_pureza = 0.0
            count_pureza = 0
            
            for pos in posicoes:
                x, y = pos
                item_id = self.grid.itens_array[x, y]
                item = self.grid.itens_obj[item_id]
                f_xi = self.calcular_f_heterogeneo(item, x, y)
                soma_pureza += f_xi
                count_pureza += 1
            
            pureza_media = soma_pureza / count_pureza if count_pureza > 0 else 0.0
        else:
            pureza_media = 0.0
        
        return {
            'formigas_carregando': total_carregando,
            'itens_no_grid': int(total_itens_grid),
            'pureza_media': pureza_media
        }
    
    def simular(self, num_iteracoes: int, verbose: bool = False, intervalo_log: int = 1000000,
                iteracoes_para_plotar: List[int] = None, pasta_saida: str = '2_graficos',
                intervalo_metricas: int = 1000000):
        import os
        if not os.path.exists(pasta_saida):
            os.makedirs(pasta_saida)
        if iteracoes_para_plotar is None:
            iteracoes_para_plotar = []
        
        tempo_inicio = time.time()
        for i in range(num_iteracoes):
            self.executar_iteracao()
            
            if (i + 1) in iteracoes_para_plotar:
                self.estados_salvos[i + 1] = self.grid.copiar_estado()
            
            if (i + 1) % intervalo_metricas == 0:
                metricas = self.calcular_metricas()
                self.historico_metricas.append((i + 1, metricas['pureza_media']))
            
            if verbose and (i + 1) % intervalo_log == 0:
                tempo_decorrido = time.time() - tempo_inicio
                metricas = self.calcular_metricas()
                print(f"  [{i + 1:>8}/{num_iteracoes}] Tempo: {tempo_decorrido:>6.1f}s | Grid: {metricas['itens_no_grid']:>3} | Carregando: {metricas['formigas_carregando']:>2} | Pureza: {metricas['pureza_media']:>5.3f}")
        
        tempo_total = time.time() - tempo_inicio
        
        print("\nGerando gráficos...")
        for iteracao in iteracoes_para_plotar:
            if iteracao in self.estados_salvos:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                self._plotar(ax, f"Iteração {iteracao:,}", self.estados_salvos[iteracao])
                plt.tight_layout()
                plt.savefig(f'{pasta_saida}/iteracao_{iteracao}.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  Salvo: iteracao_{iteracao}.png")
        
        if iteracoes_para_plotar:
            num_plots = min(len(iteracoes_para_plotar), 6)
            indices_selecionados = np.linspace(0, len(iteracoes_para_plotar)-1, num_plots, dtype=int)
            plots_selecionados = [iteracoes_para_plotar[i] for i in indices_selecionados]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            for idx, it in enumerate(plots_selecionados):
                if it in self.estados_salvos:
                    self._plotar(axes[idx], f"Iter {it:,}", self.estados_salvos[it])
            plt.tight_layout()
            plt.savefig(f'{pasta_saida}/clustering_evolucao.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Salvo: clustering_evolucao.png")
        
        if self.historico_metricas:
            fig, ax = plt.subplots(figsize=(12, 6))
            iteracoes_hist = [x[0] for x in self.historico_metricas]
            pureza_hist = [x[1] for x in self.historico_metricas]
            
            ax.plot(iteracoes_hist, pureza_hist, linewidth=2.5, color='#667eea', alpha=0.9)
            ax.fill_between(iteracoes_hist, pureza_hist, alpha=0.2, color='#764ba2')
            ax.set_xlabel('Iterações', fontsize=13, fontweight='600', color='#2d3748')
            ax.set_ylabel('Pureza Média dos Clusters', fontsize=13, fontweight='600', color='#2d3748')
            ax.set_title('Evolução da Pureza Média ao Longo das Iterações', 
                        fontsize=15, fontweight='700', pad=20, color='#1a202c')
            ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#cbd5e0')
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#a0aec0')
            ax.spines['bottom'].set_color('#a0aec0')
            ax.tick_params(colors='#4a5568', labelsize=10)
            plt.tight_layout()
            plt.savefig(f'{pasta_saida}/evolucao_pureza.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Salvo: evolucao_pureza.png")
        
        return tempo_total
    
    def _plotar(self, ax, titulo: str, grid_estado=None):
        if grid_estado is None:
            grid_estado = self.grid
        
        ax.set_facecolor('#1a1a2e')
        cores_modernas = ['#FF1744', '#00E676', '#FFEA00', '#E91E63', '#00E5FF', '#FF6E40', '#7C4DFF', '#4CAF50', '#FFC107', '#2196F3', '#FF5722', '#9C27B0', '#F06292', '#CDDC39', '#00BCD4', '#8BC34A', '#673AB7', '#FF9800', '#EC407A', '#03A9F4']
        
        posicoes = np.argwhere(grid_estado.itens_array >= 0)
        for pos in posicoes:
            i, j = pos
            item_id = grid_estado.itens_array[i, j]
            item = grid_estado.itens_obj[item_id]
            cor = cores_modernas[item.cor % len(cores_modernas)]
            # Removida a borda preta (markeredgewidth=0)
            ax.plot(j, i, 'o', color=cor, markersize=9, 
                   markeredgewidth=0, alpha=0.85)
        
        ax.set_xlim(-1.5, grid_estado.largura + 0.5)
        ax.set_ylim(-1.5, grid_estado.altura + 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5, color='#cbd5e0')
        ax.set_axisbelow(True)
        ax.set_xlabel('Posição X', fontsize=12, fontweight='600', color='#2d3748')
        ax.set_ylabel('Posição Y', fontsize=12, fontweight='600', color='#2d3748')
        ax.set_title(titulo, fontsize=14, fontweight='700', pad=15, color='#1a202c')
        
        # Bordas mais suaves
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#a0aec0')
        ax.spines['bottom'].set_color('#a0aec0')
        ax.tick_params(axis='both', which='major', labelsize=10, colors='#4a5568')

if __name__ == "__main__":
    ARQUIVO = "base_sintetica_4_grupos.txt"  # ou "base_sintetica_15_grupos.txt"
    GRID_SIZE = 64
    NUM_AGENTES = 100
    K1 = 0.3
    K2 = 0.6
    ALPHA = 11.8029
    RAIO_VISAO = 1
    NUM_ITERACOES = 2000000
    NORMALIZAR = False
    TEMPERATURA_MOVIMENTO = 2.0  
    
    print("=" * 75)
    print(" " * 8 + "ANT CLUSTERING - ITENS HETEROGÊNEOS")
    print("=" * 75)
    
    print(f"\n  Arquivo:             {ARQUIVO}")
    print(f"  Grid:                {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Agentes:             {NUM_AGENTES}")
    print(f"  k1:                  {K1}")
    print(f"  k2:                  {K2}")
    print(f"  alpha:               {ALPHA}")
    print(f"  Raio de visão:       {RAIO_VISAO}")
    print(f"  Temperatura mov.:    {TEMPERATURA_MOVIMENTO}")
    print(f"  Total de iterações:  {NUM_ITERACOES:,}")
    print(f"  Normalizar dados:    {NORMALIZAR}")
    print("=" * 75)
    
    clustering = AntClusteringHeterogêneo(
        largura_grid=GRID_SIZE,
        altura_grid=GRID_SIZE,
        num_formigas=NUM_AGENTES,
        k1=K1,
        k2=K2,
        raio_visao=RAIO_VISAO,
        alpha=ALPHA,
        temperatura_movimento=TEMPERATURA_MOVIMENTO
    )
    
    clustering.carregar_dados(ARQUIVO, normalizar=NORMALIZAR)
    
    import os
    nome_base = ARQUIVO.replace('.txt', '')
    pasta_saida = f'2_graficos'
    os.makedirs(pasta_saida, exist_ok=True)
    clustering.estados_salvos[0] = clustering.grid.copiar_estado()
    
    print("\nExecutando simulação...\n")
    tempo = clustering.simular(
        num_iteracoes=NUM_ITERACOES,
        verbose=True,
        intervalo_log=100000,
        iteracoes_para_plotar=[0, 100000, 300000, 500000, 1000000, 1500000, 2000000],
        pasta_saida=pasta_saida,
        intervalo_metricas=100000
    )
    
    print("\n" + "=" * 75)
    print("SIMULAÇÃO CONCLUÍDA!")
    print("=" * 75)
    metricas = clustering.calcular_metricas()
    print(f"\nTempo total: {tempo:.1f}s ({tempo/60:.1f}min)")
    print(f"Estado final: Grid={metricas['itens_no_grid']}, Carregando={metricas['formigas_carregando']}, Pureza={metricas['pureza_media']:.3f}")
    print(f"\nParâmetros: k1={K1}, k2={K2}, alpha={ALPHA}, raio={RAIO_VISAO}, temp={TEMPERATURA_MOVIMENTO}")

    print("=" * 75)

