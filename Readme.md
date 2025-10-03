# Ant Clustering Simulation

Implementação de algoritmos de clustering baseados em colônias de formigas para agrupamento de dados homogêneos e heterogêneos.

---

## Estrutura do Projeto

- **homogeneo.py** → Clustering com itens homogêneos (todos do mesmo tipo)  
- **config_1.py** → Configuração 1 para itens heterogêneos  
- **config_2.py** → Configuração 2 para itens heterogêneos (50M iterações)  
- **config_3.py** → Configuração 3 para itens heterogêneos (5M iterações)  
- **config_ajustada.py** → Configuração ajustada (30M iterações)  
- **teste_inicial.py** → Teste inicial rápido (500K iterações)  

---

## Como Executar

### Usando Make

```bash
# Ver todos os comandos disponíveis
make help

# Executar configurações individuais
make homogeneo    # Itens homogêneos
make 1            # Configuração 1
make 2            # Configuração 2
make 3            # Configuração 3
make ajustado     # Configuração ajustada
make inicial      # Teste rápido

# Executar todas as simulações
make all

# Limpar gráficos gerados
make clean
```

### Execução Direta 

```bash
python3 homogeneo.py
python3 config_1.py
python3 config_2.py
python3 config_3.py
python3 config_ajustada.py
python3 teste_inicial.py
```

---

## Arquivos de Dados

Certifique-se de ter os arquivos de dados na mesma pasta:

- `base_sintetica_4_grupos.txt`
- `base_sintetica_15_grupos.txt` *(opcional)*

---

## Saída

Cada simulação gera gráficos em sua própria pasta:

- `1_graficos/` → Resultados de **homogeneo.py**  
- `2_graficos/` → Resultados de **config_1.py**  
- `4_graficos/` → Resultados de **config_2.py**  
- `5_graficos/` → Resultados de **config_3.py**  
- `config_ajustada/` → Resultados de **config_ajustada.py**  
- `teste_graficos/` → Resultados de **teste_inicial.py**  
