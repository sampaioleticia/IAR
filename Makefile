# Makefile para execução dos scripts de clustering

PYTHON = python3

.PHONY: all 1 2 3 homogeneo inicial ajustado clean help

help:
	@echo "Comandos disponíveis:"
	@echo "  make 1         - Executa config_1.py"
	@echo "  make 2         - Executa config_2.py"
	@echo "  make 3         - Executa config_3.py"
	@echo "  make homogeneo - Executa homogeneo.py"
	@echo "  make inicial   - Executa teste_inicial.py"
	@echo "  make ajustado  - Executa config_ajustada.py"
	@echo "  make clean     - Remove pastas de gráficos gerados"
	@echo "  make help      - Mostra esta mensagem"

1:
	$(PYTHON) config_1.py

2:
	$(PYTHON) config_2.py

3:
	$(PYTHON) config_3.py

homogeneo:
	$(PYTHON) homogeneo.py

inicial:
	$(PYTHON) teste_inicial.py

ajustado:
	$(PYTHON) config_ajustada.py

clean:
	rm -rf 1_graficos 2_graficos 4_graficos 5_graficos teste_graficos config_ajustada
	@echo "Pastas de gráficos removidas"

all: homogeneo 1 2 3 ajustado inicial
	@echo "Todas as simulações executadas"