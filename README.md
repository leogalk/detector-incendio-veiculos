# Sistema Inteligente de Detecção de Incêndio em Veículos

Interface web interativa desenvolvida em **Python** e **Streamlit** para classificação automatizada de imagens de veículos quanto à presença de incêndio, aplicando técnicas de Visão Computacional (espaços de cor RGB e HSV) e Aprendizado de Máquina (**Gradient Boosting Classifier**)

Projeto desenvolvido para a disciplina de **Projetos com Aprendizado de Máquina** (Universidade de Pernambuco - UPE).

---

## Acesso à Aplicação Online (Deploy)
* **Link público funcional:** [https://detector-incendio-veiculos.streamlit.app/]

---

## Visão Geral do Modelo e Pipeline
* **Bases de Dados Utilizadas:** Integração entre *Vehicle Fire Images - FireNet* (classe positiva: veículos em chamas) e *Stanford Cars Dataset* (classe negativa: veículos normais).
* **Extração de Atributos (14 Features):** Redimensionamento padronizado ($224 \times 224$ pixels) com extração de estatísticas cromáticas nos canais RGB e HSV, proporção de pixels ativos de chama (`fire_pixel_ratio`), índice de intensidade térmica, dominância do canal vermelho e codificação ordinal de brilho.
* **Pré-processamento sem Data Leakage:** Pipeline composto por imputação pela mediana (`SimpleImputer`), tratamento de outliers via winsorização (`WinsorizerIQR` entre 1% e 99%) e escalonamento robusto (`RobustScaler`).
* **Classificador:** `GradientBoostingClassifier` (`n_estimators=150`, `learning_rate=0.08`, `max_depth=5`) combinado com limiar de segurança para maximizar a detecção de focos de incêndio.

---

## Estrutura dos Arquivos
* `app.py`: Código-fonte da interface web interativa, extração de features em tempo real e visualização diagnóstica[cite: 1].
* `dados_treino_fogo.joblib`: Conjunto de treinamento estruturado utilizado para ajustar o pipeline em cache na inicialização da aplicação.
* `requirements.txt`: Lista de bibliotecas Python necessárias para execução local e em nuvem.
* `README.md`: Documentação de instalação e uso do sistema.

---

##  Como Executar Localmente
1. Certifique-se de ter o **Python 3.10+** instalado e acesse a pasta do projeto pelo terminal:
   ```bash
   cd detector-incendio-veiculos
