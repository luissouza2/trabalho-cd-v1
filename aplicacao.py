import spacy
import nltk
from spacy.cli import download
from googleapiclient.discovery import build
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
import unicodedata
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import collections
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.pyplot as plt

# Baixar modelo do spaCy se não estiver instalado
try:
    spacy.load("pt_core_news_sm")
except OSError:
    download("pt_core_news_sm")

chave_youtube = "AIzaSyD45DMsi8ON4ZjlTaANKnMEiao4EkPVSjE"
youtube = build('youtube','v3', developerKey=chave_youtube)

def data_valida(data: str) -> bool:
    data_inicio = datetime(2023, 8, 1)
    data_fim = datetime(2024, 8, 31)

    data_convertida = datetime.strptime(data, "%Y-%m-%d")

    return data_inicio <= data_convertida <= data_fim

#extrair informações de uma playlist
def obtem_infos_playlist(playlist_id: str) -> dict:
  next_page_token = None
  infos_playlist = []
  videos_id = []
  titulos_videos = []

  while True:
    requisicao = youtube.playlistItems().list(part='snippet', playlistId = playlist_id, pageToken=next_page_token).execute()
    infos_playlist += requisicao['items']
    next_page_token = requisicao.get('nextPageToken')

    if next_page_token is None:
      break

  for video in infos_playlist:
    data = video["snippet"]["publishedAt"][:10]

    if data_valida(data):
      video_id = video["snippet"]["resourceId"]["videoId"]
      videos_id.append(video_id)

      titulo_video = video["snippet"]["title"]
      titulos_videos.append(titulo_video)

  return {
      "VideosId": videos_id,
      "TitulosVideos": titulos_videos
  }

def obtem_legendas(videos_ids: list) -> list:
  legendas = []

  for video_id in videos_ids:
    try:
      transcricao = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt'])
    except Exception as e:
      print(f"Erro no video {video_id}")
      transicao = None

    legendas.append(transcricao)

  return legendas

# PRIMO RICO
playlist_id_thiago_nigro = "PLHvsswG2Q9jLekFHaBYgTP83-G4MfA_J9"
infos_playlist_thiago_nigro = obtem_infos_playlist(playlist_id_thiago_nigro)
videos_id_thiago_nigro = infos_playlist_thiago_nigro["VideosId"]
legendas_primo_rico = obtem_legendas(videos_id_thiago_nigro)

# BRUNO PERINI
playlist_id_bruno_perini = "PL5W1zc_5Ic9rHS7QI9kdwPn3jw2dHjzUK"
infos_playlist_bruno_perini = obtem_infos_playlist(playlist_id_bruno_perini)
videos_ids_bruno_perini = infos_playlist_bruno_perini["VideosId"]
legendas_bruno_perini = obtem_legendas(videos_ids_bruno_perini)

#extrair apenas os textos das lengendas
def extrai_textos(legendas: list) -> list:
  textos = []

  for i in range(len(legendas)):
    video_legenda = ""

    for j in range(len(legendas[i])):
      video_legenda += legendas[i][j]['text'] + " "

    video_legenda = video_legenda.casefold()
    video_legenda = unicodedata.normalize('NFD', video_legenda)
    video_legenda = ''.join(char for char in video_legenda if unicodedata.category(char) != 'Mn')
    textos.append(video_legenda)

  return textos

# Baixar recursos do NLTK
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Carregar modelo do spaCy para lematização
nlp = spacy.load("pt_core_news_sm")

# Inicializar analisador de sentimentos
sia = SentimentIntensityAnalyzer()

acoes_conhecidas = [
    "Petrobras", "banco do brasil", "Cemig", "Itaú", "Bradesco", "Magazine Luiza", "Via Varejo",
    "CCR", "Raia Drogasil", "Equatorial", "Hapvida", "Ambev", "EcoRodovias",
    "B3", "Copasa", "Klabin","Banco Brasil", "Banco ABC", "Sanepar", "Transmissão Paulista"
]

# Função para remover stopwords e aplicar lematização
def processa_texto(texto: str) -> str:
    stop_words = set(stopwords.words("portuguese"))
    doc = nlp(texto)
    palavras_processadas = [token.lemma_ for token in doc if token.text.lower() not in stop_words]

    return " ".join(palavras_processadas)

# TALVEZ DÊ PARA TIRAR
# Processar arquivo e associar IDs corretos
def processa_arquivo_com_ids(nome_arquivo):
    with open(nome_arquivo, "r", encoding="utf-8") as file:
        linhas = file.readlines()

    id_atual = 0
    texto_processado = []
    id_por_linha = {}

    for i, linha in enumerate(linhas):
        if linha.strip() == "":  # Incrementa ID para cada linha em branco
            id_atual += 1
        else:
            id_por_linha[i] = id_atual  # Associa a linha ao ID atual
            texto_processado.append((linha.strip(), id_atual))  # Salva a linha e o ID correspondente

    return texto_processado

# Função para extrair contexto das ações com ID correto
def extrair_contexto_com_id(texto_processado):
    dados = []

    for linha, id_linha in texto_processado:
        palavras = linha.split()
        for i, palavra in enumerate(palavras):
            for acao in acoes_conhecidas:
                if palavra.lower() == acao.lower():
                    # Extrai 5 palavras antes e 5 palavras depois
                    inicio = max(i - 5, 0)
                    fim = min(i +11, len(palavras))
                    frase = " ".join(palavras[inicio:fim])

                    # Classifica o sentimento da frase
                    sentimento = sia.polarity_scores(frase)["compound"]
                    sentimento = "positivo" if sentimento >= 0.05 else "negativo" if sentimento <= -0.05 else "neutro"

                    # Adiciona ao dataset
                    dados.append([frase, acao, id_linha, sentimento])

    return dados

def salvar_em_txt(influencer: str, texto: list) -> None:
  nome_arquivo = "legendas-primorico.txt" if influencer == "primo_rico" else "legendas-brunoperine.txt"

  with open(nome_arquivo, 'w',encoding='utf-8') as file:
    for item in texto:
        file.write(item + '\n\n')

texto_primo_rico = extrai_textos(legendas_primo_rico)
salvar_em_txt("primo_rico", texto_primo_rico)

# Lista de possíveis ações citadas (com base no contexto do vídeo)
acoes_conhecidas = [
    "Petrobras", "banco do brasil", "Cemig", "Itaú", "Bradesco", "Magazine Luiza", "Via Varejo",
    "CCR", "Raia Drogasil", "Equatorial", "Hapvida", "Ambev", "EcoRodovias",
    "B3", "Copasa", "Klabin","Banco Brasil", "Banco ABC", "Sanepar", "Transmissão Paulista"
]

# Função para contar as menções às ações no texto
def contar_acoes(texto):
    contagem = collections.Counter()

    # Criar um padrão regex para capturar os nomes das ações
    padrao = re.compile(r"\b(" + "|".join(acoes_conhecidas) + r")\b", re.IGNORECASE)

    # Encontrar todas as ocorrências no texto
    matches = padrao.findall(texto)

    # Contar as ocorrências
    for match in matches:
        contagem[match] += 1

    return contagem

# Processar texto
texto = " ".join(texto_primo_rico)
contagem_acoes = contar_acoes(texto)

# Preparar dados para o gráfico
acoes = [acao for acao, quantidade in contagem_acoes.most_common()]
quantidades = [quantidade for acao, quantidade in contagem_acoes.most_common()]

# Criar gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(acoes, quantidades, color='skyblue')
plt.xlabel('Ações')
plt.ylabel('Número de Menções')
plt.title('Ações mais citadas')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Mostrar gráfico
plt.show()

# Nome do arquivo de legendas
arquivo = "legendas-primorico.txt"

# Processar texto e IDs
texto_processado = processa_arquivo_com_ids(arquivo)

# Extrair contexto das ações com ID correto
dados_acoes = extrair_contexto_com_id(texto_processado)

# Criar DataFrame e salvar CSV
df = pd.DataFrame(dados_acoes, columns=["Frase", "Ação", "Id do Elemento", "Sentimento"])
df.to_csv("pre_acoes_primorico.csv", index=False, encoding="utf-8")

texto_bruno_perini = extrai_textos(legendas_bruno_perini)
salvar_em_txt("bruno_perini", texto_bruno_perini)

# Nome do arquivo de legendas
arquivo = "legendas-brunoperine.txt"

# Processar texto e IDs
texto_processado = processa_arquivo_com_ids(arquivo)

# Extrair contexto das ações com ID correto
dados_acoes = extrair_contexto_com_id(texto_processado)

# Criar DataFrame e salvar CSV
df = pd.DataFrame(dados_acoes, columns=["Frase", "Ação", "Id do Elemento", "Sentimento"])
df.to_csv("pre_acoes_brunoperine.csv", index=False, encoding="utf-8")

# Baixar stopwords do NLTK (se ainda não tiver feito)
nltk.download('stopwords')

# Carregar os dados
df_primorico = pd.read_csv("acoes_processadas_primorico.csv")
df_brunoperine = pd.read_csv("acoes_processadas_brunoperine.csv")

# Concatenar os dois DataFrames
df = pd.concat([df_primorico, df_brunoperine], ignore_index=True)

# Remover stopwords
stop_words = set(stopwords.words('portuguese'))

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])

df['Frase'] = df['Frase'].apply(remove_stopwords)

# Vetorização das frases usando TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limitando a 5000 features para evitar sobrecarga
X = vectorizer.fit_transform(df['Frase'])

# Variável alvo (sentimento)
y = df['Sentimento']

# Binarizar as labels para multiclasse (necessário para a curva ROC)
y_bin = label_binarize(y, classes=['positivo', 'negativo', 'neutro'])
n_classes = y_bin.shape[1]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_test_bin = label_binarize(y_test, classes=['positivo', 'negativo', 'neutro'])

# Treinar o modelo Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Fazer previsões
y_pred = rf_classifier.predict(X_test)
y_pred_proba = rf_classifier.predict_proba(X_test)

# Avaliar o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['positivo', 'negativo', 'neutro'], yticklabels=['positivo', 'negativo', 'neutro'])
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()

# Curva ROC para multiclasse
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])

# Plotar a curva ROC para cada classe
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='Curva ROC da classe {0} (AUC = {1:0.2f})'
             ''.format(['positivo', 'negativo', 'neutro'][i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC para Multiclasse')
plt.legend(loc="lower right")
plt.show()

# Gráfico de Precisão, Recall e F1-Score por classe
report = classification_report(y_test, y_pred, output_dict=True)
metrics_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
metrics_df.drop('accuracy', errors='ignore').plot(kind='bar', figsize=(12, 6))
plt.title('Precisão, Recall e F1-Score por Classe')
plt.xlabel('Classes')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.show()

# Carregar os dados
df_bruno = pd.read_csv('acoes_processadas_brunoperine.csv')
df_primorico = pd.read_csv('acoes_processadas_primorico.csv')

# Adicionar uma coluna para identificar o influenciador
df_bruno['Influenciador'] = 'Bruno Perine'
df_primorico['Influenciador'] = 'Thiago Nigro'

# Juntar os dados em um único DataFrame
df = pd.concat([df_bruno, df_primorico])

# Função para analisar sentimentos por ação e Id
def analisar_sentimentos(df, influenciador):
    print(f"Análise para {influenciador}:")

    # Filtrar dados do influenciador
    df_filtrado = df[df['Influenciador'] == influenciador]

    # Agrupar por 'Id do Elemento' e 'Ação', e contar os sentimentos
    sentiment_counts = df_filtrado.groupby(['Id do Elemento', 'Ação', 'Sentimento']).size().unstack(fill_value=0)

    # Adicionar uma coluna de total de menções
    sentiment_counts['Total'] = sentiment_counts.sum(axis=1)

    # Calcular a proporção de cada sentimento
    sentiment_counts['% Positivo'] = sentiment_counts['positivo'] / sentiment_counts['Total'] * 100
    sentiment_counts['% Neutro'] = sentiment_counts['neutro'] / sentiment_counts['Total'] * 100
    sentiment_counts['% Negativo'] = sentiment_counts['negativo'] / sentiment_counts['Total'] * 100

    # Exibir os resultados
    print(sentiment_counts)

    # Visualização para cada ação
    acoes = df_filtrado['Ação'].unique()
    for acao in acoes:
        df_acao = sentiment_counts.loc[sentiment_counts.index.get_level_values('Ação') == acao]

        if not df_acao.empty:
            df_acao[['% Positivo', '% Neutro', '% Negativo']].plot(kind='bar', stacked=True, figsize=(10, 6))
            plt.title(f'Sentimentos sobre {acao} ao longo do tempo - {influenciador}')
            plt.xlabel('Id do Elemento (Mês)')
            plt.ylabel('Proporção de Sentimentos (%)')
            plt.legend(title='Sentimento')
            plt.show()


analisar_sentimentos(df, 'Thiago Nigro')
analisar_sentimentos(df, 'Bruno Perine')