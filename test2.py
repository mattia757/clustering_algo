import nltk
import umap
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import string

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from transformers import AutoTokenizer, AutoModel

nltk.download('punkt')
nltk.download('stopwords')

sentences = [
    "Acquisto online di vestiti 4 !",
    "Spesa presso negozio di articoli sportivi",
    "Ristorante italiano per cena",
    "Acquisto di susine al mercato",
    "Prenotazione di biglietti per un concerto",
    "Viaggio in treno per visita turistica",
    "Acquisto di cibo e bevande al supermercato",
    "Abbonamento palestra per il fitness",
    "Prenotazione albergo per vacanza",
    "Acquisto biglietti cinema",
    "Spesa al bar per caffè e cornetto",
    "Acquisto libri online",
    "Prenotazione ristorante per compleanno",
    "Acquisto articoli per la casa",
    "Corso di cucina",
    "Prenotazione spa per trattamento relax",
    "Acquisto regalo per matrimonio",
    "Abbonamento streaming per film e serie TV",
    "Acquisto prodotti di bellezza",
    "Prenotazione biglietti parco divertimenti",
    "Acquisto biglietti per partita di calcio",
    "Cena in famiglia al ristorante",
    "Acquisto souvenir durante viaggio",
    "Prenotazione tour guidato",
    "Acquisto accessori per lo sport",
    "Aperitivo con amici al bar",
    "Prenotazione lezioni di yoga",
    "Acquisto gadget tecnologici",
    "Spesa settimanale al supermercato",
    "Prenotazione escursione in montagna",
    "Acquisto prodotti biologici",
    "Pranzo veloce al bar",
    "Prenotazione biglietti per spettacolo teatrale",
    "Acquisto abbigliamento sportivo",
    "Visita museo durante viaggio",
    "Prenotazione corsi di danza",
    "Acquisto strumenti musicali",
    "Cena romantica al ristorante",
    "Prenotazione volo per vacanza",
    "Acquisto giocattoli per bambini",
    "Visita zoo con la famiglia",
    "Prenotazione lezioni di pilates",
    "Acquisto gadget per la casa",
    "Cena di lavoro al ristorante",
    "Prenotazione crociera",
    "Acquisto articoli per il giardinaggio",
    "Pausa caffè durante lavoro",
    "Prenotazione biglietti per parco acquatico",
    "Acquisto accessori per animali domestici",
    "Pranzo al sacco durante escursione",
    "Prenotazione tour in bicicletta",
    "Acquisto articoli per la scuola",
    "Cena con amici al ristorante",
    "Prenotazione biglietti per concerto",
    "Acquisto prodotti per la pulizia della casa",
    "Spesa al mercato per frutta e verdura",
    "Prenotazione lezioni di fitness",
    "Acquisto souvenir durante viaggio",
    "Aperitivo in terrazza con vista",
    "Prenotazione tour gastronomico",
    "Acquisto articoli per il campeggio",
    "Cena di compleanno al ristorante",
    "Prenotazione biglietti per parco divertimenti",
    "Acquisto giochi da tavolo",
    "Visita giardini botanici",
    "Prenotazione lezioni di pittura",
    "Acquisto accessori per la casa",
    "Pranzo al sacco durante escursione",
    "Prenotazione biglietti per museo",
    "Acquisto gadget per la cucina",
    "Cena di anniversario al ristorante",
    "Prenotazione weekend in agriturismo",
    "Acquisto articoli per il tempo libero",
    "Spesa al supermercato per la settimana",
    "Prenotazione lezioni di zumba",
    "Acquisto souvenir durante viaggio",
    "Aperitivo con vista panoramica",
    "Prenotazione degustazione vini",
    "Acquisto attrezzatura per il trekking",
    "Cena con colleghi al ristorante",
    "Prenotazione tour in barca",
    "Acquisto decorazioni per la casa",
    "Escursione in montagna con pranzo al sacco",
    "Prenotazione biglietti per parco avventura",
    "Acquisto articoli per il fai da te",
    "Visita planetario",
    "Prenotazione lezioni di cucina",
    "Acquisto regalo per festa di laurea",
    "Spesa al mercato per ingredienti freschi",
    "Prenotazione biglietti per evento sportivo",
    "Acquisto articoli per la pesca",
    "Cena di Natale al ristorante",
    "Prenotazione weekend in spa",
    "Acquisto giocattoli educativi",
    "Tour enogastronomico in campagna",
    "Prenotazione lezioni di piloxing",
    "Acquisto articoli per la casa in saldo",
    "Pausa caffè con amici al bar",
    "Prenotazione biglietti per parco divertimenti acquatico",
    "Acquisto accessori per il ciclismo",
    "Cena di San Valentino al ristorante",
    "Prenotazione tour culturale",
    "Acquisto articoli per la ristorazione",
    "Spesa al supermercato per la famiglia",
    "Prenotazione lezioni di arti marziali",
    "Acquisto souvenir durante viaggio",
    "Aperitivo in terrazza con amici",
    "Prenotazione degustazione formaggi",
    "Acquisto abbigliamento casual",
    "Cena di Pasqua al ristorante",
    "Prenotazione weekend in montagna",
    "Acquisto giocattoli per animali domestici",
    "Escursione in natura con picnic",
    "Prenotazione biglietti per parco tematico",
    "Acquisto articoli per la casa vintage",
    "Visita osservatorio astronomico",
    "Prenotazione lezioni di artigianato",
    "Acquisto regalo per compleanno",
    "Spesa al mercato biologico",
    "Prenotazione biglietti per partita di basket",
    "Acquisto accessori per la fotografia",
    "Cena di Capodanno al ristorante",
    "Prenotazione tour in jeep",
    "Acquisto decorazioni natalizie",
    "Pranzo al sacco durante gita fuori porta",
    "Prenotazione biglietti per concerto all'aperto",
    "Acquisto articoli per la casa minimalisti",
    "Aperitivo in terrazza con vista sul mare",
    "Prenotazione tour naturalistico",
    "Acquisto abbigliamento vintage",
    "Cena di Halloween al ristorante",
    "Prenotazione weekend in castello",
    "Acquisto giocattoli educativi per bambini",
    "Tour enogastronomico in città",
    "Prenotazione lezioni di scultura",
    "Acquisto articoli per la casa eco-sostenibili",
    "Spesa al mercato per frutta di stagione",
    "Prenotazione biglietti per parco zoologico",
    "Acquisto gadget tecnologici",
    "Cena in spiaggia al ristorante",
    "Prenotazione tour fotografico",
    "Acquisto souvenir durante viaggio",
    "Escursione in montagna con pranzo al rifugio",
    "Prenotazione biglietti per parco dei dinosauri",
    "Acquisto articoli per la casa shabby chic",
    "Visita acquario con la famiglia",
    "Prenotazione lezioni di teatro",
    "Acquisto regalo per anniversario",
    "Pranzo al sacco durante giornata al mare",
    "Prenotazione biglietti per parco divertimenti indoor",
    "Acquisto accessori per la bicicletta",
    "Cena di ringraziamento al ristorante",
    "Prenotazione tour enologico",
    "Acquisto decorazioni pasquali",
    "Aperitivo in terrazza con vista sulla città",
    "Prenotazione tour archeologico",
    "Acquisto abbigliamento alla moda",
    "Cena di carnevale al ristorante",
    "Prenotazione weekend in agriturismo",
    "Acquisto giocattoli interattivi",
    "Tour enogastronomico in campagna",
    "Prenotazione lezioni di danza del ventre",
    "Acquisto articoli per la casa vintage",
    "Spesa al mercato per frutta e verdura fresca",
    "Prenotazione biglietti per evento sportivo",
    "Acquisto articoli per il fai da te",
    "Cena di compleanno al ristorante",
    "Prenotazione weekend in spa",
    "Acquisto gadget tecnologici",
    "Tour gastronomico in città",
    "Prenotazione lezioni di yoga",
    "Acquisto souvenir durante viaggio",
    "Pranzo al sacco durante escursione in montagna",
    "Prenotazione biglietti per parco divertimenti",
    "Acquisto accessori per il campeggio",
    "Cena con amici al ristorante",
    "Prenotazione tour in barca",
    "Acquisto decorazioni per la casa",
    "Escursione in montagna con picnic",
    "Prenotazione biglietti per parco tematico",
    "Acquisto articoli per la casa vintage",
    "Visita osservatorio astronomico",
    "Prenotazione lezioni di artigianato",
    "Acquisto regalo per compleanno",
    "Spesa al mercato biologico",
    "Prenotazione biglietti per partita di basket",
    "Acquisto accessori per la fotografia",
    "Cena di Capodanno al ristorante",
    "Prenotazione tour in jeep",
    "Acquisto decorazioni natalizie",
    "Pranzo al sacco durante gita fuori porta",
    "Prenotazione biglietti per concerto all'aperto",
    "Acquisto articoli per la casa minimalisti",
    "Aperitivo in terrazza con vista sul mare",
    "Prenotazione tour naturalistico",
    "Acquisto abbigliamento vintage",
    "Cena di Halloween al ristorante",
    "Prenotazione weekend in castello",
    "Acquisto giocattoli educativi per bambini",
    "Tour enogastronomico in città",
    "Prenotazione lezioni di scultura",
    "Acquisto articoli per la casa eco-sostenibili",
    "Spesa al mercato per frutta di stagione",
    "Prenotazione biglietti per parco zoologico",
    "Acquisto gadget tecnologici",
    "Cena in spiaggia al ristorante",
    "Prenotazione tour fotografico",
    "Acquisto souvenir durante viaggio",
    "Escursione in montagna con pranzo al rifugio",
    "Prenotazione biglietti per parco dei dinosauri",
    "Acquisto articoli per la casa shabby chic",
    "Visita acquario con la famiglia",
    "Prenotazione lezioni di teatro",
    "Acquisto regalo per anniversario",
    "Pranzo al sacco durante giornata al mare",
    "Prenotazione biglietti per parco divertimenti indoor",
    "Acquisto accessori per la bicicletta",
    "Cena di ringraziamento al ristorante",
    "Prenotazione tour enologico",
    "Acquisto decorazioni pasquali",
    "Aperitivo in terrazza con vista sulla città",
    "Prenotazione tour archeologico",
    "Acquisto abbigliamento alla moda",
    "Cena di carnevale al ristorante",
    "Prenotazione weekend in agriturismo",
    "Acquisto giocattoli interattivi"
]
sentences = list(set(sentences))

def preprocess_sentence(sentences):
    # Tokenizzazione del testo
    tokens = nltk.word_tokenize(sentences)

    # Rimozione delle stopwords
    stop_words = set(stopwords.words('italian'))
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # Rimozione dei caratteri speciali, numeri e punteggiatura
    tokens = [word for word in tokens if word.isalpha()]

    # Ricostruzione della frase preprocessata
    preprocessed_sentence = ' '.join(tokens)

    return preprocessed_sentence

# Esempio di utilizzo
preprocessed_sentences = [preprocess_sentence(sentence) for sentence in sentences]
#print("Frasi originali:", sentences)
#print("\nFrasi preprocessate:")
#for i, preprocessed in enumerate(preprocessed_sentences):
    #print(f"{i + 1}. {preprocessed}")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Sentences are encoded by calling model.encode()
embeddings = model.encode(preprocessed_sentences)

# Print the embeddings
# for preprocessed_sentence, embedding in zip(preprocessed_sentences, embeddings):
#     print("Sentence:", preprocessed_sentence)
#     print("Embedding:", embedding)
#     print("")

num_clusters = 27

# Calcolo dell'inertia per valutare il numero ottimale di cluster
inertia = []

kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(embeddings)
inertia.append(kmeans.inertia_)

# Plot del metodo del gomito
# plt.plot(range(num_clusters), inertia, marker='s')
# plt.xlabel('Numero di cluster')
# plt.ylabel('Inertia')
# plt.title('Metodo del gomito per la scelta del numero di cluster')
# plt.show()

# Esegui il clustering con K-Means sugli embeddings originali
kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(embeddings)

# Calcola il Davies-Bouldin Index
davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
print('davies_bouldin_score: ', davies_bouldin)

# Creazione di un dizionario per raggruppare le frasi per cluster
cluster_sentences = {}
for i, label in enumerate(cluster_labels):
    if label not in cluster_sentences:
        cluster_sentences[label] = []
    cluster_sentences[label].append(sentences[i])

# Stampa le frasi per ciascun cluster
for cluster, sentences in cluster_sentences.items():
    print(f"Cluster {cluster}:")
    for sentence in sentences:
        print(sentence)
    print()

print(len(preprocessed_sentences))
print(len(cluster_sentences))


# Riduzione dimensionale degli embeddings utilizzando UMAP
#umap_embeddings = umap.UMAP(n_neighbors=180, min_dist=0.1, n_components=2).fit_transform(embeddings)

# Plot dei risultati della riduzione dimensionale
# plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=5)
# plt.title('Riduzione dimensionale degli embeddings con UMAP')
# plt.xlabel('UMAP Dimension 1')
# plt.ylabel('UMAP Dimension 2')
# plt.show()

#print(umap_embeddings.shape)