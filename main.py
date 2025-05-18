import scipy
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"  # Dataset from UCI Machine Learning Repository
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] # Column names
dataset = read_csv(url, names=names) # Dataset will contain the data from the url organized by the columns defined in "names"

# Printing Shape, which consists of instances (rows) and attributes (columns)
print("----------------SHAPE:--------------- \n")
print(dataset.shape)

# Printing Head, actually viewing 20 rows of data
print("----------------HEAD 20:--------------- \n")
print(dataset.head(20))

# Looking at data by attribute
print("----------------ATTRIBUTE DESCRIPTION:--------------- \n")
print(dataset.describe())

# Looking at rows that belong to each class
print("----------------CLASS DESCRIPTION:--------------- \n")
print(dataset.groupby('class').size())

# Questo box plot serve per capire come sono distribuiti i numeri in ogni colonna del dataset.
# Ogni "scatola" (box) rappresenta una colonna e ti fa vedere:
# - La riga nel mezzo è la mediana: cioè il numero che sta proprio a metà se ordini tutti i valori.
# - Il box (la scatola) va dal 25% al 75% dei valori: mostra dove si concentrano la maggior parte dei dati.
# - Le linee che escono dalla scatola (chiamate "baffi" o whiskers) vanno verso i valori più piccoli e più grandi,
#   ma solo finché sono ancora "normali" (non troppo lontani).
# - I puntini fuori dai baffi sono i valori anomali (outlier): numeri troppo alti o troppo bassi rispetto agli altri.
# In pratica, con questo grafico puoi vedere velocemente com’è fatta ogni colonna: se ha valori strani, se i dati
# sono distribuiti in modo equilibrato o se sono tutti spostati verso l’alto o il basso.
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# Istogrammi sulle colonne
dataset.hist()
plt.show()

# Questo comando crea una "scatter plot matrix", cioè una griglia di grafici a dispersione (uno per ogni coppia di colonne numeriche).
# Ogni piccolo grafico mostra come si comportano due variabili del dataset una rispetto all’altra:
# - sull'asse orizzontale c'è una variabile, su quello verticale un'altra.
# - ogni punto rappresenta una riga del dataset, cioè un'osservazione (es. una persona, un oggetto, ecc.).

# Il grafico serve soprattutto per vedere se c’è una "correlazione", cioè:
# - Se al crescere di una variabile cresce anche l’altra (correlazione positiva), i punti tendono a formare una linea diagonale che sale.
# - Se al crescere di una variabile l’altra diminuisce (correlazione negativa), i punti vanno in diagonale ma verso il basso.
# - Se non c’è nessuna relazione, i punti sono sparsi a caso senza una forma chiara.

# Nella diagonale della griglia spesso ci sono istogrammi o curve: mostrano com’è distribuita ogni variabile da sola (quali valori sono più frequenti).

# In sintesi: questo grafico ti aiuta a capire se alcune colonne del dataset si influenzano tra loro
# (cioè se sapere il valore di una ti aiuta a prevedere l’altra), e a individuare gruppi, tendenze o valori anomali.
scatter_matrix(dataset)
plt.show()

# Qui si divide il dataset in due parti: una per addestrare il modello (train) e una per testarlo (validation).

# 1. `array = dataset.values`
#    Converte il DataFrame Pandas in un array NumPy. È utile per usarlo con funzioni di scikit-learn che si aspettano array.

# 2. `X = array[:, 0:4]`
#    Seleziona tutte le righe (`:`) e le colonne da 0 a 3 (cioè le prime 4).
#    Queste colonne rappresentano le **caratteristiche** (input) che useremo per fare previsioni.

# 3. `y = array[:, 4]`
#    Seleziona la colonna 4, che è la **variabile target** (output), cioè quella che vogliamo prevedere.

# 4. `train_test_split(...)`
#    Questa funzione (importata da `sklearn.model_selection`) serve per dividere i dati in due insiemi:
#    - uno per l'addestramento del modello (`X_train`, `Y_train`)
#    - uno per la validazione/test del modello (`X_validation`, `Y_validation`)

#    I parametri:
#    - `test_size=0.20`: indica che il **20% dei dati** andrà al set di validazione,
#      e quindi il **restante 80%** verrà usato per l'addestramento.
#    - `random_state=1`: imposta il seme del generatore casuale.
#      Serve per avere sempre la **stessa suddivisione** ogni volta che si esegue il codice (utile per confronto tra modelli).

# Risultato: hai 4 array pronti per allenare il modello su 80% dei dati e testarlo sul 20% rimanente.
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Qui stiamo facendo uno "spot check", cioè una prova veloce di vari algoritmi di Machine Learning.
# L'obiettivo è vedere, senza troppe impostazioni complicate, quali modelli si comportano meglio con il nostro dataset.
# È una tecnica utile all'inizio di un progetto per farsi un'idea su quale modello conviene concentrarsi poi in modo più approfondito.

models = []  # Creiamo una lista vuota dove andremo a inserire diversi modelli di classificazione da provare.

# Adesso aggiungiamo i modelli alla lista.
# Ogni elemento che aggiungiamo è una coppia (tupla): un nome abbreviato del modello e l'oggetto che rappresenta il modello stesso.

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# 'LR' sta per Logistic Regression (regressione logistica), un algoritmo base di classificazione.
# È spesso il punto di partenza nei progetti di classificazione.
# 'liblinear' è il solver (cioè l'algoritmo interno che ottimizza il modello) ed è adatto per dataset piccoli.
# 'multi_class=ovr' significa "One-vs-Rest", cioè il modello viene adattato per classificare più di due classi
# confrontando ogni classe contro tutte le altre una alla volta.

models.append(('LDA', LinearDiscriminantAnalysis()))
# 'LDA' sta per Linear Discriminant Analysis.
# È un algoritmo che cerca di trovare una combinazione lineare di variabili (features) che separi meglio le classi.
# Funziona bene se le classi sono ben distribuite e separate.

models.append(('KNN', KNeighborsClassifier()))
# 'KNN' sta per K-Nearest Neighbors.
# Questo modello classifica un nuovo dato in base ai 'k' dati più vicini a esso nel dataset.
# È molto intuitivo: se i tuoi vicini sono quasi tutti "gatti", probabilmente sei un "gatto" anche tu.

models.append(('CART', DecisionTreeClassifier()))
# 'CART' sta per Classification and Regression Tree.
# È un albero decisionale: costruisce una serie di domande (del tipo "il valore X è maggiore di 3.5?")
# e segue i rami dell'albero fino a decidere la classe a cui appartiene un dato.

models.append(('NB', GaussianNB()))
# 'NB' sta per Naive Bayes, qui nella sua versione Gaussiana.
# È un modello probabilistico molto veloce e semplice.
# Funziona bene se i dati seguono una distribuzione normale (a campana) e se le features sono indipendenti tra loro.

models.append(('SVM', SVC(gamma='auto')))
# 'SVM' sta per Support Vector Machine.
# Questo modello cerca di tracciare il confine (detto iperpiano) che separa al meglio le classi.
# È molto potente per dati complessi, anche se può essere più lento e difficile da ottimizzare.
# 'gamma=auto' imposta un parametro interno che controlla quanto il modello è "rigido" o "flessibile".

# Ora vogliamo testare ciascun modello per vedere quanto è preciso (accuratezza).
# Per farlo useremo una tecnica chiamata "validazione incrociata".

results = []  # Lista dove salveremo i risultati (le percentuali di accuratezza) di ogni modello.
names = []  # Lista dove salveremo solo i nomi dei modelli, per riferimento successivo.

for name, model in models:
    # Usiamo StratifiedKFold, una tecnica di validazione incrociata.
    # Significa che dividiamo il nostro set di dati in 10 parti (fold).
    # Ogni volta alleniamo il modello su 9 parti e lo testiamo sulla parte rimanente.
    # Lo facciamo 10 volte, ruotando ogni volta la parte usata per il test.
    # 'Stratified' vuol dire che ogni fold mantiene la stessa proporzione di classi del dataset originale.
    # Questo è molto importante nei problemi di classificazione per non falsare i risultati.

    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    # n_splits=10 → facciamo 10 divisioni
    # random_state=1 → imposta il seme casuale, per avere risultati ripetibili ogni volta che esegui il codice
    # shuffle=True → mescola i dati prima di fare le divisioni, utile per evitare partizioni sbilanciate

    # cross_val_score esegue effettivamente la validazione incrociata.
    # Allena il modello su 9 parti e lo valuta sulla decima, ripetuto 10 volte.
    # 'scoring="accuracy"' significa che usiamo la metrica "accuratezza", cioè la percentuale di casi classificati correttamente.

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

    # Aggiungiamo i risultati ottenuti alla lista 'results' per usarli più avanti (es. per confronti o grafici)
    results.append(cv_results)

    # Aggiungiamo anche il nome del modello alla lista dei nomi
    names.append(name)

    # Stampiamo i risultati in modo leggibile:
    # - name: il nome del modello
    # - cv_results.mean(): la media delle accuratezze sui 10 fold → ci dice quanto è "buono" il modello in media
    # - cv_results.std(): la deviazione standard delle accuratezze → ci dice se il modello è stabile o se le prestazioni variano molto
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    # Stampiamo i risultati per la visualizzazione su plot
    plt.boxplot(results, tick_labels=names)
    plt.title('Algorithm Comparison')
    plt.show()

    # Ora che abbiamo testato diversi modelli, scegliamo uno di essi (in questo caso la SVM)
    # e lo usiamo per fare previsioni su dati mai visti prima: quelli del validation set (20% dei dati totali).
    # Questo è il test "finale" per vedere come il modello si comporta nel mondo reale, fuori dal training.

    model = SVC(gamma='auto')
    # Creiamo un'istanza del modello SVM (Support Vector Machine).
    # È lo stesso tipo di modello che abbiamo usato prima nello spot check.
    # 'gamma=auto' imposta un parametro tecnico che regola quanto il modello è sensibile ai singoli punti del dataset.
    # Per ora non modifichiamo nulla: lo usiamo con impostazioni base.

    model.fit(X_train, Y_train)
    # Addestriamo (alleniamo) il modello usando il training set.
    # 'X_train' contiene i dati di input (le caratteristiche, cioè le colonne da 0 a 3).
    # 'Y_train' contiene le etichette da prevedere (es. il tipo di fiore, o qualunque altra classificazione).
    # In questa fase il modello "impara" a riconoscere i pattern nei dati per poi poter fare previsioni su nuovi casi.

    predictions = model.predict(X_validation)
    # Facciamo le previsioni sul set di validazione, cioè sui dati che il modello NON ha mai visto.
    # 'X_validation' sono i dati di input (come X_train, ma non usati per l'addestramento).
    # Il risultato sarà un array con una previsione (classe) per ogni riga di 'X_validation'.
    # Ora possiamo confrontare queste previsioni con i valori reali (Y_validation) per valutare l'accuratezza del modello.