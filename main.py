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