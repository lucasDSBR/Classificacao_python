from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.svm import SVC
import matplotlib.pyplot as plt

################## Pr�-processamento ###################
# Coleta e Integra��o
iris = load_iris()

caracteristicas = iris.data
rotulos = iris.target

print("Caracteristicas:\n", caracteristicas[:2])
print("R�tulos:\n", rotulos[:2])
print('########################################################')

# Parti��o dos dados
carac_treino, carac_teste, rot_treino, rot_teste = train_test_split(caracteristicas, rotulos)

################### Minera��o #####################

############---------- Arvore de Decisão -----------############
arvore = DecisionTreeClassifier(max_depth=2)
arvore.fit(X=carac_treino, y=rot_treino)

rot_preditos = arvore.predict(carac_teste)
acuracia_arvore = accuracy_score(rot_teste, rot_preditos)

# adicionando visualização da árvore
fig, ax = plt.subplots(figsize=(12, 12))
plot_tree(arvore, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, ax=ax)
plt.show()

############-------- M�quina de Vetor Suporte ------############
clf = SVC()
clf.fit(X=carac_treino, y=rot_treino)

rot_preditos_svm = clf.predict(carac_teste)
acuracia_svm = accuracy_score(rot_teste, rot_preditos_svm)

################ P�s-processamento ####################
print("Acur�cia �rvore de Decis�o:", round(acuracia_arvore, 5))
print("Acur�cia SVM:", round(acuracia_svm, 5))
print('########################################################')

r = export_text(arvore, feature_names=iris['feature_names'])
print("Estrutura da �rvore")
print(r)