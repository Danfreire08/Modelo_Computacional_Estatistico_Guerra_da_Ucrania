import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, shapiro, bartlett, mannwhitneyu
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import statsmodels.api as sm

print("Importações concluídas com sucesso.")

# Caminho do arquivo CSV
caminho_arquivo = '/Users/daniel/Desktop/TCC ECEME/russia_losses_personnel_alterado.csv'

# Carregar os dados CSV
dados_csv = pd.read_csv(caminho_arquivo)
print("Arquivo CSV carregado com sucesso.")

# Definir variáveis X e y para os testes
X = dados_csv[['day']]
y = dados_csv['personnel']

# Criando objeto KFold para a validação cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Listas para armazenar os scores de cada fold
scores = []

# Loop sobre os folds da validação cruzada
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Criando um modelo de regressão linear
    modelo = sm.OLS(y_train, X_train).fit()

    # Fazendo previsões com o conjunto de teste
    y_pred = modelo.predict(X_test)

    # Calculando o score (nesse caso, usaremos o R-squared)
    score = modelo.rsquared
    scores.append(score)

# Exibindo os scores de cada fold
print("Scores de Validacão Cruzada:", scores)
print("Média dos Scores:", np.mean(scores))

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir variáveis X e y para RandomForestRegressor
X_rf = dados_csv[['day']]
y_rf = dados_csv['personnel']

# Dividir os dados em treinamento e teste para RandomForestRegressor
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
print("Dados divididos em treinamento e teste para RandomForestRegressor.")

# Definir o modelo de Floresta Aleatória (Random Forest)
rf = RandomForestRegressor(random_state=42)

# Definir os parâmetros para GridSearchCV para RandomForestRegressor
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
print("Parâmetros definidos para GridSearchCV para RandomForestRegressor.")

# Iniciar GridSearchCV para encontrar os melhores hiperparâmetros para RandomForestRegressor
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, scoring='r2')
print("Iniciando GridSearchCV para RandomForestRegressor...")
grid_search_rf.fit(X_train_rf, y_train_rf)
print("GridSearchCV concluído para RandomForestRegressor.")

# Melhor modelo encontrado pelo GridSearchCV para RandomForestRegressor
best_rf = grid_search_rf.best_estimator_
print("Melhor modelo encontrado pelo GridSearchCV para RandomForestRegressor.")

# Fazer previsões com o conjunto de teste usando RandomForestRegressor
y_pred_rf = best_rf.predict(X_test_rf)
print("Previsões feitas no conjunto de teste usando RandomForestRegressor.")

# Calcular o MSE e R-squared para RandomForestRegressor
mse_rf = mean_squared_error(y_test_rf, y_pred_rf)
r2_rf = r2_score(y_test_rf, y_pred_rf)
print(f'Mean Squared Error para RandomForestRegressor: {mse_rf}')
print(f'R-squared para RandomForestRegressor: {r2_rf}')

# Calculando a correlação de Spearman para RandomForestRegressor
spearman_corr_rf, spearman_p_value_rf = spearmanr(dados_csv['day'], dados_csv['personnel'])
print("\nCorrelação de Spearman para RandomForestRegressor:")
print(f"Coeficiente de correlação de Spearman: {spearman_corr_rf}")
print(f"P-valor da correlação de Spearman: {spearman_p_value_rf}")

# integrar o código para a regressão linear com statsmodels

# Adicionar a constante para o termo independente (intercepto) para a regressão linear
X_sm = sm.add_constant(dados_csv[['day']])
y_sm = dados_csv['personnel']

# Dividir os dados em treinamento e teste para statsmodels
X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)
print("Dados divididos em treinamento e teste para statsmodels (Regressão Linear).")

# Criar modelo de regressão linear com statsmodels
modelo_sm = sm.OLS(y_train_sm, X_train_sm).fit()
print("Modelo de regressão linear ajustado com statsmodels.")

# Fazer previsões com o conjunto de teste usando statsmodels
y_pred_sm = modelo_sm.predict(X_test_sm)
print("Previsões feitas no conjunto de teste usando statsmodels (Regressão Linear).")

# Calcular o MSE e R-squared para statsmodels (Regressão Linear)
mse_sm = mean_squared_error(y_test_sm, y_pred_sm)
r2_sm = r2_score(y_test_sm, y_pred_sm)
print(f'\nMean Squared Error para statsmodels (Regressão Linear): {mse_sm}')
print(f'R-squared para statsmodels (Regressão Linear): {r2_sm}')

# Calculando a correlação de Spearman para statsmodels (Regressão Linear)
spearman_corr_sm, spearman_p_value_sm = spearmanr(dados_csv['day'], dados_csv['personnel'])
print("\nCorrelação de Spearman para statsmodels (Regressão Linear):")
print(f"Coeficiente de correlação de Spearman: {spearman_corr_sm}")
print(f"P-valor da correlação de Spearman: {spearman_p_value_sm}")

# Teste de normalidade de Shapiro-Wilk
shapiro_stat, shapiro_p_value = shapiro(y)
print("\nTeste de Normalidade de Shapiro-Wilk:")
print(f"Estatística de Shapiro-Wilk: {shapiro_stat}")
print(f"P-valor de Shapiro-Wilk: {shapiro_p_value}")

# Teste de variância (Bartlett)
bartlett_stat, bartlett_p_value = bartlett(y_train, y_test)
print("\nTeste de Variância (Bartlett):")
print(f"Estatística de Bartlett: {bartlett_stat}")
print(f"P-valor de Bartlett: {bartlett_p_value}")

# Teste de Mann-Whitney U (substituindo o Wilcoxon)
try:
    mannwhitneyu_stat, mannwhitneyu_p_value = mannwhitneyu(y_train, y_test)
    print("\nTeste de Mann-Whitney U:")
    print(f"Estatística de Mann-Whitney U: {mannwhitneyu_stat}")
    print(f"P-valor de Mann-Whitney U: {mannwhitneyu_p_value}")
except ValueError as e:
    print(f"\nErro ao executar o teste de Mann-Whitney U: {e}")

# Gráfico combinado para RandomForestRegressor e statsmodels (Regressão Linear)

# Estimativa adicional para ambos os modelos
taxa_mortalidade = 0.0002
constante_multiplicativa = 900000
y_estimado_rf = taxa_mortalidade * constante_multiplicativa * X_test_rf['day']
y_estimado_sm = taxa_mortalidade * constante_multiplicativa * X_test_sm['day']

# Definindo o estilo do gráfico (utilizando seaborn para gráficos mais detalhistas)
sns.set(style="whitegrid")

# Ajustando o tamanho e a resolução da figura
plt.figure(figsize=(12, 8), dpi=150)

# Gráfico para RandomForestRegressor
plt.subplot(2, 1, 1)
sns.scatterplot(x=X_test_rf['day'], y=y_test_rf, color='black', label='Dados Reais', s=100, edgecolor='w')
sns.lineplot(x=X_test_rf['day'], y=y_pred_rf, color='blue', linewidth=2, label='RandomForestRegressor')
sns.lineplot(x=X_test_rf['day'], y=y_estimado_rf, color='gray', linewidth=2, linestyle='--', label='Estimativa DAMEPLAN')

plt.title('Regressão com RandomForestRegressor e Estimativa Adicional', fontsize=16)
plt.xlabel('Dia', fontsize=14)
plt.ylabel('Pessoal', fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Gráfico para statsmodels (Regressão Linear)
plt.subplot(2, 1, 2)
sns.scatterplot(x=X_test_sm['day'], y=y_test_sm, color='black', label='Dados Reais', s=100, edgecolor='w')
sns.lineplot(x=X_test_sm['day'], y=y_pred_sm, color='green', linewidth=2, label='Regressão Linear statsmodels')
sns.lineplot(x=X_test_sm['day'], y=y_estimado_sm, color='gray', linewidth=2, linestyle='--', label='Estimativa DAMEPLAN')

plt.title('Regressão Linear com statsmodels e Estimativa Adicional', fontsize=16)
plt.xlabel('Dia', fontsize=14)
plt.ylabel('Pessoal', fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Ajustes finais para o layout da figura
plt.tight_layout()

# Salvando a figura com uma resolução mais alta
resultado_filepath = '/Users/daniel/Desktop/TCC ECEME/resultado_regressao_comparativa.png'
plt.savefig(resultado_filepath, dpi=150, bbox_inches='tight')

# Exibir o gráfico
plt.show()

# Exibir os melhores hiperparâmetros encontrados para RandomForestRegressor
print("\nMelhores Hiperparâmetros encontrados para RandomForestRegressor:")
print(grid_search_rf.best_params_)

# Exibir as estatísticas do modelo de regressão linear com statsmodels
print("\nEstatísticas do modelo de regressão linear com statsmodels:")
print(modelo_sm.summary())
