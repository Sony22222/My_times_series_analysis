# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  EXERCICE 1 — Random walk : simulation, tests, prévisions naïves             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Simulate and forecast a random walk
# Simulate a different random walk than the one we have worked with in this chapter.
# You can simply change the seed and get new values:
    
# 1 Generate a random walk of 500 timesteps. Feel free to choose an initial value
# different from 0. Also, make sure you change the seed by passing a different
# integer to np.random.seed().
# 2 Plot your simulated random walk.
# 3 Test for stationarity.
# 4 Apply a first-order difference.
# 5 Test for stationarity.
# 6 Split your simulated random walk into a train set containing the first 400 timesteps.
# The remaining 100 timesteps will be your test set.
# 7 Apply different naive forecasting methods and measure the MSE. Which
# method yields the lowest MSE?
# 8 Plot your forecasts.
# 9 Forecast the next timestep over the test set and measure the MSE. Did it decrease?
# 10 Plot your forecasts.

# ── A) Imports & setup ─────────────────────────────────────────────────────────
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# ── B) (1) Simulation d’un random walk (graine et valeur initiale personnalisées)
np.random.seed(2025)
n=500
eps=np.random.normal(0,1,n)
rw=10+np.cumsum(eps)  # valeur initiale=10

# ── C) (2) Visualisation du random walk
plt.figure(dpi=300)  # 150 ou 200 pour plus de netteté
plt.plot(rw)
plt.title('Random Walk (n=500)')
plt.show()

# ── D) (3) Test de stationnarité (ADF) sur la série brute
adf_stat,pval,*_=adfuller(rw)
print(f"ADF stat={adf_stat:.3f}, p-value={pval:.4f}")  # p>0.05 ⇒ non stationnaire

ADF_result = adfuller(rw)
print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

# LA P-VALUE EST SUPERIEUR AU SEUIL DE 0.05 ON NE REJETTE PAS  L'HYPOTHESE NULL DE NON STATIONARITE
# LA SERIE CONTIENT UNE RACINE UNITAIRE

# ── E) (4) Différenciation d’ordre 1 + (5) test ADF sur la diff
d1=np.diff(rw)

# 150 ou 200 pour plus de netteté
plt.figure(dpi=300); plt.plot(d1); plt.title('First Difference'); plt.xlabel('t'); plt.ylabel('Δvalue'); plt.show()

adf_stat_d1,pval_d1,*_=adfuller(d1)
print(f"ADF(Δ) stat={adf_stat_d1:.3f}, p-value={pval_d1:.4f}")  # p<0.05 ⇒ stationnaire
# LA PVALUE EST INFERIEUR AU SEUIL DE 0.05 DONC ON REJETTE LHYPOTHESE NULL DE NON STATIONARITE
# LA SERIE DIFFERENCIE EST STATIONNAIRE 

# ── F) (6) Découpage en train/test
rw = pd.DataFrame({'value': rw})
train=rw[:400]; test=rw[400:]

print(len(rw))
print(len(train), len(test))

# ── G) (7) Méthodes naïves : moyenne / dernière valeur / drift
# Moyenne historique = prédit moyenne du train
mean = np.mean(train)
test.loc[:, 'pred_mean'] = mean
test.head() 

# Predict the last known value
last_value = train.iloc[-1].value
test.loc[:, 'pred_last'] = last_value
test.head()         

# Predict with drift
deltaX = 400 - 0
deltaY = last_value - 10
drift = deltaY / deltaX
x_vals = np.arange(400, 500, 1)
pred_drift = drift * x_vals + 10
test.loc[:, 'pred_drift'] = pred_drift
test.head() 

# ── H) (7) Évaluation : MSE des trois méthodes
from sklearn.metrics import mean_squared_error
mse_mean = mean_squared_error(test['value'], test['pred_mean'])
mse_last = mean_squared_error(test['value'], test['pred_last'])
mse_drift = mean_squared_error(test['value'], test['pred_drift'])
print(mse_mean, mse_last, mse_drift)

# ── I) (8) Visualisation train/test + prévisions naïves
fig, ax = plt.subplots()
ax.plot(train['value'], 'b-')
ax.plot(test['value'], 'b-')
ax.plot(test['pred_mean'], 'r-.', label='Mean')
ax.plot(test['pred_last'], 'g--', label='Last value')
ax.plot(test['pred_drift'], 'k:', label='Drift')
ax.axvspan(400, 500, color='#808080', alpha=0.2)
ax.legend(loc=2)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
plt.tight_layout()

# ── J) (9–10) One-step ahead : décalage d’1 pas + MSE + tracé zoomé
# we take our initial observed value
# and use it to predict the next timestep. Once we record a new value, it will be used as
# a forecast for the following timestep. This process is then repeated into the future.
rw_shift = rw.shift(periods=1)
mse_one_step = mean_squared_error(test['value'], rw_shift[400:])
mse_one_step

plt.close('all')
fig, ax = plt.subplots(dpi=150)
ax.clear()  # vide les anciens tracés de cet axe
ax.plot(rw, 'r-', label='actual')
ax.plot(rw_shift[400:], 'g-.', label='forecast')
ax.legend(loc=2)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
plt.xlim(300, 500)
plt.tight_layout()
plt.show()



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  EXERCICE 2 — GOOGL : split, prévisions naïves, MSE, one-step               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Forecast the daily closing price of GOOGL
# Using the GOOGL dataset that we worked with in this chapter, apply the forecasting
# techniques we’ve discussed and measure their performance:
    
# 1 Keep the last 5 days of data as a test set. The rest will be the train set.
# 2 Forecast the last 5 days of the closing price using naive forecasting methods and
#   measure the MSE. Which method is the best?
# 3 Plot your forecasts.
# 4 Forecast the next timestep over the test set and measure the MSE. Did it
#   decrease?
# 5 Plot your forecasts.

# ── A) Chargement & split (1)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"C:\Users\mbarg\Downloads\dataTS\GOOGL.csv")
df.head()

googl_train = df[['Date','Close']][:-5]  # toutes les lignes sauf les 5 dernières
googl_test  = df[['Date','Close']][-5:]  # uniquement les 5 dernières

print(len(df))
print(len(googl_train), len(googl_train))

# ── B) Méthodes naïves (2)
# Moyenne historique
mean = np.mean(googl_train['Close'])
googl_test.loc[:, 'pred_mean'] = mean

# Dernière valeur connue
last_value = googl_train['Close'].iloc[-1]
googl_test.loc[:, 'pred_last'] = last_value         

# Drift (tendance linéaire)
deltaX = len(googl_train)    # longueur du train
deltaY = last_value - googl_train['Close'].iloc[0]
drift = deltaY / deltaX
x_vals = np.arange(248, 253, 1)
pred_drift = drift * x_vals + googl_train['Close'].iloc[0]
googl_test.loc[:, 'pred_drift'] = pred_drift
googl_test

# ── C) MSE (3)
mse_mean = mean_squared_error(googl_test['Close'], googl_test['pred_mean'])
mse_last = mean_squared_error(googl_test['Close'], googl_test['pred_last'])
mse_drift = mean_squared_error(googl_test['Close'], googl_test['pred_drift'])
print(mse_mean, mse_last, mse_drift)

# ── D) Visualisation (4)
fig, ax = plt.subplots()
ax.plot(googl_train['Close'], 'b-')
ax.plot(googl_test['Close'], 'b-')
ax.plot(googl_test['pred_mean'], 'r-.', label='Mean')
ax.plot(googl_test['pred_last'], 'g--', label='Last value')
ax.plot(googl_test['pred_drift'], 'k:', label='Drift')
ax.axvspan(248, 253, color='#808080', alpha=0.2)
ax.legend(loc=2)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
plt.xlim(230, 252)
plt.tight_layout()

# ── E) One-step ahead : MSE & tracé (5)
df_shift = df.shift(periods=1)
mse_one_step = mean_squared_error(googl_test['Close'], df_shift['Close'].iloc[248:])
mse_one_step

fig, ax = plt.subplots()
ax.plot(df['Close'], 'b-', label='actual')
ax.plot(df_shift['Close'].iloc[248:], 'r-.', label='forecast')
ax.axvspan(248, 252, color='#808080', alpha=0.2)
ax.legend(loc='best')
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
plt.xlim(240, 252)
plt.ylim(2200, 2400)
plt.tight_layout()



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  EXERCICE 3 — S&P 500 (^GSPC) : RW vs non-RW, diff, ACF                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
 
# Forecast the daily closing price of a stock of your choice
# The historical daily closing price of many stocks is available for free on finance.yahoo
# .com. Select a stock ticker of your choice, and download its historical daily closing
# price for 1 year:
    
# 1 Plot the daily closing price of your chosen stock.
# 2 Determine if it is a random walk or not.
# 3 If it is not a random walk, explain why.
# 4 Keep the last 5 days of data as a test set. The rest will be the train set.
# 5 Forecast the last 5 days using naive forecasting methods, and measure the MSE.
#   Which method is the best?
# 6 Plot your forecasts.
# 7 Forecast the next timestep over the test set, and measure the MSE. Did it
#   decrease?
# 8 Plot your forecasts.

# ── A) Téléchargement des données (1 an, quotidien)
import yfinance as yf
import pandas as pd

# Télécharger les données hebdomadaires du S&P500
df = yf.download("^GSPC", period="1y", interval="1d")  # dernière 5 années
# df.to_csv("gspc_weekly.csv")
print(df.head())

# ── B) Tracé du cours de clôture
plt.figure(dpi=300)  # 150 ou 200 pour plus de netteté
plt.plot(df['Close'])
plt.title(' gspc_weekly(close)')
plt.show()

# ── C) Est-ce une marche aléatoire ? Décomposition + stats glissantes + ACF
# Une serie temporelle est une marchealeatoire si sa premiere diference est stationnaire et 
# les autocorrelations serielles ne sont pas significatives

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_acf

# (2) Décomposition saisonnière (approximative)
decomposition = seasonal_decompose(df['Close'], model='additive', period=30)  # 30 ≈ un mois
fig = decomposition.plot()
fig.set_size_inches(10, 8)
fig.suptitle("Décomposition de la série S&P 500", fontsize=14)
plt.tight_layout()
plt.show()

# (3) Moyenne / écart-type / variance glissantes
rolling_window = 20  # 20 jours ≈ 1 mois de bourse
df['rolling_mean'] = df['Close'].rolling(window=rolling_window).mean()
df['rolling_std']  = df['Close'].rolling(window=rolling_window).std()
df['rolling_var']  = df['Close'].rolling(window=rolling_window).var()

fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ax[0].plot(df['Close'], label='Close', color='blue')
ax[0].plot(df['rolling_mean'], label='Moyenne glissante', color='red')
ax[0].set_ylabel('Prix / Moyenne'); ax[0].legend()
ax[1].plot(df['rolling_std'], label='Écart-type glissant', color='orange')
ax[1].set_ylabel('Écart-type'); ax[1].legend()
ax[2].plot(df['rolling_var'], label='Variance glissante', color='green')
ax[2].set_ylabel('Variance'); ax[2].legend()
plt.suptitle('Évolution moyenne / écart-type / variance', fontsize=14)
plt.tight_layout()
plt.show()

# (4) ACF sur le niveau
plt.figure(figsize=(8, 4))
plot_acf(df['Close'], lags=30)
plt.title("ACF - S&P 500")
plt.show()

# ── D) Différenciation d’ordre 1 + ADF + ACF sur Δ
if isinstance(df.columns, pd.MultiIndex):
    close = df['Close']['^GSPC']
else:
    close = df['Close']

close = close.astype(float).dropna()
d1=np.diff(close)

plt.figure(dpi=200)  # 150 ou 200 pour plus de netteté
plt.plot(d1)
plt.title(' gspc_weekly')
plt.show()

# TEST ADF sur la différence (stationnarité attendue si RW)
adf_stat_d1,pval_d1,*_=adfuller(d1)
print(f"ADF(Δ) stat={adf_stat_d1:.3f}, p-value={pval_d1:.4f}")  # p<0.05 ⇒ stationnaire

# ACF de la différence
plt.figure(figsize=(8, 4))
plot_acf(d1, lags=30)
plt.title("ACF - S&P 500")
plt.show()






















