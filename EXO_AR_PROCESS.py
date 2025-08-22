# =========================================
# 0) Imports & réglages
# =========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error






# ╔══════════════════════════════════════════════════════════════════════╗
# ║  A. Simulation d’un processus AR(2)                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np

np.random.seed(60)    
ma2 = np.array([1, 0, 0])
ar2 = np.array([1, -0.33, -0.50])

AR2_process = ArmaProcess(ar2, ma2).generate_sample(nsample=1000)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  B. Visualisation de la série simulée                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
# 2 Plot your simulated autoregressive process.
plt.figure(figsize=(8, 4))        
plt.plot(AR2_process) 
plt.title("AR2_process")   # Titre du graphique
plt.show()          # Affichage


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  C. Stationnarité : Test ADF                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝
# 3 Run the ADF test and check if the process is stationary. If not, apply differencing.

# TEST ADF — Hypothèses :
# H0 : la série n’est pas stationnaire
# H1 : la série est stationnaire
                  
adf_stat, pval, *_ = adfuller(AR2_process)  
print(f"ADF(Δ) stat={adf_stat:.3f}, p-value={pval:.4f}")

# Décision automatique (seuil 5 %)
if pval < 0.05:
    print("✅ La série est stationnaire (on rejette H0).")
else:
    print("❌ La série est non stationnaire (on ne rejette pas H0).")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  D. ACF : signature du processus                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
# 4 Plot the ACF. Is it slowly decaying?
plt.figure(figsize=(8, 4))        
plot_acf(AR2_process, lags=30)          # ACF jusqu'à 30 retards
plt.title("ACF - Foot trafic")          # Titre du graphique
plt.show()

# Interprétation (commentaire) :
# • Décroissance progressive (non abrupte) → signature d’un processus AR(p), pas MA(q).
# • La présence de coefficients significatifs confirme une dépendance auto-régressive.


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  E. PACF : identification de l’ordre AR                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
# 5 Plot the PACF. Are there significant coefficients after lag 2?
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
plot_pacf(AR2_process, lags=20, ax=ax)

ax.set_title("PACF d’un processus AR(2)", fontsize=18, fontweight="bold", pad=12)
ax.set_xlabel("Retards (lags)", fontsize=14)
ax.set_ylabel("PACF", fontsize=14)
ax.tick_params(labelsize=12)

plt.tight_layout()
plt.show()

# Interprétation (commentaire) :
# • Coefficients non significatifs au-delà du lag 2 → AR(2) plausible.


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  F. Partition train/test                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝
# 6 Separate your simulated series into train and test sets.
# Take the first 800 time steps for the train set and assign the rest to the test set.

train = AR2_process[:800]   # The training set is all the data except the last 800 data points. 
test  = AR2_process[800:]   # The test set is the last 52 data points.

print(len(AR2_process)) 
print(len(train))    
print(len(test))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  G. Visualisation : zone de test                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
# — Mise en contexte : on surligne la période de test (800 → N) —
train_len = 800
N = len(AR2_process)
test_len = N - train_len

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 8))
    
ax1.plot(AR2_process, color= '#d95f02' )
ax1.set_xlabel('Time')
ax1.set_ylabel('Avg. weekly foot traffic')
ax1.axvspan(train_len, N, color='#1b9e77', alpha=0.2)  # Surlignage de la zone test

# # --- Différenciée ---
# ax2.plot(np.arange(N-1), AR2_diff, color='#d95f02')
# ax2.set_xlabel('Time')
# ax2.set_ylabel('Diff. avg. weekly foot traffic')
# ax2.axvspan(train_len - 1, N - 1, color='#1b9e77', alpha=0.2)

ax2.plot(AR2_process, color= '#d95f02')
ax2.set_xlabel('Time')
ax2.set_ylabel('Diff. avg. weekly foot traffic')
ax2.axvspan(train_len, N, color='#1b9e77', alpha=0.2)  # Même surlignage (non différencié)

# Remarque (commentaire) :
# • Pour une série différenciée, la fenêtre test serait décalée d’1 (N-1).
# • Palette suggérée : '#1b9e77', '#d95f02', '#7570b3', '#e7298a'.

# plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2))
# fig.autofmt_xdate()
plt.tight_layout()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  H. Prévisions par fenêtre glissante (mean / last / AR(2))           ║
# ╚══════════════════════════════════════════════════════════════════════╝
# 7 Make forecasts over the test set. Use the historical mean method, last known
# value method, and an AR(2) model. Use the rolling_forecast function, and
# use a window length of 2.

# ==Forecasting an autoregressive process
# =================================================================================================================================
AR2_process = pd.DataFrame({'AR2_process': AR2_process})
AR2_process.head()

# — Définition de la fonction de rolling forecast —
from statsmodels.tsa.statespace.sarimax import SARIMAX
def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int,
                     window: int, method: str) -> list:
    total_len = train_len + horizon
    end_idx = train_len

    if method == 'mean':
        pred_mean = []
        for i in range(train_len, total_len, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))
        return pred_mean

    elif method == 'last':
        pred_last_value = []
        for i in range(train_len, total_len, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))
        return pred_last_value

    elif method == 'AR':
        pred_AR = []
        for i in range(train_len, total_len, window):
            model = SARIMAX(df[:i], order=(2, 0, 0))  # AR(2) (inchangé)
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_AR.extend(oos_pred)
        return pred_AR


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  I. Paramétrage et calcul des prévisions                             ║
# ╚══════════════════════════════════════════════════════════════════════╝
# === Paramètres du rolling forecast ===
TRAIN_LEN = len(train)    
HORIZON   = len(test)    
WINDOW    = 2

# === Génération des prédictions (mean / last / AR) ===
pred_mean        = rolling_forecast(AR2_process, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_last_value  = rolling_forecast(AR2_process, TRAIN_LEN, HORIZON, WINDOW, 'last')   
pred_AR          = rolling_forecast(AR2_process, TRAIN_LEN, HORIZON, WINDOW, 'AR')

# === Injection des prédictions dans le DataFrame test ===
test= test.copy()
# test = pd.DataFrame({'AR2_process_test': test})

test['pred_mean']        = pred_mean               
test['pred_last_value']  = pred_last_value   
test['pred_AR']          = pred_AR                   
test.head()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  J. Visualisation des prévisions vs. série réelle                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
# 8 Plot your forecasts.

# === Visualisation : réel (diff) vs prédictions ===
fig, ax = plt.subplots()
ax.plot(train)    
ax.plot(test['AR2_process'], label='actual')    
ax.plot(test['pred_mean'], 'g:', label='mean')    
ax.plot(test['pred_last_value'], 'y-.', label='last')    
ax.plot(test['pred_AR'], 'k--', label='AR(3)')

ax.set_xlabel('Timesteps')
ax.set_ylabel('Valeur')
ax.legend(loc='best')

ax.axvspan(train_len, N, color='#808080', alpha=0.2)   # mise en évidence de la fenêtre test
plt.xlim(700, 1000)
# plt.xticks([936, 988], [2018, 2019])
# fig.autofmt_xdate()
plt.tight_layout()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  K. Évaluation hors-échantillon (MSE)                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
# 9 Measure the MSE, and identify your champion model.

# === Évaluation : MSE des trois méthodes ===
from sklearn.metrics import mean_squared_error

mse_mean = mean_squared_error(test['AR2_process'], test['pred_mean'])
mse_last = mean_squared_error(test['AR2_process'], test['pred_last_value'])
mse_AR   = mean_squared_error(test['AR2_process'], test['pred_AR'])
print(mse_mean, mse_last, mse_AR)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  K. Évaluation hors-échantillon (MSE)                                ║
# ╚══════════════════════════════════════════════════════════════════════╝


# 10 Plot your MSEs in a bar plot.

fig, ax = plt.subplots()
x = ['mean', 'last_value', 'AR(2)']
y = [mse_mean, mse_last, mse_AR]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Methods')
ax.set_ylabel('MSE')
ax.set_ylim(0, 4)

for index, value in enumerate(y):
    plt.text(x=index, y=value+0.25, s=str(round(value, 2)), ha='center')

plt.tight_layout()