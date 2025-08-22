
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PARTIE A — MODELING AN AUTOREGRESSIVE PROCESS                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝



#=======Modeling an autoregressive process=====================================================================

df= pd.read_csv(r"C:\Users\mbarg\Downloads\foot_traffic.csv")
df.head()
print(len(df))

plt.figure(figsize=(8, 4))        
plt.plot(df["foot_traffic"]) # ACF jusqu'à 30 retards
plt.title("foot_traffic")   # Titre du graphique
plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2))
fig.autofmt_xdate()
plt.show()          # Affichage

.x
                  # TEST ADF                  
                  # Ho= La serie n est pas stationnaire 
                  # H1= La serie est stationnaire
                  
                  
adf_stat, pval, *_ = adfuller(df)  
print(f"ADF(Δ) stat={adf_stat:.3f}, p-value={pval:.4f}")
# Auto-décision
if pval < 0.05:
    print("✅ La série est stationnaire (on rejette H0).")
else:
    print("❌ La série est non stationnaire (on ne rejette pas H0).")
    
    #-------------On applique la premiere difference vu que le processus est non stationnaire 
    
    
foot_traffic_diff = np.diff(df['foot_traffic'], n=1)

plt.figure(figsize=(8, 4))        
plt.plot(foot_traffic_diff) # ACF jusqu'à 30 retards
plt.title("foot_traffic_diff")   # Titre du graphique
plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2))
fig.autofmt_xdate()
plt.show()          # Affichage



adf_stat, pval, *_ = adfuller(foot_traffic_diff)  
print(f"ADF(Δ) stat={adf_stat:.3f}, p-value={pval:.4f}")
# Auto-décision
if pval < 0.05:
    print("✅ La série est stationnaire (on rejette H0).")
else:
    print("❌ La série est non stationnaire (on ne rejette pas H0).")

    
plt.figure(figsize=(8, 4))        
plot_acf(foot_traffic_diff, lags=30)          # ACF jusqu'à 30 retards
plt.title("ACF - Foot trafic")           # Titre du graphique
plt.show()   

                                  
# ACF plot of the differenced average weekly foot traffic at a retail store. Notice how the plot is slowly decaying. This is a behavior that we have not observed before, and it is indicative of an autoregressive process.
   
# Therefore, there is no lag at which the coefficients abruptly become non
# significant. This means that we do not have a moving average process and that we are likely studying an autoregressive process.   

# When the ACF plot of a stationary process exhibits a pattern of exponential decay, we probably have an autoregressive process in play, and we must find another way to identify the order p of the AR(p) process. Specifically, we must turn our attention to the partial autocorrelation function (PACF) plot. 










# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PARTIE B — THE PARTIAL AUTOCORRELATION FUNCTION (PACF)                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝




#=========================The partial autocorrelation function (PACF)==================================================================



import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_pacf

# AR(2): phi1=0.33, phi2=0.50  → polynôme AR = [1, -phi1, -phi2]
ar2 = np.array([1, -0.33, -0.50])
ma2 = np.array([1, 0, 0])  # pas de MA

AR2_process = ArmaProcess(ar2, ma2).generate_sample(nsample=1000)

# --- Tracé PACF ---
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
plot_pacf(AR2_process, lags=20, ax=ax)

ax.set_title("PACF d’un processus AR(2) simulé", fontsize=18, fontweight="bold", pad=12)
ax.set_xlabel("Retards (lags)", fontsize=14)
ax.set_ylabel("PACF", fontsize=14)
ax.tick_params(labelsize=12)

plt.tight_layout()
plt.show()

# Plot of the PACF for our simulated AR(2) process. You can clearly see here that after lag 2, the partial autocorrelation coefficients are not significantly different from 0. Therefore, we can identify the order of a stationary 
# AR(p) model using the PACF plot.




# --- Tracé PACF weekly_foot ---

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
plot_pacf(foot_traffic_diff, lags=20, ax=ax)
ax.set_title("PACF weekly_foot", fontsize=18, fontweight="bold", pad=12)
ax.set_xlabel("Retards (lags)", fontsize=14)
ax.set_ylabel("PACF", fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.show()

# Looking at figure 5.8, you can see that there are no significant coefficients after lag 3. Therefore, the differenced average weekly foot traffic is an autoregressive process of order 3, which can also be denoted as AR(3).






# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PARTIE C — FORECASTING AN AUTOREGRESSIVE PROCESS (Split + Viz)             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝



#=========================Forecasting an autoregressive process

df_diff = pd.DataFrame({'foot_traffic_diff': foot_traffic_diff})

train = df_diff[:-52]   #The training set is all the data except the last 52 data points. 
test = df_diff[-52:] #The test set is the last 52 data points.

print(len(df_diff)) 
print(len(train))    
print(len(test))


# You can see that our training set contains 947 data points, while the test set contains 52 data points as expected. Note that the sum of both sets gives 999, which is one less data point than our original series. This is normal, since we applied differencing to make the series stationary, and we know that differencing removes the first data point from the series.

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, 
figsize=(10, 8))    
ax1.plot(df['foot_traffic'], color= '#d95f02' )
ax1.set_xlabel('Time')
ax1.set_ylabel('Avg. weekly foot traffic')
ax1.axvspan(948, 1000, color='#1b9e77', alpha=0.2) #colore en gris la zone test de l’axe X entre 948 et 1000 (valeurs numériques d’abscisse).
ax2.plot(df_diff['foot_traffic_diff'], color= '#d95f02')
ax2.set_xlabel('Time')
ax2.set_ylabel('Diff. avg. weekly foot traffic')
ax2.axvspan(947, 999, color='#1b9e77', alpha=0.2) #Zone grise de 947 à 999 (décalée d’1 par rapport au haut).Pourquoi décalée ? Une différence enlève la première observation ⇒ la série diff a une longueur de moins (N−1). Pour garder la même période réelle en test, on décale d’un cran vers la gauche.-------------- Autre couleurs '#1b9e77','#d95f02','#7570b3','#e7298a'
plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2))
fig.autofmt_xdate()
plt.tight_layout()











# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PARTIE D — FORECASTING (Rolling Forecast & Évaluation)                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝



#==Forecasting an autoregressive process
#=================================================================================================================================

#________________________________________________________________________________________________________
from statsmodels.tsa.statespace.sarimax import SARIMAX
def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int,
                     window: int, method: str) -> list:
    total_len = train_len + horizon

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
            model = SARIMAX(df[:i], order=(3, 0, 0))  # AR(3) (inchangé)
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_AR.extend(oos_pred)
        return pred_AR
    
    
    
       
#________________________________________________________________________________________________________

# === Paramètres du rolling forecast ===
TRAIN_LEN = len(train)    
HORIZON   = len(test)    
WINDOW    = 1

# === Génération des prédictions (mean / last / AR) ===
pred_mean        = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_last_value  = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'last')   
pred_AR          = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'AR')

# === Injection des prédictions dans le DataFrame test ===
test['pred_mean']        = pred_mean               
test['pred_last_value']  = pred_last_value   
test['pred_AR']          = pred_AR                   
test.head()




# === Visualisation : réel (diff) vs prédictions ===
fig, ax = plt.subplots()
ax.plot(df_diff['foot_traffic_diff'])    
ax.plot(test['foot_traffic_diff'], 'b-', label='actual')    
ax.plot(test['pred_mean'],        'g:',  label='mean')    
ax.plot(test['pred_last_value'],  'y-.', label='last')    
ax.plot(test['pred_AR'],          'k--', label='AR(3)')

ax.legend(loc=2)
ax.set_xlabel('Time')
ax.set_ylabel('Diff. avg. weekly foot traffic')

ax.axvspan(947, 998, color='#808080', alpha=0.2)
ax.set_xlim(920, 999)    
plt.xticks([936, 988], [2018, 2019])
fig.autofmt_xdate()
plt.tight_layout()

# === Évaluation : MSE des trois méthodes ===
from sklearn.metrics import mean_squared_error

mse_mean = mean_squared_error(test['foot_traffic_diff'], test['pred_mean'])
mse_last = mean_squared_error(test['foot_traffic_diff'], test['pred_last_value'])
mse_AR   = mean_squared_error(test['foot_traffic_diff'], test['pred_AR'])
print(mse_mean, mse_last, mse_AR)






# Since the MSE for the AR(3) model is
#  the lowest of the three, we conclude that the AR(3) model is the best-performing
#  method for forecasting next week’s average foot traffic. 

#This is expected, since we established that our stationary process was a third-order autoregressive process. 

# It makes sense that modeling using an AR(3) model will yield the best predictions.


#  Since our forecasts are differenced values, we need to reverse the transformation
#  in order to bring our forecasts back to the original scale of the data; otherwise, our pre
# dictions will not make sense in a business context. To do this, we can take the cumula
# tive sum of our predictions and add it to the last value of our training set in the
#  original series. This point occurs at index 948, since we are forecasting the last 52
#  weeks in a dataset containing 1,000 points.







# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PARTIE E — INVERSE TRANSFORM (Retour à l’échelle d’origine)                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝



df['pred_foot_traffic'] = pd.Series()

df['pred_foot_traffic'][948:] = df['foot_traffic'].iloc[948] + test['pred_AR'].cumsum()


fig, ax = plt.subplots()
ax.plot(df['foot_traffic'])
ax.plot(df['foot_traffic'], 'b-', label='actual')    
ax.plot(df['pred_foot_traffic'], 'k--', label='AR(3)')   
ax.legend(loc=2)
ax.set_xlabel('Time')
ax.set_ylabel('Average weekly foot traffic')
ax.axvspan(948, 1000, color='#808080', alpha=0.2)
ax.set_xlim(920, 1000)
ax.set_ylim(650, 770)
plt.xticks([936, 988],[2018, 2019])
fig.autofmt_xdate()
plt.tight_layout()


# you can see that our model (shown as a dashed line) follows the general
#  trend of the observed values in the test set.

from sklearn.metrics import mean_absolute_error

mae_AR_undiff = mean_absolute_error(df['foot_traffic'][948:], 
df['pred_foot_traffic'][948:])

print(mae_AR_undiff)

# This prints out a mean absolute error of 3.45. This means that our predictions are off by 3.45 people on average, either above or below the actual value for the week’s foot traffic.












