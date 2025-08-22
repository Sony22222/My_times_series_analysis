# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1) MA PROCESS — Widget sales (chargement, ADF, ACF, prévisions)     ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ── 1.1 Chargement des données ─────────────────────────────────────────
import pandas as pd 
df= pd.read_csv(r"C:\Users\mbarg\Downloads\widget_sales.csv")
df.head()


# ── 1.2 Tracé de la série brute ────────────────────────────────────────
import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi=300)  # 300 dpi pour une meilleure qualité

ax.plot(df['widget_sales'])
ax.set_xlabel('Time')
ax.set_ylabel('Widget sales (k$)')

plt.xticks(
    [0,30,57,87,116,145,175,204,234,264,293,323,352,382,409,439,468,498],
    ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct',
     'Nov','Dec','2020','Feb','Mar','Apr','May','Jun']
)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()


# ── 1.3 Test ADF (stationnarité) ───────────────────────────────────────
#### STATIONARITY_TEST

# "HO: La ts est non stationnaire
# "H1: La ts est stationnaire

from statsmodels.tsa.stattools import adfuller

# TEST ADF
adf_stat_d1, pval_d1, *_ = adfuller(df['widget_sales'])
print(f"ADF(Δ) stat={adf_stat_d1:.3f}, p-value={pval_d1:.4f}")
# Auto-décision
if pval_d1 < 0.05:
    print("✅ La série est stationnaire (on rejette H0).")
else:
    print("❌ La série est non stationnaire (on ne rejette pas H0).")
    
    
# ── 1.4 Différenciation d'ordre 1 + tracé ──────────────────────────────
###La serie n'est pas stationnaire on peut differencier pour stabiliser la tendance 

import numpy as np

widget_sales_diff = np.diff(df['widget_sales'], n=1)

###On peut visualiser la serie differencier 

fig, ax = plt.subplots(dpi=300)  # 300 dpi pour une meilleure qualité

ax.plot(widget_sales_diff)
ax.set_xlabel('Time')
ax.set_ylabel('Widget sales (k$)')

plt.xticks(
    [0,30,57,87,116,145,175,204,234,264,293,323,352,382,409,439,468,498],
    ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct',
     'Nov','Dec','2020','Feb','Mar','Apr','May','Jun'], rotation=45, ha='right'
)


# ── 1.5 Refaire le test ADF (sur la diff) ───────────────────────────────
#------------------On peut refaire un test ADF pour revoir si la serie est stationnaire                                                               --------------------------------------------------------------------------

# TEST ADF
adf_stat, pval, *_ = adfuller(widget_sales_diff)
print(f"ADF(Δ) stat={adf_stat:.3f}, p-value={pval:.4f}")
# Auto-décision
if pval < 0.05:
    print("✅ La série est stationnaire (on rejette H0).")
else:
    print("❌ La série est non stationnaire (on ne rejette pas H0).")
    
    
# ── 1.6 Identification MA(q) via ACF ────────────────────────────────────
#----------------------------------- La serie est stationnaire on peut regarder la fonction d'auto correlation serielle (ACF) pour lidentification dune marche aleatoire ou non                                                     ------------------------------------------------------------------------

from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize=(8, 4))        
plot_acf(widget_sales_diff, lags=30)          # ACF jusqu'à 30 retards
plt.title("ACF - S&P 500")                     # Titre du graphique
plt.show()                                     # Affichage

# Les coeficients sont significatifs jusqu au lag 2 apres quoi ils deviennent abruptement non significatif. Cela veut dire que nous avons un processus STATIONNAIRE MA(2) (d'ordre 2') le coeficient significatif au lag 20 est du propbablement a la chance. 


# ── 1.7 Train/Test split + visualisation ────────────────────────────────
#---------------------------Forecasting a moving average process

# ##ON CONSTITUT LES DONNEES DENTTRAINEMENT ET LES DONNEES TEST 

df_diff = pd.DataFrame({'widget_sales_diff': widget_sales_diff}) #ON PLACE LES DONNES DIFFERENCIEE DANS LE DATA FRAME
  
train = df_diff[:int(0.9*len(df_diff))] #The first 90% of the data goes in the training set.

test = df_diff[int(0.9*len(df_diff)):]  #The last 10% of the data goes  in the test set for prediction.

print(len(train))
print(len(test))

# We’ve printed out the size of the train and test sets to remind you of the data point
#  that we lose when we difference. The original dataset contained 500 data points, while
#  the differenced series contains a total of 499 data points, since we differenced once.We’ve printed out the size of the train and test sets to remind you of the data point
#  that we lose when we difference. The original dataset contained 500 data points, while
#  the differenced series contains a total of 499 data points, since we differenced once.

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)    
ax1.plot(df['widget_sales'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Widget sales (k$)')

ax1.axvspan(450, 500, color='#808080', alpha=0.2)
ax2.plot(df_diff['widget_sales_diff'])

ax2.set_xlabel('Time')
ax2.set_ylabel('Widget sales - diff (k$)')

ax2.axvspan(449, 498, color='#808080', alpha=0.2)
plt.xticks(
    [0, 30, 57, 87, 116, 145, 175, 204, 234, 264, 293, 323, 352, 382, 409, 
439, 468, 498], 
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 
'Nov', 'Dec', '2020', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
fig.autofmt_xdate()
plt.tight_layout()


# ── 1.8 Fonction de rolling forecast (mean/last/MA(2)) ─────────────────
# # First, we import the SARIMAX function from the statsmodels library. 
# # This function
# # will allow us to fit an MA(2) model to our differenced series. 

# Note that SARIMAX is a
# # complex model that allows us to consider seasonal effects, autoregressive processes,
# # non-stationary time series, moving average processes, and exogenous variables all in a
# # single model. For now, we will disregard all factors except the moving average portion.

from statsmodels.tsa.statespace.sarimax import SARIMAX

def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, 
                     window: int, method: str) -> list:
    """
    Effectue une prévision roulante (rolling forecast) sur une série temporelle.

    Paramètres
    ----------
    df : pd.DataFrame
        Série temporelle (ex : process MA simulé).
    train_len : int
        Longueur de l’échantillon d’apprentissage (par ex. 800 points).
    horizon : int
        Nombre total de pas à prévoir (par ex. 200 points).
    window : int
        Taille de la fenêtre de prévision à chaque itération (ex : 2 → on prévoit 2 pas à la fois).
    method : str
        Méthode de prévision :
            - 'mean' → prédit la moyenne de l’échantillon d’apprentissage
            - 'last' → prédit la dernière valeur observée
            - 'MA'   → ajuste un modèle MA(2) avec SARIMAX

    Retour
    ------
    list
        Liste des prédictions sur l’horizon spécifié.
    """

    total_len = train_len + horizon  # longueur totale = train + test

    # ----- Méthode 1 : baseline → prévision = moyenne -----
    if method == 'mean':
        pred_mean = []
        for i in range(train_len, total_len, window):
            # Calcul de la moyenne de la série observée jusqu’au temps i
            mean = np.mean(df[:i].values)
            # On répète cette moyenne "window" fois (par ex. 2 pas)
            pred_mean.extend(mean for _ in range(window))
        return pred_mean

    # ----- Méthode 2 : baseline → prévision = dernière valeur observée -----
    elif method == 'last':
        pred_last_value = []
        for i in range(train_len, total_len, window):
            # Dernière valeur observée jusqu’au temps i
            last_value = df[:i].iloc[-1].values[0]
            # On répète cette valeur "window" fois
            pred_last_value.extend(last_value for _ in range(window))
        return pred_last_value

    # ----- Méthode 3 : modèle MA(2) via SARIMAX -----
    elif method == 'MA':
        pred_MA = []
        for i in range(train_len, total_len, window):
            # Ajuste un modèle MA(2) sur toutes les données disponibles jusqu’au temps i
            model = SARIMAX(df[:i], order=(0,0,2))  # (AR=0, differencing=0, MA=2)
            res = model.fit(disp=False)

            # Obtenir les prévisions jusqu’au pas i+window-1
            predictions = res.get_prediction(0, i + window - 1)

            # On garde uniquement les "window" dernières prédictions (out-of-sample)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_MA.extend(oos_pred)
        return pred_MA
# Once it’s defined, we can use our function and forecast using three methods: the his
# torical mean, the last value, and the fitted MA(2) model.


# ── 1.9 Génération des prévisions (mean/last/MA) ───────────────────────
# On part du jeu de test (les 10% de fin de la série différenciée)
# On crée une copie afin de ne pas modifier l'original
pred_df = test.copy()

# --- Paramètres de la prévision ---
TRAIN_LEN = len(train)     # Nombre d'observations utilisées pour entraîner (90% de la série)
HORIZON   = len(test)      # Nombre total de points à prévoir (10% de la série)
WINDOW    = 2              # Nombre de pas prévus à chaque itération (rolling forecast par bloc de 2)

# --- Génération des prévisions selon 3 méthodes différentes ---

# 1. Baseline "mean" : prédit la moyenne des données observées
pred_mean = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'mean')

# 2. Baseline "last" : prédit toujours la dernière valeur observée
pred_last_value = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'last')

# 3. Modèle MA(2) : ajuste un modèle de moyenne mobile d'ordre 2
#    à chaque itération pour prévoir
pred_MA = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'MA')

# --- Ajout des prédictions dans le DataFrame du test ---
# Chaque nouvelle colonne contient les prévisions correspondantes
pred_df['pred_mean'] = pred_mean
pred_df['pred_last_value'] = pred_last_value
pred_df['pred_MA'] = pred_MA
pred_df['train'] = train

# --- Afficher les 5 premières lignes pour vérifier ---
pred_df.head()


# ── 1.10 Graphique de comparaison (diff) ───────────────────────────────
######### GARDONS A L'ESPRIT QUE NOUS TRAVAILLONS TOUJOURS AVEC LES VARIABLES DIFFERENCIEES DONC NOS PREDICTIONS SONT AUSSI DES VALEURS DIFFERENCIEES...  

# ####### Graphique de comparaison 

# On peut reinitialiser les index afin de mettre les graphique sur le meme axe

# Création de la figure
plt.figure(figsize=(12,6))

# --- Valeurs réelles (test set) ---
plt.plot(pred_df.index, pred_df['widget_sales_diff'], 
         label='Valeurs réelles', color='black', linewidth=2)

# --- Prédictions baseline : moyenne ---
plt.plot(pred_df.index, pred_df['pred_mean'], 
         label='Prédiction (Mean)', linestyle='--', color='blue')

# --- Prédictions baseline : dernière valeur ---
plt.plot(pred_df.index, pred_df['pred_last_value'], 
         label='Prédiction (Last Value)', linestyle='--', color='green')

# --- Prédictions avec modèle MA(2) ---
plt.plot(pred_df.index, pred_df['pred_MA'], 
         label='Prédiction (MA(2))', linestyle='--', color='red')

# --- Mise en forme du graphique ---
plt.title("Comparaison des prévisions sur le jeu de test", fontsize=14)
plt.xlabel("Temps")
plt.ylabel("Widget sales - différencié (k$)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# you’ll notice that the prediction coming from the historical mean, shown as a dotted line, is almost a straight line. This is expected; the process is stationary, so the historical mean should be stable over time. 

#we will use the mean_squared_errorfunction from the sklearn package. We simply need to pass the observed values andthe predicted values into the function.


# ── 1.11 MSE + bar chart (diff) ────────────────────────────────────────
from sklearn.metrics import mean_squared_error

mse_mean = mean_squared_error(pred_df['widget_sales_diff'], pred_df['pred_mean'])
mse_last = mean_squared_error(pred_df['widget_sales_diff'], pred_df['pred_last_value'])
mse_MA = mean_squared_error(pred_df['widget_sales_diff'], pred_df['pred_MA'])
print(mse_mean, mse_last, mse_MA)

# REPRESENTATION GRAPHIQUE 
methods = ['Mean', 'Last Value', 'MA(2)']
mse_values = [mse_mean, mse_last, mse_MA]

# --- Tracé en barres ---
plt.figure(figsize=(8,5))
bars = plt.bar(methods, mse_values, color=['blue', 'green', 'red'], alpha=0.7)
# Ajouter les valeurs au-dessus de chaque barre
for bar, val in zip(bars, mse_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f"{val:.3f}", ha='center', va='bottom', fontsize=10)
    
# --- Mise en forme ---
plt.title("Comparaison des erreurs MSE par méthode", fontsize=14)
plt.ylabel("MSE")
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# ── 1.12 Transformation inverse (cumul) + tracé sur l’échelle d’origine ─
#---------------- Transformation Inverse de la prediction : La somme cumulative
# Initialiser la colonne avec NaN
df['pred_widget_sales'] = np.nan  
# Dernière valeur observée avant la prédiction (point de départ du cumul)
last_obs = df['widget_sales'].iloc[450]

# Transformation inverse des différences (cumul)
df.loc[450:, 'pred_widget_sales'] = last_obs + pred_df['pred_MA'].cumsum().values
#---------------------------------------------------------------

#---------------- Figure de prediction

fig, ax = plt.subplots()
ax.plot(df['widget_sales'], 'b-', label='actual')    
ax.plot(df['pred_widget_sales'], 'k--', label='MA(2)')   
ax.legend(loc=2)
ax.set_xlabel('Time')
ax.set_ylabel('Widget sales (K$)')
ax.axvspan(450, 500, color='#808080', alpha=0.2)
ax.set_xlim(400, 500)
plt.xticks(
    [409, 439, 468, 498], 
    ['Mar', 'Apr', 'May', 'Jun'])
fig.autofmt_xdate()
plt.tight_layout()

#-------------------------------------------------------

#-----------------------------------------------------------------

from sklearn.metrics import mean_absolute_error
mae_MA_undiff = mean_absolute_error(df['widget_sales'].iloc[450:], 
df['pred_widget_sales'].iloc[450:])
print(mae_MA_undiff)

#----2.324*1000= 2320 --------------------------En moyenne, les prédictions du modèle s’écartent de 2 320 $ (au-dessus ou en dessous) par rapport aux valeurs réelles. Concrètement, ça veut dire que si le modèle prévoit les ventes de widgets pour la semaine prochaine, il y a de fortes chances que sa prévision soit à ± 2 320 $ de la valeur réelle.
#-------------------------------------------------------------------------



# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2) EXERCISE — Simuler MA(2) et prévoir (comparaison)                ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ── 2.1 Simulation MA(2) ───────────────────────────────────────────────
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np

np.random.seed(42)    
ma2 = np.array([1, 0.9, 0.3])
ar2 = np.array([1, 0, 0])
MA2_process = ArmaProcess(ar2, ma2).generate_sample(nsample=1000)
MA2_process


# ── 2.2 Tracé de la série simulée ──────────────────────────────────────
import matplotlib.pyplot as plt

fig, ax = plt.subplots()  # 300 dpi pour une meilleure qualité

ax.plot(MA2_process)
ax.set_xlabel('Time')
ax.set_ylabel('MA_Process')
fig.autofmt_xdate()
plt.tight_layout()
plt.show()


# ── 2.3 Test ADF (stationnarité) ───────────────────────────────────────
#-----Run the ADF test, and check if the process is stationary                ---------------------------------------------------------------------------


# "HO: La ts est non stationnaire
# "H1: La ts est stationnaire

from statsmodels.tsa.stattools import adfuller

# TEST ADF
adf_stat_d1, pval_d1, *_ = adfuller(MA2_process)
print(f"ADF(Δ) stat={adf_stat_d1:.3f}, p-value={pval_d1:.4f}")
# Auto-décision
if pval_d1 < 0.05:
    print("✅ La série est stationnaire (on rejette H0).")
else:
    print("❌ La série est non stationnaire (on ne rejette pas H0).")
      
# La serie semble stationnaire pas besoin de differencier


# ── 2.4 ACF (signature MA(q)) ──────────────────────────────────────────
#----Plot the ACF, and see if there are significant coefficients after lag 2.                                                                          --------------------------------------------------------------------------
from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize=(8, 4))        
plot_acf(MA2_process, lags=30)          # ACF jusqu'à 30 retards
plt.title("ACF - S&P 500")                     # Titre du graphique
plt.show()                                     # Affichage 

#----- L'ACF presente des coeficients non significatif apres le lag 2


# ── 2.5 Split 80/20 ────────────────────────────────────────────────────
#----Separate your simulated series into train and test sets. Take the first 800 time steps for the train set, and assign the rest to the test set.                                                                        --------------------------------------------------------------------------
print(len(MA2_process))

# ##ON CONSTITUT LES DONNEES DENTTRAINEMENT ET LES DONNEES TEST 
       
train = MA2_process[:int(0.8*len(MA2_process))] #The first 90% of the data goes in the training set.

test = MA2_process[int(0.8*len(MA2_process)):]  #The last 10% of the data goes  in the test set for prediction.

print(len(train))
print(len(test))


# ── 2.6 Rolling forecast (mean/last/MA(2)) ─────────────────────────────
#-----------------Make forecasts over the test set. Use the mean, last value, and an MA(2) model. Make sure you repeatedly forecast 2 timesteps at a time using the recursive_forecast function we defined.                                                         ----------------------------------------------------------------------------------------

# # First, we import the SARIMAX function from the statsmodels library. 
# # This function
# # will allow us to fit an MA(2) model to our differenced series. 

# Note that SARIMAX is a
# # complex model that allows us to consider seasonal effects, autoregressive processes,
# # non-stationary time series, moving average processes, and exogenous variables all in a
# # single model. For now, we will disregard all factors except the moving average portion.

from statsmodels.tsa.statespace.sarimax import SARIMAX

MA2_process= pd.to_dataframe()

# Convertion en dataframe

df= pd.DataFrame({'MA2_process': MA2_process})

def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, 
                     window: int, method: str) -> list:
    """
    Effectue une prévision roulante (rolling forecast) sur une série temporelle.

    Paramètres
    ----------
    df : pd.DataFrame
        Série temporelle (ex : process MA simulé).
    train_len : int
        Longueur de l’échantillon d’apprentissage (par ex. 800 points).
    horizon : int
        Nombre total de pas à prévoir (par ex. 200 points).
    window : int
        Taille de la fenêtre de prévision à chaque itération (ex : 2 → on prévoit 2 pas à la fois).
    method : str
        Méthode de prévision :
            - 'mean' → prédit la moyenne de l’échantillon d’apprentissage
            - 'last' → prédit la dernière valeur observée
            - 'MA'   → ajuste un modèle MA(2) avec SARIMAX

    Retour
    ------
    list
        Liste des prédictions sur l’horizon spécifié.
    """

    total_len = train_len + horizon  # longueur totale = train + test

    # ----- Méthode 1 : baseline → prévision = moyenne -----
    if method == 'mean':
        pred_mean = []
        for i in range(train_len, total_len, window):
            # Calcul de la moyenne de la série observée jusqu’au temps i
            mean = np.mean(df[:i].values)
            # On répète cette moyenne "window" fois (par ex. 2 pas)
            pred_mean.extend(mean for _ in range(window))
        return pred_mean

    # ----- Méthode 2 : baseline → prévision = dernière valeur observée -----
    elif method == 'last':
        pred_last_value = []
        for i in range(train_len, total_len, window):
            # Dernière valeur observée jusqu’au temps i
            last_value = df[:i].iloc[-1].values[0]
            # On répète cette valeur "window" fois
            pred_last_value.extend(last_value for _ in range(window))
        return pred_last_value

    # ----- Méthode 3 : modèle MA(2) via SARIMAX -----
    elif method == 'MA':
        pred_MA = []
        for i in range(train_len, total_len, window):
            # Ajuste un modèle MA(2) sur toutes les données disponibles jusqu’au temps i
            model = SARIMAX(df[:i], order=(0,0,2))  # (AR=0, differencing=0, MA=2)
            res = model.fit(disp=False)

            # Obtenir les prévisions jusqu’au pas i+window-1
            predictions = res.get_prediction(0, i + window - 1)

            # On garde uniquement les "window" dernières prédictions (out-of-sample)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_MA.extend(oos_pred)
        return pred_MA
# Once it’s defined, we can use our function and forecast using three methods: the his
# torical mean, the last value, and the fitted MA(2) model.


# ── 2.7 Prévisions + DataFrame de sortie ───────────────────────────────
# On part du jeu de test (les 20% de fin de la série différenciée) On crée une copie en dataframe afin de ne pas modifier l'original 


test= pd.DataFrame({'Test_data' : test})
pred_df = test.copy()

# --- Paramètres de la prévision ---
TRAIN_LEN = len(train)     # Nombre d'observations utilisées pour entraîner (80% de la série)
HORIZON   = len(test)      # Nombre total de points à prévoir (20% de la série)
WINDOW    = 2              # Nombre de pas prévus à chaque itération (rolling forecast par bloc de 2)

# --- Génération des prévisions selon 3 méthodes différentes ---

# 1. Baseline "mean" : prédit la moyenne des données observées
pred_mean = rolling_forecast(df, TRAIN_LEN, HORIZON, WINDOW, 'mean')

# 2. Baseline "last" : prédit toujours la dernière valeur observée
pred_last_value = rolling_forecast(df, TRAIN_LEN, HORIZON, WINDOW, 'last')

# 3. Modèle MA(2) : ajuste un modèle de moyenne mobile d'ordre 2
#    à chaque itération pour prévoir
pred_MA = rolling_forecast(df, TRAIN_LEN, HORIZON, WINDOW, 'MA')

# --- Ajout des prédictions dans le DataFrame du test ---
# Chaque nouvelle colonne contient les prévisions correspondantes

pred_df['pred_mean'] = pred_mean
pred_df['pred_last_value'] = pred_last_value
pred_df['pred_MA'] = pred_MA
pred_df['train'] = train

# --- Afficher les 5 premières lignes pour vérifier ---
pred_df.head()


# ── 2.8 Graphique comparatif (MA2 simulé) ──────────────────────────────
#---------------------------- Graphique de comparaison 
#---------------------------------------------------------------------------------------

# Création de la figure
plt.figure(figsize=(12,6))
# --- Valeurs réelles (test set) ---
plt.plot(pred_df.index, pred_df['Test_data'], 
         label='Valeurs réelles', color='black', linewidth=2)
# --- Prédictions baseline : moyenne ---
plt.plot(pred_df.index, pred_df['pred_mean'], 
         label='Prédiction (Mean)', linestyle='--', color='blue')
# --- Prédictions baseline : dernière valeur ---
plt.plot(pred_df.index, pred_df['pred_last_value'], 
         label='Prédiction (Last Value)', linestyle='--', color='green')
# --- Prédictions avec modèle MA(2) ---
plt.plot(pred_df.index, pred_df['pred_MA'], 
         label='Prédiction (MA(2))', linestyle='--', color='red')
# --- Mise en forme du graphique ---
plt.title("Comparaison des prévisions sur le jeu de test", fontsize=14)
plt.xlabel("Temps")
plt.ylabel(" Serie")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ── 2.9 MSE + bar chart ────────────────────────────────────────────────
#---------------------On calcul et on compare les erreurs de prediction
print((pred_df))
from sklearn.metrics import mean_squared_error

mse_mean = mean_squared_error(pred_df['Test_data'], pred_df['pred_mean'])
mse_last = mean_squared_error(pred_df['Test_data'], pred_df['pred_last_value'])
mse_MA = mean_squared_error(pred_df['Test_data'], pred_df['pred_MA'])
print(mse_mean, mse_last, mse_MA)

#-------------------REPRESENTATION GRAPHIQUE =====================================================================================

methods = ['Mean', 'Last Value', 'MA(2)']
mse_values = [mse_mean, mse_last, mse_MA]

# --- Tracé en barres ---
plt.figure(figsize=(8,5))
bars = plt.bar(methods, mse_values, color=['blue', 'green', 'red'], alpha=0.7)
# Ajouter les valeurs au-dessus de chaque barre
for bar, val in zip(bars, mse_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f"{val:.3f}", ha='center', va='bottom', fontsize=10)
# --- Mise en forme ---
plt.title("Comparaison des erreurs MSE par méthode", fontsize=14)
plt.ylabel("MSE")
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
#######--------------Le modele MA(2) semble donc etre le modele champion de prediction 



# ╔══════════════════════════════════════════════════════════════════════╗
# ║  3) EXERCISE — Simuler MA(3) et prévoir (comparaison)                ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ── 3.1 Simulation MA(3) ───────────────────────────────────────────────
#==========================Simulate an MA(2) process and make forecasts                 ================================================================================================================

# Recreate the previous exercise, but simulate a moving average process of your choice.
# Try simulating a third-order or fourth-order moving average process. I recommend
# generating 10,000 samples. Be especially attentive to the ACF, and see if your coeffi
# cients become non-significant after lag q.

np.random.seed(70)    
ma3 = np.array([1, 0.9, 0.3, 0.9]) #---------Format ma = [1, θ1, θ2, θ3] est la convention statsmodels
ar3 = np.array([1, 0, 0, 0])
MA3_process = ArmaProcess(ar3, ma3).generate_sample(nsample=1000)     


# ── 3.2 Tracé MA(3) ───────────────────────────────────────────────────
#  A moving average process states that the present value is linearly dependent on
#  the mean, present error term, and past error terms. The error terms are nor
# mally distributed.
plt.figure(figsize=(8,4))
plt.plot(MA3_process)
plt.title("MA(3) simulé")
plt.tight_layout() 
plt.show()


# ── 3.3 ACF de MA(3) ──────────────────────────────────────────────────
#   You can identify the order q of a stationary moving average process by studying
#  the ACF plot. The coefficients are significant up until lag q only.
plt.figure(figsize=(8,4), dpi=300)
plot_acf(MA3_process, lags=20)
plt.title("ACF – MA(3) simulé")
plt.tight_layout()
plt.show()


# ── 3.4 Test ADF (stationnarité) ───────────────────────────────────────
#-----Run the ADF test, and check if the process is stationary                ---------------------------------------------------------------------------


# "HO: La ts est non stationnaire
# "H1: La ts est stationnaire

from statsmodels.tsa.stattools import adfuller

# TEST ADF
adf_stat_d1, pval_d1, *_ = adfuller(MA3_process)
print(f"ADF(Δ) stat={adf_stat_d1:.3f}, p-value={pval_d1:.4f}")
# Auto-décision
if pval_d1 < 0.05:
    print("✅ La série est stationnaire (on rejette H0).")
else:
    print("❌ La série est non stationnaire (on ne rejette pas H0).")


# ── 3.5 Split 80/20 ────────────────────────────────────────────────────
#----Separate your simulated series into train and test sets. Take the first 800 time steps for the train set, and assign the rest to the test set.                                                                        --------------------------------------------------------------------------
print(len(MA3_process))

# ##ON CONSTITUT LES DONNEES DENTTRAINEMENT ET LES DONNEES TEST 
       
train = MA3_process[:int(0.8*len(MA3_process))] #The first 90% of the data goes in the training set.

test = MA3_process[int(0.8*len(MA3_process)):]  #The last 10% of the data goes  in the test set for prediction.

print(len(train))
print(len(test))


# ── 3.6 Rolling forecast (mean/last/MA(2) pour comparaison) ────────────
#-----------------Make forecasts over the test set. Use the mean, last value, and an MA(2) model. Make sure you repeatedly forecast 2 timesteps at a time using the recursive_forecast function we defined.                                                         ----------------------------------------------------------------------------------------

# # First, we import the SARIMAX function from the statsmodels library. 
# # This function
# # will allow us to fit an MA(2) model to our differenced series. 

# Note that SARIMAX is a
# # complex model that allows us to consider seasonal effects, autoregressive processes,
# # non-stationary time series, moving average processes, and exogenous variables all in a
# # single model. For now, we will disregard all factors except the moving average portion.

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Convertion en dataframe

df= pd.DataFrame({'MA3_process': MA3_process})

def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, 
                     window: int, method: str) -> list:
    """
    Effectue une prévision roulante (rolling forecast) sur une série temporelle.

    Paramètres et logique identiques aux sections précédentes.
    Ici, on garde MA(2) pour l'illustration de la méthode.
    """

    total_len = train_len + horizon  # longueur totale = train + test

    # ----- Méthode 1 : baseline → prévision = moyenne -----
    if method == 'mean':
        pred_mean = []
        for i in range(train_len, total_len, window):
            # Calcul de la moyenne de la série observée jusqu’au temps i
            mean = np.mean(df[:i].values)
            # On répète cette moyenne "window" fois (par ex. 2 pas)
            pred_mean.extend(mean for _ in range(window))
        return pred_mean

    # ----- Méthode 2 : baseline → prévision = dernière valeur observée -----
    elif method == 'last':
        pred_last_value = []
        for i in range(train_len, total_len, window):
            # Dernière valeur observée jusqu’au temps i
            last_value = df[:i].iloc[-1].values[0]
            # On répète cette valeur "window" fois
            pred_last_value.extend(last_value for _ in range(window))
        return pred_last_value

    # ----- Méthode 3 : modèle MA(2) via SARIMAX -----
    elif method == 'MA':
        pred_MA = []
        for i in range(train_len, total_len, window):
            # Ajuste un modèle MA(2) sur toutes les données disponibles jusqu’au temps i
            model = SARIMAX(df[:i], order=(0,0,2))  # MA(2) choisi pour l’illustration
            res = model.fit(disp=False)

            # Obtenir les prévisions jusqu’au pas i+window-1
            predictions = res.get_prediction(0, i + window - 1)

            # On garde uniquement les "window" dernières prédictions (out-of-sample)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_MA.extend(oos_pred)
        return pred_MA
# Once it’s defined, we can use our function and forecast using three methods: the his
# torical mean, the last value, and the fitted MA(2) model.


# ── 3.7 Prévisions + DataFrame de sortie ───────────────────────────────
#------------------------- On part du jeu de test (les 20% de fin de la série différenciée) On crée une copie en dataframe afin de ne pas modifier l'original 


test= pd.DataFrame({'Test_data' : test})
pred_df = test.copy()

# --- Paramètres de la prévision ---
TRAIN_LEN = len(train)     # Nombre d'observations utilisées pour entraîner (80% de la série)
HORIZON   = len(test)      # Nombre total de points à prévoir (20% de la série)
WINDOW    = 2           # Nombre de pas prévus à chaque itération (rolling forecast par bloc de 2)

# --- Génération des prévisions selon 3 méthodes différentes ---

# 1. Baseline "mean" : prédit la moyenne des données observées
pred_mean = rolling_forecast(df, TRAIN_LEN, HORIZON, WINDOW, 'mean')

# 2. Baseline "last" : prédit toujours la dernière valeur observée
pred_last_value = rolling_forecast(df, TRAIN_LEN, HORIZON, WINDOW, 'last')

# 3. Modèle MA(2) : ajuste un modèle de moyenne mobile d'ordre 2
#    à chaque itération pour prévoir
pred_MA3 = rolling_forecast(df, TRAIN_LEN, HORIZON, WINDOW, 'MA')

# --- Ajout des prédictions dans le DataFrame du test ---
# Chaque nouvelle colonne contient les prévisions correspondantes

pred_df['pred_mean'] = pred_mean
pred_df['pred_last_value'] = pred_last_value
pred_df['pred_MA'] = pred_MA3

# --- Afficher les 5 premières lignes pour vérifier ---
pred_df.head()
len((df))
len((train))


# ── 3.8 Graphique comparatif (MA3 simulé) ──────────────────────────────
#---------------------------- Graphique de comparaison 
#---------------------------------------------------------------------------------------
 
plt.figure(figsize=(12,6))
# --- Valeurs réelles (test set) ---
plt.plot(pred_df.index, pred_df['Test_data'], 
         label='Valeurs réelles', color='black', linewidth=2)
# --- Prédictions baseline : moyenne ---
plt.plot(pred_df.index, pred_df['pred_mean'], 
         label='Prédiction (Mean)', linestyle='--', color='blue')
# --- Prédictions baseline : dernière valeur ---
plt.plot(pred_df.index, pred_df['pred_last_value'], 
         label='Prédiction (Last Value)', linestyle='--', color='green')
# --- Prédictions avec modèle MA(2) ---
plt.plot(pred_df.index, pred_df['pred_MA'], 
         label='Prédiction (MA(2))', linestyle='--', color='red')
# --- Mise en forme du graphique ---
plt.title("Comparaison des prévisions sur le jeu de test", fontsize=14)
plt.xlabel("Temps")
plt.ylabel(" Serie")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ── 3.9 MSE + bar chart ────────────────────────────────────────────────
#---------------------On calcul et on compare les erreurs de prediction
print((pred_df))
from sklearn.metrics import mean_squared_error

mse_mean = mean_squared_error(pred_df['Test_data'], pred_df['pred_mean'])
mse_last = mean_squared_error(pred_df['Test_data'], pred_df['pred_last_value'])
mse_MA   = mean_squared_error(pred_df['Test_data'], pred_df['pred_MA'])
print(mse_mean, mse_last, mse_MA)

#-------------------REPRESENTATION GRAPHIQUE 

methods = ['Mean', 'Last Value', 'MA(2)']
mse_values = [mse_mean, mse_last, mse_MA]

# --- Tracé en barres ---
plt.figure(figsize=(8,5))
bars = plt.bar(methods, mse_values, color=['blue', 'green', 'red'], alpha=0.7)
# Ajouter les valeurs au-dessus de chaque barre
for bar, val in zip(bars, mse_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f"{val:.3f}", ha='center', va='bottom', fontsize=10)
# --- Mise en forme ---
plt.title("Comparaison des erreurs MSE par méthode", fontsize=14)
plt.ylabel("MSE")
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
#######--------------Le modele MA(2) semble donc etre le modele champion de prediction 
