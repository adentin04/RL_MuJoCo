import numpy as np
import matplotlib.pyplot as plt # per fare un grafico

# --- PARAMETRI ---
prob_vittoria = 0.2
ricompensa_vittoria = 10
numero_giocate = 1000

# --- SIMULAZIONE ---
risultati = [] # Qui salvo la vincita di ogni giocata

for giocata in range(numero_giocate):
    # Genero un numero casuale tra 0 e 1
    tiro = np.random.random()

    if tiro < prob_vittoria:
        # Ho vinto
        risultati.append(ricompensa_vittoria)
    else:
        # Ho perso
        risultati.append(0)

# --- ANALISI (Statistica) ---
risultati = np.array(risultati)
media_vincite = np.mean(risultati)
print(f"Dopo {numero_giocate} giocate, la mia vincita media è stata: {media_vincite:.2f} euro")

# Facciamo una verifica: la media delle prime 10 giocate
print(f"Le prime 10 vincite: {risultati[:10]}")

# --- CONCETTI CHIAVE ---
# 1. Il Valore Atteso (calcolato a mano) era 2.0. La Media simulata ci si avvicina.
# 2. Abbiamo usato un ciclo for (Python), un array e una media (NumPy),
#    e una variabile casuale (Probabilità).
# 3. Se dovessi creare un agente RL, il suo scopo sarebbe di *imparare*
#    che tirare la leva è una buona idea (valore atteso positivo > 0).