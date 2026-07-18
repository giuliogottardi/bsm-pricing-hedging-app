# BSM Pricing & Hedging Desk — Streamlit app

App interattiva basata sul notebook "BSM Pricing Engine & Dynamic Hedging Simulator"
(Giulio, LM-16 Financial Risk and Data Analysis, Sapienza Università di Roma).

## Contenuto

- `core.py` — logica pura (pricing BSM, Normal CDF custom, Greche numeriche,
  Monte Carlo delta-hedging, implied vol via bisezione). Nessun `input()`,
  nessun `print()`, nessun plotting: solo calcolo, così è riusabile e cache-abile.
- `app.py` — interfaccia Streamlit: sidebar con i parametri di scenario e 5 tab
  (Pricing & Greeks, Superfici 3D, Volatilità & IV surface, Hedging simulator,
  Come funziona).
- `requirements.txt` — dipendenze.

## Eseguire in locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

Si apre su `http://localhost:8501`.

## Deploy gratuito su Streamlit Community Cloud

1. Crea un repository GitHub (pubblico, o privato se hai Streamlit Cloud collegato
   al tuo account) e caricaci questi 3 file (`app.py`, `core.py`, `requirements.txt`).
2. Vai su https://share.streamlit.io, fai login con GitHub.
3. "New app" → seleziona il repo, il branch, e `app.py` come main file.
4. Deploy. Dopo ~1-2 minuti ottieni un URL pubblico tipo
   `https://<nome-app>.streamlit.app` da mettere nel CV/portfolio.

Nota: su Streamlit Cloud yfinance funziona normalmente (a differenza di questo
ambiente sandbox), quindi vedrai dati di mercato reali invece del fallback
sintetico.

## Deploy alternativo: Hugging Face Spaces

Stessa app, stesso `requirements.txt`: crea uno Space di tipo "Streamlit",
carica i file, e HF fa il build automaticamente. Utile se preferisci l'ecosistema
HF o vuoi affiancarla ad altri progetti ML che hai già lì.
