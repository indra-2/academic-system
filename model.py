import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

# ================= DATA =================
n_students = 300

data = pd.DataFrame({
    "sem1": np.random.uniform(5, 9, n_students),
    "sem2": np.random.uniform(5, 9, n_students),
    "sem3": np.random.uniform(5, 9, n_students),
    "sem4": np.random.uniform(5, 9, n_students),
    "math": np.random.uniform(40, 100, n_students),
    "physics": np.random.uniform(40, 100, n_students),
    "english": np.random.uniform(40, 100, n_students),
    "attendance": np.random.uniform(60, 100, n_students)
})

data["target"] = (
    data["sem4"] * 0.6 +
    data["attendance"] * 0.01 +
    np.random.normal(0, 0.3, n_students)
)

# ================= PREPROCESS =================
X = data[["sem1", "sem2", "sem3", "sem4"]].values
y = data["target"].values

scaler_seq = MinMaxScaler()
X_scaled = scaler_seq.fit_transform(X).reshape((n_students, 4, 1))

# ================= LSTM MODEL =================
def build_model():
    inp = Input(shape=(4,1))
    x = LSTM(64)(inp)
    x = Dropout(0.3)(x, training=True)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x, training=True)
    out = Dense(1)(x)

    model = Model(inp, out)
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model

model = build_model()
model.fit(X_scaled, y, epochs=30, batch_size=16, verbose=0)

# ================= MC DROPOUT =================
def mc_predict(X):
    preds = []
    for _ in range(30):
        preds.append(model(X, training=True).numpy())
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)

# ================= AUTOENCODER =================
sub = data[["math","physics","english"]].values
scaler_sub = MinMaxScaler()
sub_scaled = scaler_sub.fit_transform(sub)

inp = Input(shape=(3,))
enc = Dense(6, activation='relu')(inp)
enc = Dense(3, activation='relu')(enc)
dec = Dense(6, activation='relu')(enc)
dec = Dense(3, activation='sigmoid')(dec)

autoencoder = Model(inp, dec)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(sub_scaled, sub_scaled, epochs=30, verbose=0)

# ================= HELPERS =================
def weak_subjects(scores):
    s = scaler_sub.transform([scores])
    r = autoencoder.predict(s, verbose=0)
    err = np.abs(s - r)[0]

    subs = ["Math","Physics","English"]
    return [subs[i] for i in range(3) if err[i] > 0.1]

def recommendations(scores, weak):
    subs = ["Math","Physics","English"]
    rec = []

    for i, s in enumerate(subs):
        if s in weak:
            if scores[i] < 50:
                rec.append(f"{s}: Improve basics")
            elif scores[i] < 70:
                rec.append(f"{s}: Practice more")
            else:
                rec.append(f"{s}: Minor improvement")

    return rec

def risk(cgpa, unc):
    if cgpa > 7: r = "Low"
    elif cgpa > 5: r = "Moderate"
    else: r = "High"
    if unc > 0.5: r += " (Low Confidence)"
    return r

def percentile(score):
    return (np.sum(y < score)/len(y))*100

# ================= MAIN FUNCTION =================
def analyze_student(seq, scores):

    seq = scaler_seq.transform([seq]).reshape((1,4,1))
    mean, std = mc_predict(seq)

    cgpa = float(mean[0][0])
    unc = float(std[0][0])

    weak = weak_subjects(scores)
    rec = recommendations(scores, weak)

    return {
        "cgpa": round(cgpa,2),
        "unc": round(unc,2),
        "risk": risk(cgpa, unc),
        "weak": weak,
        "rec": rec,
        "percentile": round(percentile(cgpa),2)
    }