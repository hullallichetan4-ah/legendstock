import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

st.title("📈 Stock Market Trend Prediction Using LSTM")

stock = st.text_input(
    "Enter Stock Symbol",
    "AAPL"
)

start = "2015-01-01"
end = "2026-01-01"

data = yf.download(stock,start,end)

if len(data) == 0:
    st.error("Invalid stock symbol")
    st.stop()

st.subheader("Stock Data")
st.write(data.tail())

close_price = data[['Close']]

scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(close_price)

train_size = int(len(scaled)*0.8)

train = scaled[:train_size]
test = scaled[train_size:]

X_train=[]
Y_train=[]

for i in range(60,len(train)):
    X_train.append(train[i-60:i,0])
    Y_train.append(train[i,0])

X_train=np.array(X_train)
Y_train=np.array(Y_train)

X_train=np.reshape(
    X_train,
    (X_train.shape[0],
     X_train.shape[1],
     1)
)

model=Sequential()

model.add(
    LSTM(
        50,
        return_sequences=True,
        input_shape=(60,1)
    )
)

model.add(
    LSTM(50)
)

model.add(Dense(25))
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

st.write("Training Model...")

model.fit(
    X_train,
    Y_train,
    epochs=5,
    batch_size=32,
    verbose=0
)

inputs=scaled[
    train_size-60:
]

X_test=[]

for i in range(60,len(inputs)):
    X_test.append(
        inputs[i-60:i,0]
    )

X_test=np.array(X_test)

X_test=np.reshape(
    X_test,
    (X_test.shape[0],
     X_test.shape[1],
     1)
)

pred=model.predict(X_test)

pred=scaler.inverse_transform(pred)

actual=close_price[
    train_size:
]

st.subheader(
    "Actual vs Predicted"
)

fig=plt.figure(figsize=(10,5))

plt.plot(
    actual.values,
    label="Actual"
)

plt.plot(
    pred,
    label="Predicted"
)

plt.legend()

st.pyplot(fig)

last_60=scaled[-60:]

future=np.array(
    last_60
).reshape(1,60,1)

next_day=model.predict(
    future
)

price=scaler.inverse_transform(
    next_day
)

st.success(
    f"Predicted Next Day Price: ${price[0][0]:.2f}"
)
