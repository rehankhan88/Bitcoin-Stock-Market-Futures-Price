# import numpy as np
# import pandas as pd
# import yfinance as yf
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# import streamlit as st


# model = load_model('C:\Users\MUHIB\Desktop\ppppp\Bitcoin Stock Market Futures Price\Bitcoin_Price_prediction_Model.keras')
# st.header('Bitcoin Price prediction Model')
# st.subheader('Bitcoin Price Data')
# data = pd.DataFrame(yf.download("BTC-USD","2014-01-01","2024-05-30"))
# data.reset_index()
# st.write(data)

# st.subheader('Bitcoin Line Chart')
# data.drop(columns=["Date","Open","High","Low","Adj Close","Volume"],inplace=True)
# st.line_chart(data)

# train_data= data[:-100]
# test_data= data[:-200]

# scaler= MinMaxScaler(feature_range=(0,1))
# train_data_scale= scaler.fit_transform(train_data)
# test_data_scale= scaler.fit_transform(test_data)
# base_days=100
# x = []
# y = []
# for i in range(base_days, train_data_scale.shape[0]):
#     x.append(train_data_scale[i - base_days:i])
#     y.append(train_data_scale[i, 0])
    
    
# x,y= np.array(x), np.array(y)
# x=np.reshape(x,(x.shape[0],x.shape[1],1))

# st.subheader('Predicted Vs Original Price')
# pred=model.predict(x)
# pred=scaler.inverse_transform(pred)
# preds= pred.reshape(-1,1)
# ys=scaler.inverse_transform(y.reshape(-1,1))
# preds=pd.DataFrame(preds,columns=['Predicted Price'])
# ys= pd.DataFrame(ys,columns=['Original Price'])
# chart_data=pd.concat((preds,ys),axis=1)
# st.write(chart_data)
# st.subheader('Predicted Vs Original Prices Chart ')
# st.line_chart(chart_data)


# # Future predictions
# m = y
# z = []
# future_days = 5 
# for i in range(base_days,len(m)+future_days):
#     m=m.reshape(1,-1)
#     inter = [m[-base_days:,0]] 
#     inter=np.array(inter)
#     inter = np.reshape(inter, (inter.shap[0], inter.shap[1], 1))
#     pred = model.predict(inter)
#     m = np.append(m, pred)
#     z=np.append(z,pred)  
# st.subheader('Predictted Future Days Bitcoin Price')    
# z=np.array(z) 
# z= scaler.inverse_transform(z.reshape(-1,1))   
# st.line_chart(z)



import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Load the model
model = load_model('C:/Users/MUHIB/Desktop/ppppp/Bitcoin Stock Market Futures Price/Bitcoin_Price_prediction_Model.keras')

# Streamlit headers
st.header('Bitcoin Price Prediction Model')
st.subheader('Bitcoin Price Data')

# Download and display data
data = pd.DataFrame(yf.download("BTC-USD", "2014-01-01", "2024-05-30"))
data.reset_index(inplace=True)
st.write(data)

# Line chart of closing prices
st.subheader('Bitcoin Line Chart')
st.line_chart(data[['Date', 'Close']].set_index('Date'))

# Prepare the data for training and testing
train_data = data['Close'][:-100]
test_data = data['Close'][-100:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
test_data_scaled = scaler.transform(test_data.values.reshape(-1, 1))

base_days = 100
x_train = []
y_train = []

for i in range(base_days, len(train_data_scaled)):
    x_train.append(train_data_scaled[i - base_days:i])
    y_train.append(train_data_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Predictions
st.subheader('Predicted vs Original Price')
predictions = model.predict(x_train)
predictions = scaler.inverse_transform(predictions)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))

# Creating dataframe for comparison
comparison_df = pd.DataFrame({
    'Predicted Price': predictions.flatten(),
    'Original Price': y_train.flatten()
})

st.write(comparison_df)
st.subheader('Predicted vs Original Prices Chart')
st.line_chart(comparison_df)

# Future predictions
m = train_data_scaled[-base_days:]
future_predictions = []

for _ in range(5):  # Number of days to predict into the future
    m = m.reshape(1, -1, 1)
    pred = model.predict(m)
    future_predictions.append(pred[0, 0])
    m = np.append(m[:, 1:, :], [[[pred[0, 0]]]], axis=1)  # Ensure pred has 3 dimensions

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

st.subheader('Predicted Future Days Bitcoin Price')
st.line_chart(future_predictions)





