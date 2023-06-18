import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

import tensorflow as tf

df = pd.read_csv("diabetes.csv")

#for i in range(len(df.columns[:-1])):
#    label = df.columns[i]
#    plt.hist(df[df['Outcome']==1][label], color='blue', label="Has Diabetes", alpha=0.7, density=True, bins=15)
#    plt.hist(df[df['Outcome']==0][label], color='red', label="No Diabetes", alpha=0.7, density=True, bins=15)
#    plt.title(label)
#    plt.ylabel("Probability")
#    plt.xlabel(label)
#    plt.legend()
#    plt.show()

#Features
X = df[df.columns[:-1]].values
#Labels
y = df[df.columns[-1]].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

data = np.hstack((X,np.reshape(y,(-1,1))))
transformed_df = pd.DataFrame(data, columns=df.columns)

over = RandomOverSampler()
X, y = over.fit_resample(X, y)

#for i in range(len(df.columns[:-1])):
#    label = df.columns[i]
#    plt.hist(transformed_df[transformed_df['Outcome']==1][label], color='blue', label="Has Diabetes", alpha=0.7, density=True, bins=15)
#    plt.hist(transformed_df[transformed_df['Outcome']==0][label], color='red', label="No Diabetes", alpha=0.7, density=True, bins=15)
#    plt.title(label)
#    plt.ylabel("Probability")
#    plt.xlabel(label)
#    plt.legend()
#    plt.show()

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

#Create Neaural net model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu'), #if x <= 0 ==> 0, if x > 0 ==> x
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss=tf.keras.losses.BinaryCrossentropy(), 
              metrics=['accuracy'])

#Train Model
model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_valid, y_valid))
#model.evaluate(X_valid, y_valid)
model.evaluate(X_test, y_test)