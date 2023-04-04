import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=100, verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Courbe d\'erreur')
plt.ylabel('Erreur')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Test'], loc='upper right')
plt.show()

Y_pred = model.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print('Matrice de confusion:')
print(cm)
plt.matshow(cm)
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()