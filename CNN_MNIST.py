import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D, MaxPooling2D,LeakyReLU
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# 데이터 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 레이블 정의
fashion_mnist_labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
                        "Sandal",     "Shirt",  "Sneaker", "Bag",  "Ankle boot"]

#데이터 전처리
x_train = x_train / 255.0
x_test = x_test / 255.0

#학습용과 검증용으로 분리
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)
x_validate = x_validate.reshape(x_validate.shape[0], 28,28,1)

# our 3 models

#Model_1
cnn_model_1 = Sequential([
    Conv2D(32, kernel_size=3,input_shape=[28,28,1],activation='relu', kernel_initializer='he_normal'),
    MaxPooling2D(pool_size=2),
    Dropout(0.3),
    Conv2D(64, kernel_size=3,activation='relu'),
    Dropout(0.3),
    Conv2D(128, kernel_size=3,activation='relu'),
    Dropout(0.4),
    #fully connected
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
    ], name='Model_1')

#Model_2
cnn_model_2 = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=[28,28,1]),
    MaxPooling2D(pool_size=2),
    Dropout(0.5),
    Conv2D(64, kernel_size=3, activation='relu'),
    Dropout(0.5),
    #fully connected
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
    ], name='Model_2')

#Model_3
cnn_model_3 = Sequential([
    Conv2D(32, kernel_size=3, activation='relu',
           input_shape=[28,28,1], kernel_initializer='he_normal'),
    MaxPooling2D(pool_size=2),
    Dropout(0.25),
    Conv2D(64, kernel_size=3, activation='relu'),
    Dropout(0.25),
    Conv2D(128, kernel_size=3, activation='relu'),
    Dropout(0.4),
    #fully connected
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4, name='Dropout'),
    Dense(10, activation='softmax')
    ], name='Model_3')

cnn_models = [cnn_model_1, cnn_model_2, cnn_model_3]

#history를 dictionary에 저장
history_dict = {}

#Early_stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience = 10, verbose=1, mode='auto')

for model in cnn_models:
    #컴파일
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )
    #훈련
    history = model.fit(
        x_train, y_train,
        batch_size=512,
        epochs=100, verbose=2,
        validation_data=(x_validate, y_validate),
        callbacks=[early_stopping]
    )
    #HISTORY 저장
    history_dict[model.name] = history

#그래프 4개
fig, (ax1, ax2, ax11, ax21) = plt.subplots(4, figsize=(8, 6))

#각 모델마다 Accuracy, loss를 그래프에 그림
for history in history_dict:
    val_acc = history_dict[history].history['val_acc']
    acc = history_dict[history].history['acc']
    val_loss = history_dict[history].history['val_loss']
    loss = history_dict[history].history['loss']
    ax1.plot(val_acc, label=history)
    ax11.plot(acc, label=history)
    ax2.plot(val_loss, label=history)
    ax21.plot(loss, label=history)

ax1.set_ylabel('validation accuracy')
ax2.set_ylabel('validation loss')
ax11.set_ylabel('accuracy')
ax21.set_ylabel('loss')
ax21.set_xlabel('epochs')
ax1.legend()
ax2.legend()
ax11.legend()
ax21.legend()
plt.show()

#앙상블 기법
i=1
predictions = 0
for model in cnn_models:
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy',i ,':','%0.4f'%score[1],'   Test loss',i ,':','%0.4f'%score[0])
    i=i+1
    predictions += model.predict(x_test)

predictions /=3.0

#Total Accuracy
count = 0
for i in range(0,10000):
    predictions_label = np.argmax(predictions[i])
    if predictions_label == y_test[i]:
        count= count+1
    i = i+1
count= count/10000.0

print('\nTotal accuracy: ', count)