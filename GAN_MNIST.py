from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

img_shape = (28, 28, 1) # 이미지 shape
latent_dim = 100 #은닉층

#생성기
def generator_model():

    model = Sequential()
    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D( 1 , kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    model.summary()
    noise = Input(shape=(latent_dim,))
    img = model(noise)
    return Model(noise, img)

#분류기 모델
def discriminator_model():

    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    img = Input(shape=img_shape)
    validity = model(img)
    return Model(img, validity)

# Adam으로 최적화
optimizer = Adam(0.001, 0.5)

# 분류기 모델 컴파일
discriminator = discriminator_model()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 생성기 모델
generator = generator_model()
z = Input(shape=(latent_dim,))
img = generator(z)

# 생성기만 사용
discriminator.trainable = False

# 분류기는 생성된 이미지를 가져와 거짓 판별 시 사용
valid = discriminator(img)

# 결합 모델 ( 분류기 속이도록 훈련 )
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

# 훈련
def train(epochs, batch_size=128, save_interval=50):
    os.makedirs('images', exist_ok=True)

    # 데이터 로드 ( 테스트 데이터가 필요없다 )
    (X_train, _), (_, _) = mnist.load_data()

    # 음수로 변환
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    # 반대로 대립 시킴
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # 실제 이미지를 랜덤으로 가져옴
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]

        # 노이즈와 거짓 이미지를 생성
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(noise)

        # 분류기 훈련
        # fit은 epochs와 batch_size를 한번에 넘겨주지만
        # train_on_batch는 현재 전달받은 데이터를 모두 활용하여 gradient vector를
        # 계산하여 사용하는데 GAN의 경우 생성기에서 거짓 이미지를 만들기 때문에,
        # epoch마다 새로운 데이터를 넘겨주어야 한다.
        D_loss_real = discriminator.train_on_batch(real_imgs, valid)
        D_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
        D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)

        # 생성기 훈련
        g_loss = combined.train_on_batch(noise, valid)

        # 저장 간격
        if epoch % save_interval == 0:
            # N 에폭당 D_loss와 acc, g_loss를 프린트
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, D_loss[0], 100 * D_loss[1], g_loss))
            # 샘플 이미지 저장
            save_imgs(epoch)

#이미지 저장
def save_imgs(epoch):
    noise = np.random.normal(0, 1, (25, latent_dim))
    gen_imgs = generator.predict(noise)

    # 이미지를 0-1 사이로 변환
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(5, 5)
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("images/mnist_%d.png" % epoch)
    plt.close()


train(epochs=10000, batch_size=32, save_interval=1000)
