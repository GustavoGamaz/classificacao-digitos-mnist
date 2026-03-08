import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Carregamento dos dados
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalização dos pixels
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# Redimensionamento das imagens para vetores
X_train = X_train.reshape((60000, 784))
X_test = X_test.reshape((10000, 784))

# Conversão dos rótulos para formato categórico
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Construção do modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(784,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compilação
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Treinamento
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# Avaliação
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia no teste: {accuracy:.4f}")

# Predições
predictions = model.predict(X_test)

# Visualização das previsões
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Pred: {predictions[i].argmax()}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("resultado_modelo.png", dpi=300)
plt.show()