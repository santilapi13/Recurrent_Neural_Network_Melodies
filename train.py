from preprocess import generate_training_sequences, SEQUENCE_LENGTH
from tensorflow import keras

def build_model(input_units, output_units, num_units, loss, learning_rate):
	# Creacion de la arquitectura
	input = keras.layers.Input(shape=(input_units, output_units))
	x = keras.layers.LSTM(num_units[0])(input)
	x = keras.layers.Dropout(0.2)(x)
	
	output = keras.layers.Dense(output_units, activation="softmax")(x)

	model = keras.Model(input, output)

	# Compilación del modelo
	model.compile(
		loss=loss, 
		optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
		metrics=["accuracy"]
	)

	model.summary()

	return model

OUTPUT_UNITS = 38   # Tamaño del vocabulario -> Neuronas de la capa de salida
NUM_UNITS = [256]  # Neuronas de las capas internas
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64  # Muestras que se verán antes del backpropagation
SAVE_MODEL_PATH = "model.h5"

def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
	# Generar las secuencias de entrenamiento
	inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

	# Armar la RNN
	model = build_model(SEQUENCE_LENGTH, output_units, num_units, loss, learning_rate)

	# Entrenar el modelo
	model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

	# Guardar el modelo
	model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
	train()