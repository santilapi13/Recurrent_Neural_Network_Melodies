from tensorflow import keras
import numpy as np
import json
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
import music21 as m21

class MelodyGenerator:
	def __init__(self, model_path="model.h5"):
		self.model_path = model_path
		self.model = keras.models.load_model(model_path)
		
		with open(MAPPING_PATH, "r") as fp:
			self._mappings = json.load(fp)

		self._start_symbols = ["/"] * SEQUENCE_LENGTH

	def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
		# Creación de la semilla con símbolos iniciales
		seed = seed.split()
		melody = seed
		seed = self._start_simbols + seed  # Se concatena luego de los símbolos iniciales

		# Mapear semilla en enteros según nuestro vocabulario
		seed = [self._mappings[symbol] for symbol in seed]

		for _ in range(num_steps):
			# Limitar la semilla según la longitud máxima de secuencia
			seed = seed[-max_sequence_length:]
			
			# Representar semilla como one-hot encoding
			onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
			onehot_seed = onehot_seed[np.newaxis, ...]  # Array de 3 dimensiones

			# Hacer la predicción
			probabilities = self.model.predict(onehot_seed)[0]  # Array de distribución de probabilidades
			# [0.1, 0.2, 0.1, 0.6] -> 1 (por la softmax) el modelo predice cuál es el símbolo con mayor probabilidad de venir luego de lo anterior.
			output_int = self._sample_with_temperature(probabilities, temperature)

			# Actualizar semilla
			seed.append(output_int)

			# Mapear enteros obtenidos en símbolos
			output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

			# Chequear si se está al final de una melodía
			if output_symbol == "/":
				break

			# Actualizar melodía
			melody.append(output_symbol)

		return melody

	def _sample_with_temperature(self, probabilities, temperature):
		predictions = np.log(probabilities) / temperature
		probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

		choices = range(len(probabilities))
		index = np.random.choice(choices, p=probabilities)

		return index

	def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
		# Crear la melodía en formato de music21
		stream = m21.stream.Stream()		

		# Parsear los símbolos de la melodía y crear los objetos de notas/silencios
		start_symbol = None
		step_counter = 1
		
		for i, symbol in enumerate(melody):
				if symbol != "_" or i + 1 == len(melody):  # Caso 1: evento de nota/silencio (o final)
					if start_symbol is not None:  # diferenciar del símbolo inicial
						quarter_length_duration = step_duration * step_counter
						if start_symbol == "r":  # silencio
							m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
						else:  # nota
							m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
						
						stream.append(m21_event)							
						step_counter = 1
						start_symbol = symbol
				else:  # Caso 2: prolongación del último evento "_"
					step_counter += 1

		# Guardarlo en archivo MIDI
		stream.write(format, file_name)

if __name__ == "__main__":
	mg = MelodyGenerator()
	seed ="55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
	melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.7)
	mg.save_melody(melody)
	print(melody)