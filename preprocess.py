import os
import music21 as m21
import json
from tensorflow import keras
import numpy as np

KERN_DATASET_PATH = "songs/allerkbd"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
SEQUENCE_LENGTH = 64  # Cant. de elementos de la secuencia (fijado)
MAPPING_PATH = "mapping.json"

ACCEPTABLE_DURATIONS = [   # respecto de una negra
	0.25,   # semicorchea
	0.5,   # corchea
	0.75,  # corchea con puntillo
	1.0,  #  negra
	1.5,  # negra con puntillo
	2,  # blanca
	3,  # blanca con puntillo
	4  # redonda
]

def load_songs_in_kern(dataset_path):
	songs = []

	for path, subdirs, files in os.walk(dataset_path):
		for file in files:
			if file[-3:] == "krn":
				song = m21.converter.parse(os.path.join(path, file))
				songs.append(song)
	return songs

def has_acceptable_durations(song, acceptable_durations):
	for note in song.flatten().notesAndRests: # operacion de flatten y solo quedarse con notas y silencios
		if note.duration.quarterLength not in acceptable_durations:
			return False
	return True

def transpose(song):
	# Obtener tonalidad de la canción (si es explícita)
	parts = song.getElementsByClass(m21.stream.Part)  # Obtiene las distintas pistas
	measures_part0 = parts[0].getElementsByClass(m21.stream.Measure) # Separa cada pista en sus barras
	key = measures_part0[0][4] 

	# Estimar la tonalidad con music21 (si no es explícita)
	if not isinstance(key, m21.key.Key):
		key = song.analyze("key")  # Ordena adivinar la tonalidad

	#print(key)

	# Obtener intervalo de transposición
	if key.mode == "major":
		interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
	elif key.mode == "minor":
		interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

	# Transponer la canción en base al intervalo calculado
	transposed_song = song.transpose(interval)

	return transposed_song

def encode_song(song, time_step = 0.25):
	# 1 negra = 4 semicorcheas
	encoded_song = []	

	for event in song.flatten().notesAndRests:
		# Notas
		if isinstance(event, m21.note.Note):
			symbol = event.pitch.midi
		# Silencios
		elif isinstance(event, m21.note.Rest):
			symbol = "r"

		# Convertir nota/silencio en la notación de series temporales
		steps = int(event.duration.quarterLength / time_step) 
		for step in range(steps):
			if step == 0:
				encoded_song.append(symbol)
			else:
				encoded_song.append("_")

	# Convertir lista en string
	encoded_song = " ".join(map(str, encoded_song))

	return encoded_song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
	new_song_delimiter = "/ " * sequence_length   # Cantidad fija de "/ " entre canciones
	songs = ""

	# Carga de canciones codificadas y agregar delimitadores
	for path, _, files in os.walk(dataset_path):
		for file in files:
			file_path = os.path.join(path, file)
			song = load(file_path)
			songs = songs + song + " " + new_song_delimiter

	songs = songs[:-1]  # Para eliminar espacio vacío al final del string

	# Guardar el string con todos los datos
	with open(file_dataset_path, "w") as fp:
		fp.write(songs)
	
	return songs

def load(file_path):
	with open(file_path, "r") as fp:
		song = fp.read()
	return song

def create_mapping(songs, mapping_path):
	mappings = {}

	# Identificar vocabulario
	songs = songs.split()
	vocabulary = list(set(songs))  # Elimina símbolos repetidos
	
	# Crear el mapeo
	for i, symbol in enumerate(vocabulary):
		mappings[symbol] = i  # Se le asigna a cada símbolo un número distinto
	
	# Guardar vocabulario en un archivo json
	with open(mapping_path, "w") as fp:
		json.dump(mappings, fp, indent = 4)

	return mappings	

def convert_songs_to_int(songs):
	# Cargar el mapeo simbolo-numero
	with open(MAPPING_PATH, "r") as fp:
		mappings = json.load(fp)	

	# Castear el string de canciones a una lista
	songs = songs.split()

	# Mapear canciones a entero
	int_songs = []

	for symbol in songs: 
		int_songs.append(mappings[symbol])

	return int_songs

def generate_training_sequences(sequence_length):
	# Cargar canciones y mapearlas en enteros
	songs = load(SINGLE_FILE_DATASET)
	int_songs = convert_songs_to_int(songs)
	
	# Generar las secuencias de entrenamiento
	inputs = []
	targets = []

	num_sequences = len(int_songs) - sequence_length
	for i in range(num_sequences):
		inputs.append(int_songs[i:i+sequence_length])
		targets.append(int_songs[i+sequence_length])

	# Formatear secuencias en one-hot encoding
	# inputs: (num. de secuencias, longitud de la secuencia, vocabulary_size)
	vocabulary_size = len(set(int_songs))  # Elimina elementos repetidos
	inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
	targets = np.array(targets)

	return inputs, targets

def preprocess(dataset_path):
	print("Loading songs")
	songs = load_songs_in_kern(dataset_path)
	
	for i, song in enumerate(songs):
		# Filtrado de canciones según duraciones de sus notas
		if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
			continue
	
		# Transposición de canciones a tonalidad de C/Am
		song = transpose(song)

		# Representar canciones en series temporales
		encoded_song = encode_song(song)
		
		# Guardar canciones en archivo de texto
		save_path = os.path.join(SAVE_DIR, str(i))
		with open(save_path, "w") as fp:
			fp.write(encoded_song)

def main():
	preprocess(KERN_DATASET_PATH)
	songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
	create_mapping(songs, MAPPING_PATH)
	inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

if __name__ == "__main__":
	main()