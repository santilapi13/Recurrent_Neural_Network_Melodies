import music21 as m21
import os

def save_melody(melody, step_duration=0.25, format="midi", file_name="mel.mid"):
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

def load(file_path):
	with open(file_path, "r") as fp:
		song = fp.read()
	return song

def main():
    # Obtener la ruta al directorio del escritorio
    escritorio = os.path.expanduser("~/OneDrive")

    # Construir la ruta al directorio "programacion\dataset"
    ruta_dataset = os.path.abspath(os.path.join(escritorio,"Escritorio", "programacion", "dataset"))

    # Iterar sobre todos los archivos en el directorio
    for archivo in os.listdir(ruta_dataset):
        ruta_completa = os.path.join(ruta_dataset, archivo)
        song = load(ruta_completa)
        melody = song.split()
        save_melody(melody, file_name=archivo+".mid".format(archivo[:-4]))

if __name__ == "__main__":
    main()