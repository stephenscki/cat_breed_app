# converting from keras file to tflite file using tflite converte
import tensorflow.lite as lite



def convert(filename):
	converter = lite.TFLiteConverter.from_keras_model_file(filename)
	tflite_model = converter.convert()
	open('catid_graph.tflite','wb').write(tflite_model) 

def main():
	convert('weight_files/20200201-15:00/w-48-0.74.hdf5')


if __name__ == '__main__':
	main()
