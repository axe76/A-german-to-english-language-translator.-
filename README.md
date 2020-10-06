# Neural Translator.-
A german to english translator. Dataset taken from http://www.manythings.org/anki/deu-eng.zip.
Run create_data.py and split.py to preprocess data and split into training and test sets stored in pickle objects.
Run train.py to train model. Validate.py is to test how the model is working. In validate.py load in the model weights after training them.
# NMT Folder.-
This folder contains a seq2seq neural machine translator for spannish to english translation. The dataset used is also provided or can be downloaded using <br/>
#path_to_zip = tf.keras.utils.get_file('spa-eng.zip', origin='http://download.tensorflow.org/data/spa-eng.zip', extract=True) <br/>
#path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt" <br/>
The reference link used: https://colab.research.google.com/github/tensorflow/tensorflow/blob/r1.9/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb#scrollTo=DUQVLVqUE1YW 
