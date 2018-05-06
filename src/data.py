import pprint
import untangle as ut
import xmltodict
from glob import glob
import tensorflow as tf
from partial import _, __, ___
from random import shuffle
from config import Config

config = Config()
sess = tf.InteractiveSession()
f = lambda g: lambda *p: g(p[0])
tobytes = lambda x: sess.run(tf.cast(x, tf.uint8)).tobytes()
sff = lambda x: shuffle(x)
bytesList = lambda x: tf.train.BytesList(value=x)
bytesFeature = lambda x: tf.train.Feature(bytes_list=x)
int64Feature = lambda x: tf.train.Feature(int64_list=x)


pp = pprint.PrettyPrinter().pprint

_.go(glob('data/Images/*/*'),
		_.take(config.split),
		_.shuffle,
		_.map(f(tf.read_file)),
		_.map(f(tf.image.decode_jpeg)),
		_.map(f(tobytes)),
		bytesList,
		bytesFeature,
		pp
)









pp(len(glob('data/Images/*/*')))

# obj = xmltodict.parse(open('data/Annotation/n02085620-Chihuahua/n02085620_7').read())
# pp(obj['annotation']['object']['bndbox'])
annotation = ut.parse(
    'data/Annotation/n02085620-Chihuahua/n02085620_7').annotation
pp(int(annotation.object.bndbox.xmin.cdata))
