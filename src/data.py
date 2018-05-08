from pprint import PrettyPrinter
from glob import glob
from xml.etree.ElementTree import parse
import tensorflow as tf
from partial import _, __, ___
from random import shuffle
from config import Config
import numpy as np
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder

pp = PrettyPrinter().pprint
le = LabelEncoder()

config = Config()
sess = tf.InteractiveSession()
# graph = tf.get_default_graph()
# sess = tf.Session(graph=graph)
# sess.run(tf.global_variables_initializer())
f = lambda g: lambda *p: g(p[0])
toInt8 = lambda x: sess.run(tf.cast(x, tf.uint8))
toFloat16 = lambda x: sess.run(tf.cast(x, tf.float16))
tobytes = lambda x: x.tobytes()
sff = lambda x: shuffle(x)
bytesList = lambda x: tf.train.BytesList(value=[x])
int64List = lambda x: tf.train.Int64List(value=[x])
bytesFeature = lambda x: tf.train.Feature(bytes_list=x)
int64Feature = lambda x: tf.train.Feature(int64_list=x)
resizeImage = lambda x: tf.image.resize_images(x, [config.crop_size, config.crop_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def saveLabels(x):
		global labels
		labels = x[1::2]
		return x[::2]


def _cropImage(r, *s):
		name = r[0].find('name').text
		box = r[0].find('bndbox')
		offset_y = int(box.find('ymin').text)
		offset_x = int(box.find('xmin').text)
		target_h = int(box.find('ymax').text) - offset_y
		target_w = int(box.find('xmax').text) - offset_x
		return tf.image.crop_to_bounding_box(r[1][0], offset_y, offset_x, target_h, target_w), name


def cropImage(r, *s):
		origin = parse(r[1]).getroot()
		boxes = origin.findall('object')
		return _.go(
					_.zip(boxes, [r[0] for i in range(len(boxes))]),
					_.map(_cropImage),
					_.flatten
					)


def cropImages(images):
		return _.go(_.zip(images, annots),
								_.map(f(cropImage)),
								_.flatten
								)


images = sorted(glob('data/Images/*/*'))
labels = _.map(images, lambda x, *r: x.split('/')[2])
le.fit(labels)
class_num = len(glob('data/Images/*'))
annots = sorted(glob('data/Annotation/*/*'))
annots = _.take(annots, config.split)

image_features = _.go(
		images,
		_.take(config.split),
		# _.shuffle,
		_.map(f(tf.read_file)),
		_.map(f(tf.image.decode_jpeg)),
		cropImages,
		saveLabels,
		_.tap(pp),
		_.map(f(resizeImage)),
		_.tap(pp),
		_.map(f(tf.image.rgb_to_grayscale)),
		_.tap(pp),
		_.map(f(toInt8)),
		_.tap(pp),
		_.map(f(tobytes)),
		_.tap(pp),
		_.map(f(bytesList)),
		_.tap(pp),
		_.map(f(bytesFeature)),
		_.tap(pp)
)



label_features = _.go(
		labels,
		le.fit_transform,
		list,
		_.map(lambda *r: tf.one_hot(r[0], class_num, 1, 0)),
		_.tap(pp),
		_.map(f(toInt8)),
		_.tap(pp),
		_.map(f(tobytes)),
		_.tap(pp),
		_.map(f(bytesList)),
		_.tap(pp),
		_.map(f(bytesFeature)),
		_.tap(pp),
)

writer = tf.python_io.TFRecordWriter(config.record_filename)

_.go(
		_.zip(image_features, label_features),
		_.map(lambda *r: {'image': r[0][0], 'label': r[0][1]}),
		# _.tap(pp),
		_.map(lambda *r: tf.train.Features(feature=r[0])),
		_.map(lambda *r: tf.train.Example(features=r[0])),
		_.map(lambda *r: r[0].SerializeToString()),
		_.tap(pp),
		_.map(lambda *r: writer.write(r[0])),
)



