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
from multiprocessing import Process, cpu_count
import itertools


cores = int(cpu_count())
pp = PrettyPrinter().pprint
le = LabelEncoder()
# tf.executing_eagerly()
config = Config()
# sess = tf.InteractiveSession()
# graph = tf.Graph()
# sess = tf.Session(graph=tf.Graph())
# sess.run(tf.global_variables_initializer())
# f = lambda g: lambda *p: g(p[0])
readFile = lambda *x: tf.read_file(x[0])
decodeJpeg = lambda *x: tf.image.decode_jpeg(x[0])
rgbToGrayscale = lambda *x: tf.image.rgb_to_grayscale(x[0])
toInt8 = lambda *x: tf.cast(x[0], tf.uint8)
toFloat16 = lambda *x: tf.cast(x[0], tf.float16)
tobytes = lambda s: lambda *x: s.run(x[0]).tobytes()
bytesList = lambda *x: tf.train.BytesList(value=[x[0]])
int64List = lambda *x: tf.train.Int64List(value=[x[0]])
bytesFeature = lambda *x: tf.train.Feature(bytes_list=x[0])
int64Feature = lambda *x: tf.train.Feature(int64_list=x[0])
resizeImage = lambda *x: tf.image.resize_images(x[0], [config.crop_size, config.crop_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
features = lambda *x: tf.train.Features(feature=x[0])
example = lambda *x: tf.train.Example(features=x[0])
serializeToString = lambda *x: x[0].SerializeToString()
write = lambda writer: lambda *x: writer.write(x[0])
oneHot = lambda class_num: lambda *x: tf.one_hot(x[0], class_num, 1, 0)
split = lambda l, n: [l[i:i+n] for i in range(0, len(l), n)]
slice = lambda start, end: lambda l: l[start:end]

def saveLabels(labels):
		def saveLabel(x):
				labels['val'] = x[1::2]
				return x[::2]
		return saveLabel
		

def _cropImage(r, *s):
		name = r[0].find('name').text
		box = r[0].find('bndbox')
		offset_y = int(box.find('ymin').text)
		offset_x = int(box.find('xmin').text)
		target_h = int(box.find('ymax').text) - offset_y
		target_w = int(box.find('xmax').text) - offset_x
		return tf.image.crop_to_bounding_box(r[1], offset_y, offset_x, target_h, target_w), name


def cropImage(r, *s):
		origin = parse(r[1]).getroot()
		boxes = origin.findall('object')
		return _.go(
					_.zip(boxes, [r[0] for i in range(len(boxes))]),
					_.map(_cropImage),
					_.flatten
					)


def cropImages(annots):
		return lambda images: _.go(_.zip(images, annots),
															_.map(cropImage),
															_.flatten
															)


images = sorted(glob('data/Images/*/*'))
# labels = _.map(images, lambda x, *r: x.split('/')[2])
labels: dict = dict()
# le.fit(labels)
class_num = len(glob('data/Images/*'))
annots = sorted(glob('data/Annotation/*/*'))
# annots = _.take(annots, 1024)


images = _.go(
		images,
		# _.take(1024),
		# _.take(config.split),
		# _.shuffle,
		_.map(readFile),
		_.map(decodeJpeg),
)

croppedImages = _.go(
		images,
		cropImages(annots),
		saveLabels(labels),
)

imageToFeature = lambda s: _.pipe(
		_.map(rgbToGrayscale),
		_.map(resizeImage),
		_.map(toInt8),
		_.map(tobytes(s)),
		_.map(bytesList),
		_.map(bytesFeature),
)

labelToFeature = lambda s: _.pipe(
		# _.values,
		# _.first,
		# le.fit_transform,
		# list,
		_.map(oneHot(class_num)),
		_.map(toInt8),
		_.map(tobytes(s)),
		_.map(bytesList),
		_.map(bytesFeature),
)

writeFeatures = lambda writer: _.pipe(
		_.map(lambda *r: {'image': r[0][0], 'label': r[0][1]}),
		_.tap(pp),
		_.map(features),
		_.map(example),
		_.map(serializeToString),
		_.map(write(writer)),
)

def writeExample(process):
		def write(*x):
				import tensorflow as tf
				data = _.unzip(x[0])
				sess = tf.InteractiveSession()
				filename = f'data/Records/data-{process}-{x[1]}.tfrecords'
				writer = tf.python_io.TFRecordWriter(filename)
				image_features = imageToFeature(sess)(data[0])
				pp('why!!!!!!!!!!!!!111111')
				label_features = labelToFeature(sess)(data[1])
				pp('why!!!!!!!!!!!!22222222222')
				writeFeatures(writer)(_.zip(image_features, label_features))
				pp('why!!!!!!!!!!!!!333333333333')
				writer.close()
				sess.close()
		return write

def writeExamples(*x):
		_.go(
			x[0],
			_.map(writeExample(x[1])),
		)

makeProcess = lambda *x: Process(target=writeExamples, args=(x))

group = lambda n: lambda l: ([e for e in t if e != None] for t in itertools.zip_longest(*([iter(l)] * n)))
# chunk = lambda l, n: [l[i::n] for i in range(n)]

def chunk(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

# pp(len(croppedImages))
# pp(len(split(croppedImages, config.split)))
# pp(len(_.range(0, len(croppedImages), config.split)))
# pp(len(chunk(split(croppedImages, config.split), cores)))

labels['val'] = le.fit_transform(labels['val'])
# pp(len(labels['val']))

_.go(
		chunk(split(_.zip(croppedImages, labels['val']), config.split), cores), #_.range(cores)),
		_.map(makeProcess),
		_.tap(_.map(lambda *x: x[0].start())),
		_.tap(_.map(lambda *x: x[0].join())),
)
