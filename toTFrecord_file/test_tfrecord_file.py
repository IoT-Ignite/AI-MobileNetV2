import tensorflow as tf
import numpy as np
import sys
import cv2
import glob
training_filename="./tfrecord/train.tfrecord"
def draw_box(img, box):
    x0, y0, x1, y1 = box[:4]
    h, w = img.shape[:2]
    cv2.rectangle(
            img,
            tuple( int(e) for e in (w*x0, h*y0) ),
            tuple( int(e) for e in (w*x1, h*y1) ),
            (255,0,0),
            int(max(1, 0.01 * (h+w)/2.0))
            )

def main():
    tf.enable_eager_execution()
    rec_files = glob.glob(training_filename)
    #print rec_files
    np.random.shuffle(rec_files)
    rec_file = rec_files[0]
    print('\trecord file : {}'.format(rec_file))
    rec = tf.data.TFRecordDataset(rec_file)

    image_feature_description = {
        'image/height':tf.io.FixedLenFeature([], tf.int64),
        'image/width':tf.io.FixedLenFeature([], tf.int64),
        'image/filename':tf.io.FixedLenFeature([], tf.string),
        'image/source_id':tf.io.FixedLenFeature((), tf.string),
        'image/key/sha256':tf.io.FixedLenFeature((), tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format':tf.io.FixedLenFeature((), tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':tf.io.VarLenFeature(tf.float32),
        'image/object/class/text':tf.io.VarLenFeature(tf.string),
        'image/object/class/label':tf.io.VarLenFeature(tf.int64),
        }

    parse = lambda x : tf.parse_single_example(x, image_feature_description)
    dataset = rec.map(parse)
    shuf = dataset.shuffle(buffer_size = 50)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    for rec in shuf.take(10):
        ks = rec.keys()
        xmin = tf.sparse.to_dense(rec['image/object/bbox/xmin']).numpy()
        xmax = tf.sparse.to_dense(rec['image/object/bbox/xmax']).numpy()
        ymin = tf.sparse.to_dense(rec['image/object/bbox/ymin']).numpy()
        ymax = tf.sparse.to_dense(rec['image/object/bbox/ymax']).numpy()
        print( rec['image/object/class/text'])
        print (rec['image/object/class/label'])

        jpeg = tf.image.decode_jpeg(rec['image/encoded'])
        img = jpeg.numpy()

        for box in zip(xmin,ymin,xmax,ymax):
            draw_box(img, box)

        cv2.imshow('win', img[...,::-1])
        k = cv2.waitKey(0)

        if k in [ord('q'), 27]:
            break
    cv2.destroyWindow('win')

if __name__ == "__main__":
    main()

