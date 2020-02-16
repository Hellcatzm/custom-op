import tensorflow as tf


rok_module = tf.load_op_library('bazel-bin/tensorflow_rok/_rok_ops.so')
zero_out_ops = tf.load_op_library('bazel-bin/tensorflow_zero_out/python/ops/_zero_out_ops.so')
maximum_filter_ops = tf.load_op_library('bazel-bin/tensorflow_maximum_filter/_maximum_filter_ops.so')

@tf.function
def test(hp, coo):
    rok = rok_module.ROK(images=hp, coords=coo)
    zo = zero_out_ops.zero_out([[1, 2], [3, 4]])
    print(rok, zo)
    return rok

with tf.device("/gpu:0"):
    img = tf.cast(tf.reshape(tf.range(0, 2*256*256), [2, 256, 256]), dtype=tf.float32)
    coo = tf.constant([[0, 0, 1], [1, 50, 100], [2, 100, 200]])
    rok = test(img, coo)
    hp = tf.constant([[[0,0,0,0,0],[0,0,1,0,0],[0,1,2,1,0],[0,0,1,0,0],[0,0,0,0,0]]])
    mo = maximum_filter_ops.maximum_filter(hp, footprint=tf.constant([[0,1,0],[1,1,1],[0,1,0]]))
    print(mo)