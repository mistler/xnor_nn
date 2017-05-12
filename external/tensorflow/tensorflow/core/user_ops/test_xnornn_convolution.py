import tensorflow as tf

class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    xnornn_convolution_module = tf.load_op_library('external/tensorflow/xnornn_convolution.so')
    with self.test_session():
      result = xnornn_convolution_module.xnornn_convolution([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [0, 1, 2, 3, 4])

if __name__ == "__main__":
  tf.test.main()
