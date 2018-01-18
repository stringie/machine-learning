#%%
import tensorflow as tf
import numpy as np
import random
import gym
import sys
import os

path = "/home/string/dev/machine-learning"
if path not in sys.path:
    sys.path.append(path)

# from envs import plotting
from collections import namedtuple, deque

#%%
env = gym.make("Breakout-v0")
#%%
# Actions for Breakout are [Noop, Fire, Left, Right]
VALID_ACTIONS = [0,1,2,3]
#%%
class StateProcessor(){
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210,160,3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_states)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(self.output, 84, 84, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state){
        return sess.run(self.output, feed_dict={self.input_state: state})
    }
}
#%%
class Estimator():
    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        self.summary_writer = None
        with tf.variable_scope(scope):
            self.build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.train.SummaryWriter(summary_dir)
    
    def build_model(self):
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(X_pl) / 255.0

        batch_size = tf.shape(self.X_pl)[0]
        self.predictions = tf.zeros(shape=[batch_size, len(VALID_ACTIONS)])
        self.loss = tf.constant(0.0)
        self.train_op = tf.no_op("train_pp")

        self.summaries = tf.merge_summary([
            tf.scalar_summary("loss", self.loss)
        ])

    def predict(self, sess, s):
        return sess.run(self.predictions, feed_dict={self.X_pl: s})

    def update(self, sess, s, a, y):
        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a}
        summaries, global_time_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict
        )
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_time_step)
        return loss

#%%
def copy_model_parameters(sess, estimator1, estimator2):
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)

