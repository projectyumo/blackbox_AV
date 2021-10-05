import numpy as np
import cv2

import gym
import torch
import torch.nn as nn

from models import Agent

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def rgb2gray(rgb, norm=True):
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray

def preprocess_state(state):
    img_gray = rgb2gray(state)
    img_rs = cv2.resize(img_gray, dsize=(state_shape, state_shape), interpolation=cv2.INTER_CUBIC)
    stack = [img_rs] * img_stack

    return np.array(stack)


img_stack = 1
state_shape = 128

if __name__ == "__main__":
    agent = Agent()
    agent.load_param()
    env = gym.make('CarRacing-v0')
    state = env.reset()
    state = preprocess_state(state)

    # state = env.reset()
    score = 0

    for t in range(100000):
        env.render()
        action, _ = agent.select_action(state)
        state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
        score += reward
        state = preprocess_state(state_)
        if done or die:
            break

    print('Score: {:.2f}\t'.format(score))
