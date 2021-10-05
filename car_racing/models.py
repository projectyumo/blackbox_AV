import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import math
import numpy as np

img_stack = 1
state_shape = 128
gamma = 0.99
device = torch.device("cuda")

transition = np.dtype([('s', np.float64, (img_stack, state_shape, state_shape)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (img_stack, state_shape, state_shape))])

class CNN(nn.Module):
    def __init__(self, nc, nfm, out_dim=1, img_size=256):
        super(CNN, self).__init__()

        exp = int( math.log(img_size)/math.log(2) )

        self.cnn = [nn.Conv2d(nc, nfm, 4, 2, 1),
                    nn.ReLU()]

        for i in range(exp-3):
          self.cnn += [nn.Conv2d( nfm*(2**i) , nfm*( 2**(i+1) ), 4, 2, 1),
                       nn.BatchNorm2d(nfm*( 2**(i+1) )),
                       nn.ReLU()]

        self.cnn += [nn.Conv2d( nfm*( 2**(exp-3) ) , out_dim, 4, 1, 0)]

        self.cnn = nn.Sequential(*self.cnn)

    def forward(self, inputs):
        return self.cnn(inputs)

class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, nc=1, nfm=8, cnn_out=256, out_dim=100, img_size=128):
        super(Net, self).__init__()
        self.cnn_out = cnn_out
        self.cnn = CNN(nc, nfm, cnn_out, img_size)

        self.v = nn.Sequential(nn.ReLU(), nn.Linear(cnn_out, out_dim), nn.ReLU(), nn.Linear(out_dim, 1))
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(cnn_out, out_dim), nn.ReLU())

        self.alpha_head = nn.Sequential(nn.Linear(out_dim, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(out_dim, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, self.cnn_out)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v

class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self):
        self.training_step = 0
        self.net = Net().double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def save_param(self):
        torch.save(self.net.state_dict(), 'param/ppo_net_params.pkl')

    def load_param(self):
        self.net.load_state_dict(torch.load('param/ppo_net_params.pkl'))

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
