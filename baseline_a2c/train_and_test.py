import os
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
from tianshou.policy import A2CPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.discrete import Critic
from tianshou.utils.net.common import Net
import pickle
from MaskEnvrionment import MedicalEnvrionment
from ActorNet import MyActor
from Collect import MyCollector
from A2C import MyA2CPolicy
from Policy import Myonpolicy_trainer
import time
from Utils import *
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--step-per-epoch', type=int, default=32)
    parser.add_argument('--collect-per-step', type=int, default=128)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=2)
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)

    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    # a2c special
    parser.add_argument('--vf-coef', type=float, default=1.0)
    parser.add_argument('--ent-coef', type=float, default=0.001)
    parser.add_argument('--max-grad-norm', type=float, default=None)
    parser.add_argument('--max_episode_steps', type=int, default=22)
    parser.add_argument('--logpath', type=str, default='a2c/')
    return parser.parse_args()

def save_fn(policy, save_model='./model/ehr'):
    if not os.path.exists(save_model):
        os.makedirs(save_model)
    torch.save(policy, os.path.join(save_model, 'policy.pth'))


def test_a2c(args=get_args()):
    slot_set = []
    with open('./dataset/slot_set.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            slot_set.append(line.strip())
    # slot_set =
    goals = {}
    with open('./dataset/train.pk', 'rb') as f:
        goals['train'] = pickle.load(f)

    with open('./dataset/dev.pk', 'rb') as f:
        goals['dev'] = pickle.load(f)

    total_disease = []
    with open('./dataset/disease.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            total_disease.append(line.strip())
    print(len(slot_set), slot_set)
    disease_num = len(total_disease)

    env = MedicalEnvrionment(slot_set, goals['dev'], disease_num=disease_num)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    train_envs = SubprocVectorEnv(
        [lambda: MedicalEnvrionment(slot_set, goals['train'], max_turn=args.max_episode_steps, flag='train', disease_num=disease_num)
         for _ in range(args.training_num)])


    test_envs = SubprocVectorEnv(
        [lambda: MedicalEnvrionment(slot_set, goals['dev'], max_turn=args.max_episode_steps, flag="dev", disease_num=disease_num)
         for _ in range(args.test_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    random.seed(args.seed)
    # model
    net = Net(args.layer_num, args.state_shape, device=args.device)
    actor_net = Net(args.layer_num, args.state_shape, device=args.device)
    actor = MyActor(actor_net, args.action_shape, disease_num=disease_num).to(args.device)
    critic = Critic(net).to(args.device)
    optim = torch.optim.Adam(list(
        actor.parameters()) + list(critic.parameters()), lr=args.lr)
    dist = torch.distributions.Categorical
    policy = MyA2CPolicy(
        actor, critic, optim, dist, args.gamma, vf_coef=args.vf_coef,
        ent_coef=args.ent_coef, max_grad_norm=args.max_grad_norm)
    # collector
    train_collector = MyCollector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = MyCollector(policy, test_envs)
    # log
    time_name = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    writer = SummaryWriter(os.path.join(args.logdir, args.logpath+time_name))

    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        else:
            return False

    result = Myonpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.repeat_per_collect,
        len(goals['dev']), args.batch_size, writer=writer, save_fn=save_fn)

    return result


def reload(args=get_args()):
    slot_set = []
    with open('./dataset/slot_set.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            slot_set.append(line.strip())
    # slot_set =
    goals = {}
    with open('./dataset/test.pk', 'rb') as f:
        goals['test'] = pickle.load(f)

    for dic in goals['test']:
        dic['disease_tag'] = 'Esophagitis'

    total_disease = []
    with open('./dataset/disease.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            total_disease.append(line.strip())
    print(len(slot_set), slot_set)
    disease_num = len(total_disease)

    env = MedicalEnvrionment(slot_set, goals['test'], disease_num=disease_num)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    test_envs = SubprocVectorEnv(
        [lambda: MedicalEnvrionment(slot_set, goals['test'], max_turn=args.max_episode_steps, flag="test", disease_num=disease_num)
         for _ in range(args.test_num)])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)
    random.seed(args.seed)
    policy = torch.load('./model/ehr/policy.pth')
    test_collector = MyCollector(policy, test_envs)
    result = test_episode(policy, test_collector, test_fn=None, epoch=1,
                 n_episode=len(goals['test']), writer=None)

    return result

if __name__ == '__main__':
    test_a2c()
    reload()