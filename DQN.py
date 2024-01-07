import os
import copy
import random
from typing import List
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils.utils import isSet, isFreq, AttrDict
from utils.base import Base, Module, MainModule, Environment, Task
from utils.base import Logger, WandbLogger, TensorboardLogger
from utils.buffer import Buffer
from utils.net import MLP

import warnings # ignore deprecation, user warning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Qnet(MLP, Module):
    @Base.save_input()
    def __init__(self, dState, dAction, dHidden, nHidden) -> None:
        super().__init__(dInputs=[dState], dOutputs=[dAction], dHidden=dHidden, nHidden=nHidden)
        self.dState = dState
        self.dAction = dAction

# discrete action space only
class DQN(MainModule):
    @Base.save_input(exclude=["Q"])
    def __init__(self, conf:AttrDict, envConf:AttrDict, Q:Module) -> None:
        super().__init__()
        self.gamma              = conf.gamma
        self.epsilon            = conf.epsilon_init
        self.epsilon_min        = conf.epsilon_min
        self.epsilon_decay      = (conf.epsilon_init - conf.epsilon_min) / conf.epsilon_decay_step
        self.target_update_freq = conf.target_update_freq
        self.lr_q               = conf.lr_q
        
        self.dState             = envConf.dState
        self.dAction            = envConf.dAction
        
        self.Q = Q
        self.targetQ = copy.deepcopy(self.Q).requires_grad_(False) # freeze mode
        self.optimizerQ = optim.Adam(self.Q.parameters(), lr=self.lr_q)
        
    @torch.no_grad()
    def getAction(self, state, deterministic=False) -> np.ndarray: # (dState)
        if (random.random() > self.epsilon) or deterministic: # exploitation
            state = torch.from_numpy(state).float().to(self.Q.device)
            action = self.Q(state).argmax(dim=-1, keepdim=True).detach().cpu().item() # (1)
        else: # exploration
            action = np.random.randint(low=0, high=self.dAction) # (1)
        return action
    
    def updateEpsilon(self, step:int) -> None:
        # epsilon-greedy linear decay
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        
    def updateTarget(self, step:int) -> None:
        # target Q update
        if step % self.target_update_freq == 0:
            self.targetQ.load_state_dict(self.Q.state_dict())
        
    def update(self, step:int, sample:List[torch.Tensor]) -> AttrDict:
        # backpropagation
        result = self.loss(sample)
        self.optimizerQ.zero_grad()
        result.loss.backward()
        self.optimizerQ.step()
        
        return AttrDict(
            loss=result.loss.detach().cpu().item(),
            Q=result.Q,
        )
        
    def loss(self, sample:List[torch.Tensor]) -> AttrDict:
        state, action, reward, nextState, done = sample # (dBatch, -)
        
        # --- estimate ---
        estimateQ = self.Q(state).gather(dim=-1, index=action.long()) # (dBatch, 1)
        
        # --- target ---
        with torch.no_grad():
            # max_a Q'(s_t+1, a)
            nextTargetQ = self.targetQ(nextState).max(dim=-1, keepdim=True).values # (dBatch, 1)
            # r + gamma * max_a Q'(s_t+1, a)
            targetQ = reward + (1 - done) * self.gamma * nextTargetQ # (dBatch, 1)
            
        # MSE{ targetQ, Q }
        loss = F.mse_loss(targetQ.detach(), estimateQ) # scalar
        return AttrDict(
            loss=loss,
            Q=estimateQ.detach().mean().cpu().item(),
        )


def main():
    # ---------------------------------------
    # --------------- setting ---------------
    # ---------------------------------------
    conf = AttrDict(
        # --- experiment ---
        EXP_NAME            = "v1.0",
        ALGORITHM           = "DQN",
        DEVICE              = "cuda:0" if torch.cuda.is_available() else "cpu", # cpu or cuda:0, cuda:1, etc.
        # --- environment ---
        ENV                 = "CartPole-v1",
        MAX_STEP            = 10000,        # exploration step + train step (time step unit)
        EPISODE_MAX_STEP    = -1,           # defualt: -1, Using environment default setting
        TRUNCATED_DONE      = False,        # default: False
        # --- train option ---
        TEST_FREQ           = 10,           # time step unit
        SAVE_FINAL          = True,         # save last model
        SAVE_FREQ           = -1,           # default: -1, save model freq
        SAVE_PATH           = "./model",    # model save path
        # --- log ---
        TENSORBOARD         = False,        # use tensorboard
        WANDB               = True,         # use wandb
        LOG_FREQ            = 10,           # log freq, is 1, save all
    )

    hyperConf = AttrDict(
        # --- buffer ---
        buffer_size         = 3000,         # transition unit
        # --- Q ---
        n_hidden_q          = 1,            # number of hidden layer
        d_hidden_q          = 128,          # dimension of hidden layer
        lr_q                = 0.001,        # learning rate
        target_update_freq  = 30,           # target Q update freq, hard update
        # --- agent ---
        d_batch             = 128,          # batch size
        train_start         = 1000,         # n_transition
        epsilon_init        = 1.0,          # initial exploration ratio
        epsilon_min         = 0.01,         # min exploration ratio
        epsilon_decay_step  = conf.MAX_STEP - 1000, # linear decay step
        gamma               = 0.99,         # discount rate
    )

    wandbConf = AttrDict(               # when conf.WANDB = True
        logger      = WandbLogger,
        project     = conf.ALGORITHM,       # project name
        group       = conf.ENV,             # (optional) default: None
        exp_name    = f'{conf.ALGORITHM}-{conf.EXP_NAME}',
        save_code   = True,                 # works when .py, not .ipynb
        key         = None,                 # (optional) wandb API key, default: None
        log_freq    = conf.LOG_FREQ,
    )
    tensorboardConf = AttrDict(         # when conf.TENSORBOARD = True
        logger      = TensorboardLogger,
        path        = "./tensorboard",      # tensorboard log folder
        exp_name    = f'{conf.ALGORITHM}-{conf.EXP_NAME}',
        log_freq    = conf.LOG_FREQ,
    )

    # --- create folder ---
    if isSet(conf.SAVE_FREQ) or isSet(conf.SAVE_FINAL):
        os.makedirs(conf.SAVE_PATH, exist_ok=True)
    if isSet(conf.TENSORBOARD):
        os.makedirs(tensorboardConf.path, exist_ok=True)

    # ---------------------------------------
    # ---------------------------------------
    # ---------------------------------------

    # --- initialize environment ---
    env      = Environment(name=conf.ENV, episode_max_step=conf.EPISODE_MAX_STEP, truncated_done=conf.TRUNCATED_DONE)
    test_env = Environment(name=conf.ENV, episode_max_step=conf.EPISODE_MAX_STEP, truncated_done=conf.TRUNCATED_DONE, mode_test=True)
    task = Task(env=env, test_env=test_env)
    envConf = env.getConf()

    # --- initialize buffer ---
    buffer = Buffer(capacity=hyperConf.buffer_size)

    # --- initialize agent ---
    Q = Qnet(
        dState=envConf.dState, dAction=envConf.dAction,
        dHidden=hyperConf.d_hidden_q, nHidden=hyperConf.n_hidden_q,
    )

    agent = DQN(hyperConf, envConf, Q).to(conf.DEVICE)

    # --- initialize logger ---
    logger = Logger(
        config=[
            wandbConf if conf.WANDB else None, 
            tensorboardConf if conf.TENSORBOARD else None],
        hyperparameter=[conf, hyperConf],
    )
    logger.log({ # log intial hyperparameter
        "train/epsilon": agent.epsilon,
        "train/buffer": buffer.n_transition,
        "train/lr": agent.optimizerQ.param_groups[0]["lr"]
    }, step=0)

    # --------------------------------------
    # ---------------- main ----------------
    # --------------------------------------
    task.reset()
    for step in tqdm(range(1, conf.MAX_STEP + 1)):
        # --- environment interaction ---
        state = task.state
        action = agent.getAction(state, deterministic=False)
        nextState, reward, done, truncated, _info = task.step(action)
        buffer.setSample(state, action, reward, nextState, done)
        
        # done or truncated
        if task.isTerminal():
            logger.log({
                "train/episode" : task.n_episode,
                "train/score"   : task.score,
            }, step=step)
            task.reset()
        
        if buffer.n_transition < hyperConf.train_start:
            continue
        
        # --- update model ---
        sample = buffer.getSample(hyperConf.d_batch).to(agent.device)
        result = agent.update(sample)
        logger.log({
            "train/loss_Q"  : result.loss,
            "train/Q"       : result.Q,
        }, step=step, mode_freq=True)
            
        # --- update hyperparameter ---
        agent.updateEpsilon(step)
        agent.updateTarget(step)
        logger.log({
            "train/epsilon" : agent.epsilon,
            "train/buffer"  : buffer.n_transition,
            "train/lr"      : agent.optimizerQ.param_groups[0]["lr"],
        }, step=step, mode_freq=True)
        
        # --- model test ---
        if isFreq(step, conf.TEST_FREQ):
            with task.test_mode(): # env <-> test_env
                task.reset()
                while True:
                    action = agent.getAction(task.state, deterministic=True)
                    task.step(action)
                    if task.isTerminal():
                        break
                result = task.score # NOTE: must run inside a 'with' statement
            logger.log({"test/score": result}, step=step)
        
        # --- model save ---
        if isFreq(step, conf.SAVE_FREQ) or (isSet(conf.SAVE_FINAL) and step == conf.MAX_STEP):
            Q.save(path=f'{conf.SAVE_PATH}/{conf.ALGORITHM}-{conf.EXP_NAME}-step_{step}.pt')
        
    logger.close()
    print("---done---")
    
if __name__ == "__main__":
    main()
