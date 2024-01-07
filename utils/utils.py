# import warnings # warning ignore
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import torch
import gym

CLASSIC = "CLASSIC"
BOX2D = "BOX2D"
D4RL = "D4RL"
MUJOCO = "MUJOCO"
ATARI = "ATARI"

# NOTE: AttrDict는 copy.deepcopy 사용시 문제가 있음
class AttrDict(dict):
    """To store hyperparameters"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def isSet(condition):
    """ 
    condition == None, False, -1 -> False
    """
    if condition is None:
        return False
    elif isinstance(condition, bool):
        return condition
    else:
        return condition != -1
    
def isLow(step, condition):     return isSet(condition) and (step < condition)
def isLowEq(step, condition):   return isSet(condition) and (step <= condition)
def isHigh(step, condition):    return isSet(condition) and (step > condition)
def isHighEq(step, condition):  return isSet(condition) and (step >= condition)
def isFreq(step, condition):    return isSet(condition) and (step % condition == 0)

def getEnvType(env):
    """ environment 종류 반환 """
    name = env if isinstance(env, str) else env.spec.id
    
    if any([envName in name for envName in ["CartPole", "Acrobot", "MountainCar", "Pendulum"]]):
        return CLASSIC
    elif any([envName in name for envName in ["LunarLander", "BipedalWalker", "CarRacing"]]):
        return BOX2D
    elif any([envName in name for envName in ["maze2d", "antmaze",  "minigrid",     "pen",       "hammer", 
                                              "door",   "relocate", "halfcheetah-", "walker2d-", "hopper-",
                                              "ant-",   "flow-",    "kitchen-",     "carla-"]]):
        return D4RL
    elif any([envName in name for envName in ["Ant",              "HalfCheetah", "Hopper",  "Humanoid", "InvertedDoublePendulum", 
                                              "InvertedPendulum", "Reacher",     "Swimmer", "Walker2d"]]):
        return MUJOCO
    elif any([envName in name for envName in ["Adventure",    "AirRaid",      "Alien",        "Amidar",           "Assault", 
                                              "Asterix",      "Asteriods",    "Atlantis",     "BankHeist",        "Bowling",
                                              "Boxing",       "Breakout",     "Carnival",     "Centipede",        "ChopperCommand",
                                              "CrazyClimber", "Defender",     "DemonAttack",  "DoubleDunk",       "ElevatorAction",
                                              "Enduro",       "FishingDerby", "Freeway",      "Frostbite",        "Gopher",
                                              "Gravitar",     "Hero",         "IceHockey",    "Jamesbond",        "JourneyEscape",
                                              "Kangaroo",     "Krull",        "KungFuMaster", "MontezumaRevenge", "MsPacman",
                                              "NameThisGame", "Phoenix",      "Pitfall",      "Pong",             "Pooyan",
                                              "PrivateEye",   "Qbert",        "Riverraid",    "RoadRunner",       "Robotank",
                                              "Seaquest",     "Skiing",       "Solaris",      "SpaceInvaders",    "StarGunner",
                                              "Tennis",       "TimePilot",    "Tutankham",    "UpNDown",          "Venture",
                                              "VideoPinball", "WizardOfWar",  "YarsRevenge",  "Zaxxon"]]):
        return ATARI
    else:
        raise NotImplementedError("Unknown Environment!")

def getEnvConf(env) -> AttrDict:
    """ env의 dState, dAction 반환 """
    if isinstance(env.observation_space, gym.spaces.Box): # continuous state space
        observation_space = dict(high=env.observation_space.high, low=env.observation_space.low)
        dState = env.observation_space.shape[0]
    else: # discrete state space
        observation_space = None
        dState = env.observation_space.n
        
    if isinstance(env.action_space, gym.spaces.Box): # continuous action space
        action_space = dict(high=env.action_space.high, low=env.action_space.low)
        dAction = env.action_space.shape[0] if env.action_space.shape else env.action_space.n
    else: # discrete action space
        action_space = None
        dAction = env.action_space.n
    return AttrDict(dState=dState, dAction=dAction, observation_space=observation_space, action_space=action_space)

def makeEnv(name:str, mode_render:bool=False, mode_test:bool=False, time_limit:bool=True, **kwargs) -> gym.Env:
    """create environment"""
    env_type = getEnvType(name)
    if mode_render:
        kwargs.update({"render_mode":"rgb_array"})
        
    if env_type in [CLASSIC, BOX2D, MUJOCO]:
        env = gym.make(name, **kwargs)
    elif env_type == D4RL:
        # NOTE: remove import error print
        import sys, os
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = open(os.devnull, "w"), open(os.devnull, "w")
        try:
            import d4rl
            env = gym.make(name, **kwargs)
        except:
            sys.stdout, sys.stderr = stdout, stderr
            print("D4RL import error")
        finally:
            sys.stdout, sys.stderr = stdout, stderr
    elif env_type == ATARI:
        # from .utils.atari_warpper import AtariWrapper
        from .atari_wrapper import AtariWrapper
        env = gym.make(name, **kwargs)
        if isinstance(env, gym.wrappers.TimeLimit) and not time_limit:
            env = env.env # remove TimeLimit wrapper
            
        if mode_test:
            env = AtariWrapper(env, clip_reward=False)
        else:
            env = AtariWrapper(env)
    else:
        raise NotImplementedError("Unknown Environment")
    
    # NOTE: 신버전 gym에서는 문제가 없지만, 구버전 gym에서는 time limit 도달 시 
    #       done=True가 되어 버리는 문제가 있다. 반면 신버전 gym에서는 time limit 도달 시
    #       truncated=True가 되고 done은 영향이 없음
    #       따라서 time limit 클래스를 제거하고, Environmnet class에서 직접 다룬다
    if isinstance(env, gym.wrappers.TimeLimit) and not time_limit:
        env = env.env
    
    return env

def makeInput(x, dim:int, device=None) -> torch.Tensor: # ([dBatch], -)
    """ batch dim 추가 및 network의 input으로 변환 
        device=None 인 경우 device 이동하지 않음
    """
    assert x.ndim >= dim - 1
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.ndim == dim-1:
        x = x.unsqueeze(dim=0) # (1, -)
    if (device is not None) and (x.device != torch.device(device)):
        x = x.to(device)
    return x # (dBatch, -)

def makeOutput(x:torch.Tensor, device=None) -> torch.Tensor: # (dBatch, -)
    """batch dim==1 인 경우 batch dim을 제거 및 device를 지정된 device로 이동
        device=None 인 경우 device 이동하지 않음
    """
    if x.shape[0] == 1:
        x = x.squeeze(dim=0) # (-)
    if (device is not None) and (x.device != torch.device(device)):
        x = x.to(device)
    return x # ([dBatch], -)

# NOTE: state_dict으로 구현할 경우, parameter가 아닌 고정된 값도 업데이트 되어 버림
def softUpdateParam(netA:torch.nn.Module, netB:torch.nn.Module, tau:float):
    """ 
        tau의 비율로 netA의 parameter를 netB의 parameter로 update
        parameters()를 사용하므로 BatchNorm의 running_mean, running_var 
        또는 buffer는 포함 안됨을 주의
    """
    for paramA, paramB in zip(netA.parameters(), netB.parameters()):
        paramA.data = tau*paramB.detach().clone() + (1-tau)*paramA.data
        
def hardUpdateParam(netA:torch.nn.Module, netB:torch.nn.Module, strict:bool=False):
    """ netA의 parameter를 netB로 update """
    netA.load_state_dict(netB.state_dict(), strict=strict)

def createMLP(dIn:int, dHidden:int, nHidden:int, dOut:int, leakyrelu:bool=False, leakyrelu_slope:float=0.01, batchnorm:bool=False):
    """ create MLP model """
    activation = torch.nn.LeakyReLU(leakyrelu_slope) if leakyrelu else torch.nn.ReLU()
    
    layers = [torch.nn.Linear(dIn, dHidden)] # input layer
    layers.append(activation)
    for _ in range(nHidden): # hidden layer
        if batchnorm:
            layers.append(torch.nn.Linear(dHidden, dHidden, bias=False)) # batchnorm이 bias 역할을 수행하므로
            layers.append(torch.nn.BatchNorm1d(dHidden))
            # layers.append(activation(negative_slope=leakyrelu_slope))
            layers.append(activation)
        else:
            layers.append(torch.nn.Linear(dHidden, dHidden))
            layers.append(activation)
    # if batchnorm:
    #     layers.append(torch.nn.BatchNorm1d(dHidden))
    # layers.append(activation)
    layers.append(torch.nn.Linear(dHidden, dOut)) # output layer
    
    return torch.nn.Sequential(*layers)
