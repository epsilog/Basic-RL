import abc
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import createMLP
from .base import Base, Module, Distribution
from .distribution import Normal, TanhNormal

LOG_STD_MIN = -10 # stddev의 최소값을 e^-10 = 0.00004로
LOG_STD_MAX = 2 # stddev의 최대값을 e^2 = 7.38 로


class MLP:
    def __init__(self, dInputs:List[int], dOutputs:List[int], dHidden:int, nHidden:int):
        super().__init__()
        self.dInputs, self.dOutputs = dInputs, dOutputs
        self.dInput, self.dOutput = sum(dInputs), sum(dOutputs)
        self.dHidden, self.nHidden = dHidden, nHidden
        self.layers = createMLP(dIn=self.dInput, dHidden=dHidden, nHidden=nHidden, dOut=self.dOutput)

    def _validate_input(self, *args) -> None:
        """check input dimension"""
        assert all([dInput == arg.shape[-1] for dInput, arg in zip(self.dInputs, args)])

    def forward(self, *args): # (dBatch, -)
        self._validate_input(*args)
        if len(self.dInputs) != 1: # multiple input인 경우 concat
            feature = torch.cat(args, dim=-1)
        else:
            feature = args[0]
        out = self.layers(feature)
        if len(self.dOutputs) != 1: # multiple output인 경우 split
            out = out.split(self.dOutputs, dim=-1)
        return out # (dBatch, -)
    
    @torch.no_grad()
    @Module.transform_data(2, use_args=True) # NOTE: add batch dimension, input ndim=2
    def _evaluation(self, *args): # ([dBatch], -)
        return self.forward(*args) # ([dBatch], -)
    
class StochasticPolicy(MLP):
    def __init__(self, dOutputs:List[int], dist:Distribution, action_space:dict, 
                 log_std_min:int, log_std_max:int, *args, **kwargs):
        dOutputs = [dOutput*2 for dOutput in dOutputs]
        super().__init__(*args, dOutputs=dOutputs, **kwargs)
        self.register_buffer(
            "action_scale", torch.tensor( (action_space["high"] - action_space["low"]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor( (action_space["high"] + action_space["low"]) / 2.0, dtype=torch.float32)
        )
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.dist = dist
    
    def getAction(self, *args, deterministic=False, log_prob=False):
        piDist = self.forward(*args)
        action = piDist.mean if deterministic else piDist.rsample()
        # NOTE: log_prob의 sum(dim=-1) 부분 주의
        logPi = piDist.log_prob(action).sum(dim=-1, keepdim=True) if log_prob else None
        return action, logPi, piDist
    
    @torch.no_grad()
    @Module.transform_data(2, use_args=True)
    def _evaluation(self, *args, deterministic=True):
        action, _logPi, _piDist = self.getAction(*args, deterministic=deterministic, log_prob=False)
        return action


# NOTE: V(s)
#       input : (dBatch, dState)
#       output: (dBatch, 1)
class Vnet(MLP, Module):
    @Base.save_input()
    def __init__(self, dState:int, dHidden:int, nHidden:int):
        super().__init__(dInputs=[dState], dOutputs=[1], dHidden=dHidden, nHidden=nHidden)
        self.dState = dState

# NOTE: Q(s, a)
#       input : (dBatch, dState), (dBatch, dAction)
#       output: (dBatch, 1)
class Qnet(MLP, Module):
    @Base.save_input()
    def __init__(self, dState:int, dAction:int, dHidden:int, nHidden:int):
        super().__init__(dInputs=[dState, dAction], dOutputs=[1], dHidden=dHidden, nHidden=nHidden)
        self.dState = dState
        self.dAction = dAction

# NOTE: Pi(a | s) ~ Deterministic
#       input : (dBatch, dState)
#       output: (dBatch, dAction)
class Policy(MLP, Module):
    @Base.save_input()
    def __init__(self, dState:int, dAction:int, dHidden:int, nHidden:int):
        super().__init__(dInputs=[dState], dOutputs=[dAction], dHidden=dHidden, nHidden=nHidden)
        self.dState = dState
        self.dAction = dAction
        
# NOTE: Pi(a | s) ~ Normal(tanh(mu), sigma)
#       input : (dBatch, dState)
#       output: (dBatch, dAction)~Normal(tanh(mu), sigma)
#               mu   : (-1, 1), 
#               sigma: [e^-10, e^2]=[0.00004, 7.38] - hard clamp
class TruncatedNormalPolicy(StochasticPolicy, Module):
    @Base.save_input()
    def __init__(self, dState:int, dAction:int, dHidden:int, nHidden:int, action_space:dict, 
                 log_std_min:int=LOG_STD_MIN, log_std_max:int=LOG_STD_MAX):
        super().__init__(dInputs=[dState], dOutputs=[dAction], dHidden=dHidden, nHidden=nHidden,
                         action_space=action_space, log_std_min=log_std_min, log_std_max=log_std_max,
                         dist=Normal)
        self.dState, self.dAction = dState, dAction
        
    def forward(self, s):
        out = super().forward(s)
        mean, logStd = out.chunk(2, dim=-1)
        mean = mean.tanh() * self.action_scale + self.action_bias
        logStd = logStd.clamp(min=self.log_std_min, max=self.log_std_max)
        return self.dist(mean, logStd.exp())

# NOTE: Pi(a | s) ~ Normal(tanh(mu), tanh(sigma))
#       input : (dBatch, dState)
#       output: (dBatch, dAction)~Normal(tanh(mu), tanh(sigma))
#               mu   : (-1, 1)
#               sigma: [e^-10, e^2]=[0.00004, 7.38] - soft clamp
class NormalPolicy(StochasticPolicy, Module):
    @Base.save_input()
    def __init__(self, dState:int, dAction:int, dHidden:int, nHidden:int, action_space:dict,
                 log_std_min:int=LOG_STD_MIN, log_std_max:int=LOG_STD_MAX):
        super().__init__(dInputs=[dState], dOutputs=[dAction], dHidden=dHidden, nHidden=nHidden,
                         action_space=action_space, log_std_min=log_std_min, log_std_max=log_std_max,
                         dist=Normal)
        self.dState, self.dAction = dState, dAction
        
    def forward(self, s):
        out = super().forward(s)
        mean, logStd = out.chunk(2, dim=-1)
        mean = mean.tanh() * self.action_scale + self.action_bias
        logStd = self.log_std_min + 0.5 * (logStd.tanh() + 1) * (self.log_std_max-self.log_std_min)
        return self.dist(mean, logStd.exp())

# NOTE: Pi(a | s) ~ TanhNormal(mu, sigma)
#       input : (dBatch, dState)
#       output: (dBatch, dAction)~TanhNormal(mu, sigma)
#               mu   : (-1, 1)
#               sigma: [e^-10, e^2]=[0.00004, 7.38] - hard clamp
class TruncatedTanhNormalPolicy(StochasticPolicy, Module):
    @Base.save_input()
    def __init__(self, dState:int, dAction:int, dHidden:int, nHidden:int, action_space:dict,
                 log_std_min:int=LOG_STD_MIN, log_std_max:int=LOG_STD_MAX):
        super().__init__(dInputs=[dState], dOutputs=[dAction], dHidden=dHidden, nHidden=nHidden,
                         action_space=action_space, log_std_min=log_std_min, log_std_max=log_std_max,
                         dist=TanhNormal)
        self.dState, self.dAction = dState, dAction
        
    def forward(self, s):
        out = super().forward(s)
        mean, logStd = out.chunk(2, dim=-1)
        logStd = logStd.clamp(min=self.log_std_min, max=self.log_std_max)
        return TanhNormal(mean, logStd.exp())
    
    def getAction(self, *args, deterministic=False, log_prob=False):
        # NOTE: TanhNormal에서 직접 log_prob 계산 시 성능 망가짐(HalfCheetah에서 확인함)
        #       action이 -1 또는 1 경계에 가까워지면 tanh값이 -inf, inf로 발산하기 때문
        piDist = self.forward(*args)
        out = piDist.normal.mean if deterministic else piDist.normal.rsample()
        action = out.tanh() * self.action_scale + self.action_bias
        logPi = piDist.log_prob_from_pre_tanh(out) if log_prob else None
        return action, logPi, piDist
        
# NOTE: Pi(a | s) ~ TanhNormal(mu, tanh(sigma))
#       input : (dBatch, dState)
#       output: (dBatch, dAction)~TanhNormal(mu, sigma)
#               mu   : (-1, 1)
#               sigma: [e^-10, e^2]=[0.00004, 7.38] - soft clamp
class TanhNormalPolicy(StochasticPolicy, Module):
    @Base.save_input()
    def __init__(self, dState:int, dAction:int, dHidden:int, nHidden:int, action_space:dict,
                 log_std_min:int=LOG_STD_MIN, log_std_max:int=LOG_STD_MAX):
        super().__init__(dInputs=[dState], dOutputs=[dAction], dHidden=dHidden, nHidden=nHidden,
                         action_space=action_space, log_std_min=log_std_min, log_std_max=log_std_max,
                         dist=TanhNormal)
        self.dState, self.dAction = dState, dAction
        
    def forward(self, s):
        out = super().forward(s)
        mean, logStd = out.chunk(2, dim=-1)
        logStd = self.log_std_min + 0.5 * (logStd.tanh() + 1) * (self.log_std_max-self.log_std_min)
        return TanhNormal(mean, logStd.exp())
    
    def getAction(self, *args, deterministic=False, log_prob=False):
        # NOTE: TanhNormal에서 직접 log_prob 계산 시 성능 망가짐(HalfCheetah에서 확인함)
        #       action이 -1 또는 1 경계에 가까워지면 tanh값이 -inf, inf로 발산하기 때문
        piDist = self.forward(*args)
        out = piDist.normal.mean if deterministic else piDist.normal.rsample()
        action = out.tanh() * self.action_scale + self.action_bias
        logPi = piDist.log_prob_from_pre_tanh(out) if log_prob else None
        return action, logPi, piDist
        
        

# class AWACPolicy(MLP, Module):
#     def __init__(self, dState, dAction, dHidden, nHidden, action_space:dict, log_std_min=-6, log_std_max=0, **kwargs):
#         super().__init__(dInputs=[dState], dOutputs=[dAction], dHidden=dHidden, nHidden=nHidden)
#         self.register_buffer(
#             "action_scale", torch.tensor( (action_space["high"] - action_space["low"]) / 2.0, dtype=torch.float32)
#         )
#         self.register_buffer(
#             "action_bias", torch.tensor( (action_space["high"] + action_space["low"]) / 2.0, dtype=torch.float32)
#         )
#         self.dState, self.dAction = dState, dAction
#         self.logStd = nn.Parameter(torch.zeros(dAction, requires_grad=True)) # (dAction)
#         self.log_std_min, self.log_std_max = log_std_min, log_std_max
        
#     def forward(self, *args):
#         mu = super().forward(*args)
#         mu = mu.tanh() # (dBatch, dAction)
        
#         logStd = torch.sigmoid(self.logStd) # (dAction)
#         logStd = (self.log_std_max - self.log_std_min)*logStd + self.log_std_min
#         return Normal(mu, logStd.exp())
    
#     def getAction(self, *args, deterministic=False, log_prob=False):
#         piDist = self.forward(*args)
#         action = piDist.mean if deterministic else piDist.rsample()
#         logPi = piDist.log_prob(action).sum(dim=-1, keepdim=True) if log_prob else None
#         return action, logPi, piDist
        
#     @torch.no_grad()
#     @Module.transform_data(2, use_args=True)
#     def _evaluation(self, *args, deterministic=True):
#         action, _logPi, _piDist = self.getAction(*args, deterministic=deterministic, log_prob=False)
#         return action


# class FixStddevAWACPolicy(MLP, Module):
#     def __init__(self, dState, dAction, dHidden, nHidden, action_space:dict, stddev, **kwargs):
#         super().__init__(dInputs=[dState], dOutputs=[dAction], dHidden=dHidden, nHidden=nHidden)
#         self.register_buffer(
#             "action_scale", torch.tensor( (action_space["high"] - action_space["low"]) / 2.0, dtype=torch.float32)
#         )
#         self.register_buffer(
#             "action_bias", torch.tensor( (action_space["high"] + action_space["low"]) / 2.0, dtype=torch.float32)
#         )
#         self.dState, self.dAction = dState, dAction
#         self.std = stddev
#         # self.logStd = nn.Parameter(torch.zeros(dAction, requires_grad=True)) # (dAction)
#         # self.log_std_min, self.log_std_max = log_std_min, log_std_max
        
#     def forward(self, *args):
#         mu = super().forward(*args)
#         mu = mu.tanh() # (dBatch, dAction)
        
#         # logStd = torch.sigmoid(self.logStd) # (dAction)
#         # logStd = (self.log_std_max - self.log_std_min)*logStd + self.log_std_min
#         # return Normal(mu, logStd.exp())
#         return Normal(mu, self.std)
    
#     def getAction(self, *args, deterministic=False, log_prob=False):
#         piDist = self.forward(*args)
#         action = piDist.mean if deterministic else piDist.rsample()
#         logPi = piDist.log_prob(action).sum(dim=-1, keepdim=True) if log_prob else None
#         return action, logPi, piDist 
        
#     @torch.no_grad()
#     @Module.transform_data(2, use_args=True)
#     def _evaluation(self, *args, deterministic=True):
#         action, _logPi, _piDist = self.getAction(*args, deterministic=deterministic, log_prob=False)
#         return action