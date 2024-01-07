import abc
import numpy as np
import torch
from contextlib import contextmanager
from typing import Tuple, List, Dict, Union

from .utils import isSet, isLow, isLowEq, isHigh, isHighEq, isFreq, AttrDict
from .utils import makeEnv, makeInput, makeOutput, getEnvConf
from .distribution import Distribution

# class MathDict(AttrDict):
#     """ 간단한 결과값 계산 시 사용 """
#     def store(self, items:dict):
#         for name, item in items.items():
#             if self.get(name) is None:
#                 self[name] = [item]
#             else:
#                 self[name].append(item)
                
#     def mean(self):
#         for name, item in self.items():
#             self[name] = sum(self[name]) / len(self[name])
#         return self

# ------ module class ------

class ToDeviceMixin:
    """ device 이동 및 표기 제공"""
    _device = torch.device("cpu")
    
    @property
    def device(self):
        return self._device
    
    def _apply(self, fn):
        """ to(), cuda(), cpu() 등 method 호출시 내부적으로 _apply method를 호출함
            이때 위 method들은 variable로 "t"이름을 사용하므로(e.g. convert(t))
            이를 통해 device 이동 관련 method들을 구분한다.
            
            이 method는 self._modules에 포함되는 하위 모듈에서 다시 호출되므로 
            현재 instance에 대해서만 고려해서 작성하면 된다.
        """
        if "t" in fn.__code__.co_varnames:
            empty = torch.empty(0)
            device = fn(empty).device
            self._device = device
            
        # NOTE: module 내 distribution class 변수 존재 시 함께 이동
        for name, var in vars(self).items():
            if isinstance(var, Distribution):
                setattr(self, name, var.to(self._device))
        
        return super()._apply(fn)

class FreezeMixin:
    """module의 requires_grad on/off 제공"""
    _freeze_mode = False
    
    def requires_grad_(self, requires_grad:bool=True):
        self._freeze_mode = not requires_grad # NOTE: _freeze_mode와 requires_grad_는 서로 반대
        return super().requires_grad_(requires_grad)
    
    @contextmanager
    def freeze_mode(self):
        prev_mode = self._freeze_mode
        self._freeze_mode = True
        self.requires_grad_(False)
        yield
        self._freeze_mode = prev_mode
        self.requires_grad_(not prev_mode)
    
    @contextmanager
    def unfreeze_mode(self):
        prev_mode = self._freeze_mode
        self._freeze_mode = False
        self.requires_grad_(True)
        yield
        self._freeze_mode = prev_mode
        self.requires_grad_(not prev_mode)
    
    @staticmethod
    @contextmanager
    def freeze_modes(*models):
        prev_modes = []
        for i, model in enumerate(models):
            prev_modes.append(model._freeze_mode)
            model._freeze_mode = True
            model.requires_grad_(False)
        yield
        for i, model in enumerate(models):
            model._freeze_mode = prev_modes[i]
            model.requires_grad_(not prev_modes[i])
    
    @staticmethod
    @contextmanager
    def unfreeze_modes(*models):
        prev_modes = []
        for i, model in enumerate(models):
            prev_modes.append(model._freeze_mode)
            model._freeze_mode = False
            model.requires_grad_(True)
        yield
        for i, model in enumerate(models):
            model._freeze_mode = prev_modes[i]
            model.requires_grad_(not prev_modes[i])

class SaveMixin:
    """model의 parameter와 input을 포함하여 모두 저장 및 불러오기 기능 제공"""
    _input_data = None

    def export(self) -> dict:
        """ model의 parameter와 _input_data를 dict로 반환
            return e.g. 
                {"parameter": model.state_dict(), 
                 "input": model._input_data}
        """
        data = dict()
        data["parameter"] = self.state_dict()
        if self._input_data is not None:
            # NOTE: _input_data에 AttrDict 객체가 존재 할 시 dict로 변환(저장 시 AttrDict 클래스 의존성 제거)
            input_kwargs = {k:dict(v) if isinstance(v, AttrDict) else v for k, v in self._input_data.items()}
            data["input"] = input_kwargs
        return data
    
    def save(self, path:str) -> None:
        """단일 모델 저장"""
        torch.save(self.export(), path)

    @staticmethod
    def save_input(include=[], exclude=[]):
        """decorator: method의 input을 _input_data에 전부 저장
            e.g. 
                @ save_inputs()
                def __init__(self, a, b)
                
                @save_inputs(include=["asdf"])
                def __init__(self, a, b, asdf)
        """
        if include and exclude:
            raise Exception("Cannot use both options [include, exclude] togather.")
        
        def save_input_decorator(func):
            def wrapper(self, *args, **kwargs):
                # NOTE: 상속 등의 이유로 중복 호출되는 경우 skip
                if self._input_data is None: 
                    # NOTE: 변수 이름과 변수를 dict 형태로 만들어서 저장
                    varNames = list(func.__code__.co_varnames[1:]) # self 제거
                    saveKwargs = kwargs.copy() # 입력 kwargs 추가(kwargs에 아이템이 추가되는것을 방지하기 위해 copy)
                    saveKwargs.update({varName:arg for varName, arg in zip(varNames, args)}) # 입력 args 추가
                    if include:
                        saveKwargs = {varName:arg for varName, arg in saveKwargs.items() if varName in include}
                    elif exclude:
                        saveKwargs = {varName:arg for varName, arg in saveKwargs.items() if varName not in exclude}
                    self._input_data = saveKwargs
                return func(self, *args, **kwargs)
            return wrapper
        return save_input_decorator
    
    @classmethod
    def _from(cls, exported_model:dict, **kwargs):
        """ exported_model e.g.
                {"parameter": ..., "input", ...}
        """
        modelInput = {}
        if "input" in exported_model.keys():
            # NOTE: 모든 상위 dict가 전부 AttrDict로 변함에 주의(AttrDict가 dict를 상속하므로 큰 문제 없을 듯)
            modelInput.update( {k:AttrDict(v) if isinstance(v, dict) else v for k, v in exported_model["input"].items()} )
        modelInput.update(kwargs)
        
        model = cls(**modelInput)
        model.load_state_dict(exported_model["parameter"])
        return model

    @classmethod
    def load(cls, path:str, **kwargs):
        """ 저장된 파일로 부터 model 생성 및 반환
            기본적으로 저장된 파일의 "input" value를 사용해서 model을 생성하지만,
            kwargs를 넣어주면 해당 kwargs 값으로 덮어 씌워 생성
            
            load e.g.
                Policy.load(path, etc=10)
        """
        raw = torch.load(path, map_location="cpu")
        return cls._from(raw, **kwargs)
    

class Base(ToDeviceMixin, FreezeMixin, SaveMixin, torch.nn.Module):
    """ model의 parameter와 input을 포함하여 모두 저장 및 불러오기 기능 제공
    """
    @staticmethod
    def saves(path:str, model:dict, **kwargs):
        """ - 다중 모델 저장
            call e.g.
                Base.saves(
                    path=..., 
                    model={
                        "policy": policy,...}, 
                    etc=..., log=...)
            save e.g.
                {   "model": {
                        "policy": {"parameter": ..., "input": ...},}
                    "etc": ..., "log": ... }
        """
        data = dict()
        data["model"] = {name:m.export() for name, m in model.items()}
        data.update(kwargs)
        torch.save(data, path)
        
    @staticmethod
    def loads(path:str, model:dict):
        """ 파일을 불러온 후, model을 통해 instance를 생성해서 불러온 후 반환
            loads(
                path=...,
                model={
                    "Q1": Qnet, "Q2": Qnet,
                    "policy": {"model": SACPolicy, "fixed_model": etc.},
                }
            )
            model e.g.
                {"Q": Qnet, "policy": SACPolicy}
            return e.g.
                {"Q": ..., "policy": ..., etc: ...,}
        """
        raw = torch.load(path, map_location="cpu")
        raw_model = raw.pop("model")
        
        out = {}
        for name, modelInfo in model.items():
            if isinstance(modelInfo, dict):
                out[name] = modelInfo["model"]._from(
                    exported_model=raw_model[name],
                    **{k:v for k, v in modelInfo.items() if k != "model"}
                )
            else: # modelInfo == Class
                out[name] = modelInfo._from(raw_model[name])
        if raw: # NOTE: 저장된 파일에 추가적인 item이 들어 있는 경우
            out.update(raw)
        return out
            
    # for debug
    # TODO: device 이동 객체, eval 영향 받는 객체 등 표시하는 기능 추가
    def showParameters(self):
        """parameter의 이름 반환"""
        names = list(set([n.split(".")[0] for n in self.state_dict().keys()]))
        # for name, variable in vars(self).items():
        #     if isinstance(variable, Distribution):
        #         names.append(name)
        names.sort()
        print(names)
    

# 작은 module 작성 시 사용 e.g. Q, Pi
class Module(abc.ABC, Base):
    _evaluation_mode = False
    
    @abc.abstractmethod
    def forward():
        """
            train 시 사용
            1. input은 batch 단위만 허용
            2. stochastic 모델인 경우 distribution을 출력
        """
        pass
    
    @abc.abstractmethod
    def _evaluation(self, x, deterministic=True):
        """
            eval 시 사용
            1. input에 batch 차원이 없으면 자동 추가
            2. batch가 1인경우 output의 batch dimension 제거
            3. input의 device는 모델의 device로 자동 변경
            4. stochastic 모델도 sample을 출력 (따라서 deterministic 매개변수로 sample 방법 결정)
        """
        pass

    def __call__(self, *args, **kwargs):
        if self._evaluation_mode:
            return self._evaluation(*args, **kwargs)
        else:
            return self.forward(*args, **kwargs)
        
    @contextmanager
    def evaluation_mode(self):
        """ __call__ = forward <-> _evaluation """
        prev_evaluation = self._evaluation_mode
        prev_training = self.training
        self._evaluation_mode = True
        if prev_training:
            self.eval()
        yield
        self._evaluation_mode = prev_evaluation
        if prev_training:
            self.train()
    
    @staticmethod
    @contextmanager
    def evaluation_modes(*models):
        """ 여러 model을 동시에 evaluation mode로 """
        prev_evaluations = []
        prev_trainings = []
        
        for i, model in enumerate(models): # evaluation_mode, __call__ = _evaluataion
            prev_evaluations.append(model._evaluation_mode)
            prev_trainings.append(model.training)
            model._evaluation_mode = True
            if prev_trainings[i]:
                model.eval()
        yield
        for i, model in enumerate(models): # previous mode, __call__ = forward
            model._evaluation_mode = prev_evaluations[i]
            if prev_trainings[i]:
                model.train()
    
    @staticmethod
    def transform_data(
        *dims, in_batch=True, in_device=True,
        out_batch=True,
        use_args=False, debug=False
    ):
        """
            in_batch/out_batch: 입/출력 data의 batch 추가 및 제거
            in_device/out_device: 입/출력 data의 device 이동
            in_many/out_many: 입/출력이 임의의 개수일 경우
        """
        if use_args and len(dims) > 1:
            raise Exception("use_args옵션은 여러개의 dims를 받을 수 없음")
        
        def transform_data_decorator(func):                                    
            def wrapper(self, *args, **kwargs):
                new_args = []
                new_kwargs = {}
                
                if in_batch: # add batch dimension for input data
                    if use_args: # use same dimension to all input datas(args)
                        dim = dims[0]
                        for arg in args:
                            # prev_device.append(arg.device)
                            new_args.append(makeInput(arg, dim=dim, device=self.device if in_device else None))
                        new_kwargs.update(kwargs)
                    else: # use individual dimension to input datas
                        argNames = list(func.__code__.co_varnames[1:])
                        new_kwargs.update(kwargs) # 모든 함수 input을 keyward 형식으로 변경
                        new_kwargs.update({argName:arg for argName, arg in zip(argNames, args)})
                        for varName, dim in zip(argNames, dims):
                            new_kwargs[varName] = makeInput(new_kwargs[varName], dim=dim, device=self.device if in_device else None)
                else: # just use raw input data
                    new_args = args
                    new_kwargs = kwargs
                    
                # NOTE: method(_evaluation) 수행 및 출력
                result = func(self, *new_args, **new_kwargs) # (dBatch, -)
                
                # NOTE: output의 batch dim 제거 및 device 이동
                if out_batch: # remove batch dimension for output data
                    if isinstance(result, tuple): # multiple outputs
                        result = [makeOutput(r) for r in result]
                    else: # single output
                        result = makeOutput(result) # ([dBatch], -)
                return result
            return wrapper
        return transform_data_decorator
    

# module들의 상위 module 작성 시 사용 e.g. agent, VAE, etc..
class MainModule(abc.ABC, Base):
    def update(): pass
    # def loss(): pass


# ------ environment class ------
class Environment:
    """
        support both new/old version Gym
    """
    def __init__(self, name, episode_max_step:int=-1, truncated_done:bool=False, mode_test:bool=False, **kwargs):
        self._name = name
        self._env = makeEnv(name=name, mode_test=mode_test, time_limit=False, **kwargs)
        # NOTE: episode_max_step이 설정되지 않은 경우, 환경에서 기본 설정된 max_step 사용
        self._episode_max_step = episode_max_step if isSet(episode_max_step) else self._env.spec.max_episode_steps
        self._truncated_done = truncated_done
        self._step = None
    
    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, Dict]:
        self._step = 0
        result = self._env.reset(*args, **kwargs)
        if isinstance(result, tuple): # new gym
            state, info = result
        else: # old gym
            state, info = result, {}
        return state, info
    
    def step(self, action, *args, **kwargs) -> Tuple:
        result = self._env.step(action, *args, **kwargs)
        self._step += 1
        
        if len(result) == 4: # old gym
            nextState, reward, done, info = result
            truncated = False
        elif len(result) == 5: # new gym
            nextState, reward, done, truncated, info = result
        else:
            raise NotImplementedError("[Environment] Unknown environment")
        
        if isHighEq(self._step, self._episode_max_step):
            # NOTE: time limit class가 없는 environment여야 제대로 작동함
            truncated = True
        if self._truncated_done:
            done = done or truncated
        
        return (nextState, reward, done, truncated, info)
    
    def getConf(self) -> AttrDict:
        """environment dimension, space 반환"""
        return getEnvConf(self._env)

    def getAction(self) -> Union[np.ndarray, int]:
        """random action sampling"""
        return self._env.action_space.sample()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self._name})'

    @property
    def action_space(self):
        return self._env.action_space
    @property
    def observation_space(self):
        return self._env.observation_space
    @property
    def n_step(self) -> int:
        return self._step
    

class Task:
    """
        Environment의 wrapper
        number of episode 또는 score, 현재 state등을 기록하는 기능 제공
    """
    def __init__(self, env, test_env=None):
        self.env = env
        self._test_env = test_env
        self.isTest = False
        self.init()
        
    def init(self) -> None:
        # NOTE: environment를 완전히 초기화 할때 필요(e.g. # of episode init)
        self.data = dict(state=None, score=None, terminal=None, episode=0) # train env data
        self._test_data = dict(state=None, score=None, terminal=None, episode=0) # test env data

    # NOTE: env, data dependent
    def reset(self, *args, **kwargs):
        state, info = self.env.reset(*args, **kwargs)
        
        self.data["state"] = state
        self.data["score"] = 0.0
        self.data["terminal"] = False
        self.data["episode"] += 1
        return state, info
    
    # NOTE: env, data dependent
    def step(self, action, *args, **kwargs):
        nextState, reward, done, truncated, info = self.env.step(action, *args, **kwargs)
        
        self.data["state"] = nextState
        self.data["score"] += reward
        self.data["terminal"] = done or truncated
        return nextState, reward, done, truncated, info
    
    # NOTE: env dependent
    def getRandomAction(self, *args, **kwargs):
        return self.env.action_space.sample()
    
    @contextmanager
    def test_mode(self):
        prev_env = self.env
        prev_data = self.data
        prev_mode = self.isTest
        
        self.env = self._test_env
        self.data = self._test_data
        self.isTest = True
        yield
        self.env = prev_env
        self.data = prev_data
        self.isTest = prev_mode
        
    # environment dependent
    @property
    def state(self):
        return self.data["state"]
    @property
    def score(self):
        return self.data["score"]
    @property
    def n_episode(self):
        return self.data["episode"]
    
    def isTerminal(self):
        return self.data["terminal"]

    
# --- logger ---

class LoggerModule(abc.ABC):
    @abc.abstractmethod
    def __init__(self, conf:dict, hyperparameter:dict): 
        """conf: logger 별 필요 config"""
        pass
    
    @abc.abstractmethod
    def write(self, datas:dict, step:int): 
        pass
    
    @abc.abstractmethod
    def close(self): 
        pass
    
class TensorboardLogger(LoggerModule):
    def __init__(self, path, exp_name, log_freq=1, hyperparameter=dict()):
        import warnings # warning ignore
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from torch.utils.tensorboard import SummaryWriter
        from torch.utils.tensorboard.summary import hparams
        import datetime
        logger = SummaryWriter(
            log_dir=f'{path}/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}-{exp_name}'
        )
        # hyperparameter 기록
        # NOTE: 현재 tensorboard는 hyperparameter를 저장하는 방법을 제공하지 않는다.
        #       이 방법은 sweep을 위한 기능에 hyperparameter를 저장한다.
        #       문제점으로, 첫 run에 해당하는 hyperparameter종류만 기록이되고 나머지는 기록되지 않는다.
        hparam_dict = {"PATH": path, "EXP_NAME": exp_name, **hyperparameter}
        metric_dict = {"_dummy": 1} 
        # NOTE: metric_dict가 없을 시, hyperparameter가 제대로 표현되지 않는 버그가 있다.
        exp, ssi, sei = hparams(hparam_dict, metric_dict, None)
        logger.file_writer.add_summary(exp)
        logger.file_writer.add_summary(ssi)
        logger.file_writer.add_summary(sei)
        self.logger = logger
        self.freq = log_freq
        self.running = True
    
    def write(self, datas:dict, step:int):
        for k, v in datas.items():
            self.logger.add_scalar(k, v, global_step=step)
        
    def close(self):
        if self.running:
            self.logger.flush()
            self.logger.close()
            self.running = False
        
class WandbLogger(LoggerModule):
    def __init__(self, project, exp_name, save_code=True, group=None, log_freq=1, key=None, hyperparameter=dict()):
        import wandb
        if key is not None:
            wandb.login(key=key)
        wandb.init(
            project=project, name=exp_name, group=group, save_code=save_code,
            config={**hyperparameter},
        )
        self.logger = wandb
        self.freq = log_freq
        self.running = True
        
    def write(self, datas:dict, step:int):
        self.logger.log(datas, step=step)
        
    def close(self):
        if self.running:
            self.logger.finish()
            self.running = False

class LocalLogger(LoggerModule):
    def __init__(self, log_freq=1, hyperparameter=dict()):
        self.data = dict(hyperparameter=hyperparameter, log=dict()) # log 저장
        self.freq = log_freq
        self.running = True
    
    def write(self, datas:dict, step:int):
        for name, value in datas.items():
            if name not in self.data["log"].keys(): # 처음보는 종류의 데이터인 경우, dict 할당
                self.data["log"][name] = dict()
            self.data["log"][name][step] = value # step을 key로 하여 데이터 저장

    def close(self):
        if self.running:
            self.running = False
            
    def getLastLog(self, names:list) -> dict:
        """해당 이름에 속하는 가장 최근 log를 가져옴"""
        lastLog = dict()
        for name in names:
            if not (name in self.data["log"].keys()):
                # 가져오고자 하는 변수가 아직 log에 저장된 적이 없다면, 해당 변수는 skip
                continue
            lastStep = max(self.data["log"][name].keys()) # 해당 변수의 가장 마지막 log 이름
            lastLog[name] = self.data["log"][name][lastStep]

        return lastLog
    
    def getData(self) -> dict:
        return self.data


class Logger:
    def __init__(self, config:List[Dict], hyperparameter:List[Dict], mode_online_only:bool=False, mode_print:bool=False):
        """
            config: logger 개별 config
            hyperparameter: log할 hyperparameter
            mode_online_only: local에 log 저장 안함, log_freq=1 일때만 가능
            mode_print: log 시 print 출력
        """
        # packing
        if not isinstance(config, list): config = [config]
        if not isinstance(hyperparameter, list): hyperparameter = [hyperparameter]
        config = list(filter(lambda x: x is not None, config)) # remove None element
        
        self.hyperparameter = {k:v for hyper in hyperparameter for k, v in hyper.items()} # merge
        self.offlineLogger = LocalLogger(log_freq=1, hyperparameter=self.hyperparameter)
        self.onlineLoggers = []
        for conf in config:
            if not conf.get("logger"): raise Exception("[Logger]No logger in config")
            loggerCls = conf.pop("logger")
            self.onlineLoggers.append(loggerCls(**conf, hyperparameter=self.hyperparameter))
            
        self.data = dict()
        # self._logFuncs = None
        # self._logLookup = None
        self.mode_online_only = mode_online_only
        self.mode_print = mode_print
        if mode_online_only:
            for logger in self.onlineLoggers:
                assert logger.log_freq == 1, "online_only mode support log_freq=1 only"
        
        
    # def register(self, funcs:dict, lookup:dict):
    #     self._logFuncs = funcs
    #     self._logLookup = lookup
        
    # def log(self, func_name, step=None, freq_mode=False):
    #     """
    #         func_name: log 함수 설정
    #         step: 몇번째 step에 log 할지, 지정하지 않은 경우 lookup.step을 따름
    #         freq: 해당 log는 지정된 freq 주기로 서버와 sync됨
    #     """
    #     func = self._logFuncs[func_name] # log 함수 선택
    #     datas = func(self._logLookup) # 선택된 log함수를 lookup 변수 넣고 실행
    #     # NOTE: step 지정 없을 시 lookup.step을 사용함, 따라서 해당 값의 지정이 필요
    #     step = step if step is not None else self._logLookup.step
    #     # self._log(datas, step, freq)
        
    #     # value가 None인 데이터 제거
    #     post_datas = {name:v for name, v in datas.items() if v is not None}
        
    #     if self.mode_print:
    #         print(f'[step {step}] {post_datas}')
    #     if not self.mode_online_only:
    #         self.offlineLogger.write(post_datas, step)
    #     for logger in self.onlineLoggers:
    #         if (not freq_mode) or (logger.freq == 1):
    #             # freq_mode가 아니거나, log 주기가 1인 경우 -> 일반적인 log
    #             logger.write(post_datas, step)
    #         elif freq_mode and isFreq(step, logger.freq):
    #             # freq_mode이며, 해당 주기인 경우 ->
    #             # local logger에서 해당하는 이름의 가장 최근 log를 불러옴
    #             last_datas = self.offlineLogger.getLastLog(datas.keys())
    #             logger.write(last_datas, step)
    #         else:
    #             # freq_mode이며, 해당 주기가 아닌경우 -> 통과
    #             pass
    
    
    # def log(self, func_name, step=None, freq_mode=False):
    def log(self, datas:dict, step:int, mode_freq=False):
        # value가 None인 데이터 제거
        post_datas = {name:v for name, v in datas.items() if v is not None}
        
        if self.mode_print:
            print(f'[step {step}] {post_datas}')
        if not self.mode_online_only:
            self.offlineLogger.write(post_datas, step)
        for logger in self.onlineLoggers:
            if (not mode_freq) or (logger.freq == 1):
                # freq_mode가 아니거나, log 주기가 1인 경우 -> 일반적인 log
                logger.write(post_datas, step)
            elif mode_freq and isFreq(step, logger.freq):
                # freq_mode이며, 해당 주기인 경우 ->
                # local logger에서 해당하는 이름의 가장 최근 log를 불러옴
                last_datas = self.offlineLogger.getLastLog(datas.keys())
                logger.write(last_datas, step)
            else:
                # freq_mode이며, 해당 주기가 아닌경우 -> 통과
                pass
        
    def close(self):
        self.offlineLogger.close()
        for logger in self.onlineLoggers:
            logger.close()
    
    def getData(self):
        if self.mode_online_only:
            raise Exception("[Logger] Online-only mode is on; log data does not exist.")
        return self.offlineLogger.getData()
        
        