import numpy as np
import torch

# NOTE: 주로 내부적으로 numpy.ndarray만을 사용함을 주의, 이 경우 gradient는 모두 끊긴다.
NUMPY = np.ndarray
TORCH = torch.Tensor

# TODO: 입력 자료형은 변경시키지 않고 저장한 후, getSample시 자료형 float 변환시키기
# NOTE: 내부적으로 numpy.ndarray 사용
class Transition(tuple):
    """
        one-step transition
        1. 모든 data는 torch.Tensor로 변환되어 저장됨
        2. scalar 데이터(i.e. ndim==0)는 unsqueeze 변환됨
        e.g. (state, action, reward, nextState, done)
    """
    # def __new__(self, *args, **kwargs):
    #     transition = []
    #     # NOTE: torch.Tensor, int, float -> np.ndarray
    #     # NOTE: ndim == 0 -> unsqueeze
    #     for item in args:
    #         if not isinstance(item, NUMPY):
    #             item = np.array(item, dtype=np.float32)
    #         else:
    #             item = item.astype(np.float32)
    #         if item.ndim == 0:
    #             item = np.expand_dims(item, axis=0) # (1, -)
    #         transition.append(item)
    #     return super().__new__(self, transition, **kwargs)
    def __new__(self, *args, **kwargs):
        transition = []
        # NOTE: torch.Tensor, int, float -> np.ndarray
        # NOTE: ndim == 0 -> unsqueeze
        for item in args:
            if not isinstance(item, NUMPY):
                if isinstance(item, bool):
                    item = np.array(item, dtype=bool)
                else:
                    item = np.array(item, dtype=np.float32)
            else:
                if item.dtype != bool and item.dtype != np.uint8:
                    item.astype(np.float32)
            if item.ndim == 0:
                item = np.expand_dims(item, axis=0) # (1, -)
            transition.append(item)
        return super().__new__(self, transition, **kwargs)
    
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __repr__(self):
        string = f'{self.__class__.__name__}'
        if self:
            string += "(\n  " + ",\n  ".join([f'[{i}] '+repr(j) for i, j in enumerate(self)]) + "\n)"
        else:
            string += "()"
        return string
    
    def export(self):
        return np.concatenate([item for item in self], axis=-1)

    # NOTE: Transition 객체의 pickle 또는 torch.load 시 발생하는 문제 처리
    def __reduce__(self):
        return (self.__class__, tuple(self))

# NOTE: 내부적으로 torch.Tensor 사용
class Batch(list):
    def __init__(self, *args):
        """
            make batch (e.g. [states, action, rewards, nextStates, dones])
            1. numpy(bool) -> torch.float32 (e.g. done)
            2. numpy(unit8) -> item/255.0 and torch.float32 (e.g. image state)
            3. numpy -> torch.float32 (e.g. all, etc.)
            
            input e.g.
            - Batch(Transition)
            - Batch([Transition, Transition, ...])
            - Batch(states, actions, ...)
        """
        if len(args) == 1 and isinstance(args[0], Transition):
            # NOTE: single Transition - e.g. Batch(Transition)
            args = [args]
        if len(args) == 1 and isinstance(args[0][0], Transition):
            # NOTE: Transition list - e.g. Batch([Transition, Transition, ...])
            args = [np.stack(item) for item in zip(*args[0])]
        if isinstance(args[0], NUMPY):
            # NOTE: numpy.ndarray items - e.g. Batch(states, actions, ...)
            args = [
                torch.from_numpy(item.astype(np.float32) / 255.0) if item.dtype == np.uint8 else \
                torch.from_numpy(item.astype(np.float32)) for item in args]
            # NOTE: if image (dtype=uint8) -> normalize and dtype=float32
            #       else -> only dtype=float32
            # args = [torch.from_numpy(item.astype(np.float32)) for item in args]
        if isinstance(args[0], TORCH): # make batch dimension
            # NOTE: torch.tensor items
            items = [item.unsqueeze(dim=0) if item.ndim == 1 else item for item in args]
        else: raise NotImplementedError

        super().__init__(items)

    def to(self, device):
        return Batch(*[item.to(device) for item in self])
    def detach(self):
        return Batch(*[item.detach() for item in self])
    def export(self):
        return np.concatenate([item for item in self], axis=-1)
        
    @property
    def n_item(self):
        """number of object"""
        return super().__len__()
    @property
    def n_size(self):
        """number of batch size"""
        return self[0].shape[0]
    def __len__(self):
        return self.n_size
    
    def __repr__(self):
        return f'{self.__class__.__name__}(n_size={self.n_size}, n_item={self.n_item})'


# XXX: transition 단위로 저장하므로 항상 하나 더 많은 state(주로 terminal state)를 담지 못한다는 문제가 있다.
#   -> nextState까지 같이 담도록 하면 문제 없긴 하다.
# NOTE: 내부적으로 Transition 서용
class Episode(list):
    def __init__(self, *args):
        """ 
            Transition list
            e.g.
            - Episode().addTransition(Transition)
            - Episode(Transition)
            - Episode([Transition, Transition, ...])
            - Episode(states, actions, ...)
        """
        if args:
            if isinstance(args[0], NUMPY) or isinstance(args[0], TORCH):
                transitions = [Transition(*items) for items in zip(*args)]
            # NOTE: single Transition - e.g. Episode(Transition)
            elif len(args) == 1 and isinstance(args[0], Transition):
                transitions = args
            # NOTE: Transition list - e.g. Episode([Transition, Transition, ...])
            elif len(args) == 1 and isinstance(args[0][0], Transition):
                transitions = args[0]
            else: raise NotImplementedError
            super().__init__(transitions)
        else:
            # NOTE: empty episode - e.g. Episode()
            super().__init__()
        
    def addTransition(self, *args):
        """
            e.g. addTransition(Transition)
            e.g. addTransition(state, action, ...)
        """
        if isinstance(args[0], Transition):
            self.append(args[0])
        else:
            self.append(Transition(*args))
        
    def asBatch(self):
        return Batch(self)
    def export(self):
        if self:
            return self.asBatch().export()
            # return torch.cat([item for item in self.asBatch()], dim=-1)
        else: raise Exception("empty episode")
    
    def sum(self, pos:int) -> float:
        """주로 episode의 score 게산 시 사용"""
        return sum([t[pos].item() for t in self])
    
    def __repr__(self):
        return f'{self.__class__.__name__}(horizon={self.n_transition})'
    
    @property
    def n_transition(self):
        return len(self)
    
# NOTE: 내부적으로 Transition 사용
class Buffer:
    def __init__(self, capacity:int, raw_mode=False):
        self.capacity = capacity
        self.raw_mode = raw_mode
        self.memory = None
        self.clear()
        
    def clear(self):
        del self.memory
        self.memory = []
        self._pointer = 0
        
    def setSample(self, *items):
        """e.g. setSample(state, action, reward, nextState, done)"""
        # 사용 가능한 형태로 변형
        data = self._preProcess(items)
        if not self.checkCapability(data):
            return
        self._setSample(data)
        
    def _preProcess(self, items:tuple):
        if self.raw_mode:
            # NOTE: raw_mode는 data 변형 없이 그대로 저장
            data = items[0] if len(items) == 1 else items
        else:
            data = Transition(*items)
        return data
    
    def checkCapability(self, data):
        """진행이 불가능한 데이터의 경우 error를 출력하고
            단순히 저장이 불가능한 경우 False 출력, 그외 True
        """
        return True
    
    def _setSample(self, data):
        if self.n_transition < self.capacity:
            self.memory.append(data)
        else:
            self.memory[self._pointer] = data
            
        self._stepPointer()
        
    def _stepPointer(self):
        self._pointer = (self._pointer + 1) % self.capacity

    def getSample(self, size:int):
        # NOTE: sampling
        indices = torch.randint(low=0, high=self.n_transition, size=(size,))
        # NOTE: get data from indicies
        datas = [self.memory[idx] for idx in indices] # Transition list
        if not self.raw_mode:
            datas = Batch(datas)
        return datas
    
    def export(self):
        if self.memory:
            return np.stack([t.export() for t in self.memory])
        else: raise Exception("empty buffer")
    
    @property
    def n_transition(self):
        return len(self.memory)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.n_transition}/{self.capacity})'
    def __getitem__(self, key):
        return self.memory.__getitem__(key)
    
    
class EpisodeBuffer(Buffer):
    def __init__(self, capacity:int):
        self.horizon = None
        super(EpisodeBuffer, self).__init__(capacity=capacity, raw_mode=True)
        
    def clear(self):
        del self.horizon
        self.horizon = []
        super(EpisodeBuffer, self).clear()
        
    def checkCapability(self, data:Episode):
        result = super(EpisodeBuffer, self).checkCapability(data)
        assert isinstance(data, Episode), "[buffer]not Episode instance"
        return result and True
    
    def _setSample(self, data:Episode):
        if self.n_episode < self.capacity:
            self.memory.append(data)
            self.horizon.append(data.n_transition)
        else:
            self.memory[self._pointer] = data
            self.horizon[self._pointer] = data.n_transition
        
        self._stepPointer()
        
    def getEpisodeSample(self, size:int):
        # NOTE: sampling
        # indices = np.random.randint(0, self.n_episode, size=size)
        indices = torch.randint(low=0, high=self.n_episode, size=(size,))
        # NOTE: get data from indicies
        datas = [self.memory[idx] for idx in indices] # Transition list
        if not self.raw_mode:
            datas = Batch(datas)
        return datas
        
    def getSample(self, size:int, length=1, info=False):
        """
            전체 episode에서 length길이의 transition random sampling
            모든 episode의 transition을 uniform하게 sampling
        """
        horizon = np.array(self.horizon)
        counts = (horizon - length + 1).clip(min=0) # 저장된 episode별 가능한 서로다른 sample 수
        sampleIds = np.random.randint(0, counts.sum(), size=size)
        
        # NOTE: 해당 sampleId가 몇번 episode의 몇번째 transition인지 계산
        cumCounts = np.cumsum(counts)
        episodeIds = np.searchsorted(cumCounts - 1, sampleIds)
        offsets = np.concatenate(([0], cumCounts[:-1])) # episode별 시작 offset
        transitionIds = sampleIds - offsets[episodeIds]
        
        if length == 1:
            # (size, -)
            datas = Batch([self.memory[episodeId][transitionId] for episodeId, transitionId in zip(episodeIds, transitionIds)])
        else:
            # (size, length, -)
            datas = [[np.stack(item) for item in zip(*self.memory[episodeId][transitionId:transitionId+length])] for episodeId, transitionId in zip(episodeIds, transitionIds)]
            datas = Batch(*[np.stack(item) for item in zip(*datas)])
        if info:
            return datas, episodeIds
        else:
            return datas
    
    def export(self):
        """torch.Tensor(nTransition, dTransition)"""
        if self.memory:
            return np.stack([ep.export() for ep in self.memory])
        else: raise Exception("empty buffer")
        
    @property
    def n_transition(self):
        return sum(self.horizon)
    @property
    def n_episode(self):
        return len(self.memory)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(episode={self.n_episode}, transition={self.n_transition})'

class MaxTransitionEpisodeBuffer(EpisodeBuffer):
    def __init__(self, capacity:int):
        self.priority = None
        super(MaxTransitionEpisodeBuffer, self).__init__(capacity=capacity)
    
    def clear(self):
        del self.priority
        self.priority = []
        self._id = 0
        super().clear()
    
    def _getPriority(self, data:Episode):
        return self._id
    
    def _stepPointer(self):
        self._pointer = np.array(self.priority).argmin()
        
    def checkCapability(self, data: Episode):
        result = super(MaxTransitionEpisodeBuffer, self).checkCapability(data)
        cnt = 0
        
        # if self.n_transition + data.n_transition > self.capacity:
        horizon = self.horizon.copy()
        priority = self.priority.copy()
        p = self._getPriority(data)
        while sum(horizon) + data.n_transition > self.capacity:
            cnt += 1
            # NOTE: data의 priority가 현재 buffer에 있는 데이터 중 가장 낮은 priority
            #       와 비교하여 같거나 작을 경우 데이터 삽입 X
            pointer = np.array(priority).argmin()
            if priority[pointer] >= p:
                return False # input data의 priority가 저장된 data들 보다 작은 경우 생략
            if cnt > 30:
                raise Exception("[buffer]infinite loop...")
            del horizon[pointer]
            del priority[pointer]
        return result
        
    def _setSample(self, data:Episode):
        while self.n_transition + data.n_transition > self.capacity:
            del self.memory[self._pointer]
            del self.horizon[self._pointer]
            del self.priority[self._pointer]
            self._stepPointer()
            
        self.memory.append(data)
        self.horizon.append(data.n_transition)
        self.priority.append(self._getPriority(data))
        self._id += 1

class RewardPriorityMaxTransitionEpisodeBuffer(MaxTransitionEpisodeBuffer):
    def __init__(self, capacity:int, reward_pos:int, later=False):
        self.reward_pos = reward_pos
        self.later = later
        super(RewardPriorityMaxTransitionEpisodeBuffer, self).__init__(capacity=capacity)

    def checkCapability(self, data: Episode):
        result = super(MaxTransitionEpisodeBuffer, self).checkCapability(data)
        cnt = 0
        
        horizon = self.horizon.copy()
        priority = self.priority.copy()
        p = self._getPriority(data)
        while sum(horizon) + data.n_transition > self.capacity:
            cnt += 1
            # NOTE: data의 priority가 현재 buffer에 있는 데이터 중 가장 낮은 priority
            #       와 비교하여 같거나 작을 경우 데이터 삽입 X
            pointer = np.array(priority).argmin()
            if (self.later and priority[pointer] > p) or (not self.later and priority[pointer] >= p):
                return False
            
            if cnt > 30:
                raise Exception("[buffer]infinite loop...")
            del horizon[pointer]
            del priority[pointer]
        return result

    def _getPriority(self, data:Episode):
        return data.sum(pos=self.reward_pos)
    
    
    