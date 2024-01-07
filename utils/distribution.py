import torch
import torch.nn.functional as F
import abc

class Distribution(abc.ABC):
    @abc.abstractmethod
    def to(self, *args, **kwargs): 
        pass
    
    @abc.abstractmethod
    def detach(self, *args, **kwargs):
        pass


class Normal(Distribution, torch.distributions.Normal):
    """ detach, to 지원 Normal """
    @property
    def mu(self): return self.mean
    @property
    def sigma(self): return self.stddev
    @property
    def std(self): return self.stddev
    @property
    def shape(self): return self.mu.shape
    @property
    def device(self): return self.mean.device
    
    @classmethod
    def create(cls, *shape):
        mean = torch.zeros(shape)
        std = torch.ones(shape)
        return cls(mean, std)
    
    def detach(self):
        return type(self)(self.mean.detach(), self.std.detach())
    
    def to(self, *args, **kwargs):
        return type(self)(self.mean.to(*args, **kwargs), self.std.to(*args, **kwargs))
    
    def __repr__(self):
        return f'{self.__class__.__name__}' + \
                f'[mu{tuple(self.mean.shape)}, sigma{tuple(self.std.shape)}, {str(self.mean.device)}]'


class TanhNormal(Distribution, torch.distributions.Distribution):
    def __init__(self, loc, scale):
        self.normal = Normal(loc, scale)
    def cdf(self, value):
        raise NotImplementedError("dist error")
        return super().cdf(value)
    def entropy(self):
        raise NotImplementedError
        return super().entropy()
    def icdf(self, value):
        raise NotImplementedError("dist error")
        return super().icdf(value)
    def perplexity(self):
        raise NotImplementedError("dist error")
        return super().perplexity()
    
    @property
    def mean(self):
        return self.normal.mean.tanh()
    @property
    def mu(self):
        return self.normal.mean.tanh()
    @property
    def stddev(self):
        return self.normal.stddev
    @property
    def std(self):
        return self.normal.stddev
    @property
    def device(self):
        return self.normal.mean.device
    
    def log_prob_from_pre_tanh(self, pre_tanh_value):
        # NOTE: pre_tanh_value = action에 대한 inverse tanh 값
        log_prob = self.normal.log_prob(pre_tanh_value)
        correction = - 2.0 * (torch.tensor(2.0).log() - pre_tanh_value
            - F.softplus(-2.0 * pre_tanh_value)
        )
        # return (log_prob + correction).sum(dim=-1, keepdim=True)
        return (log_prob + correction)
    
    
    # XXX: 함수 밖에서 value의 범위를 -1 ~ 1로 조정한 후 사용해야함
    # NOTE: action=1인 경우 clamp로 0.999999가 되어 계산되는데, 이때
    #       stddev가 매우 작다면 해당 확률이 매우 작게 된다.
    #       ,즉 거의 100% 확률로 1이 나와야 하는데 clamp로 인해 0.999999로 계산이 되며
    #       이때 매우작은 stddev로 인해 0.999999가 나올 확률은 거의 0으로 계산되고
    #       이를 log_prob로 계산하면 -inf에 가까운 값이 나오게 된다.
    #       따라서 log_std_min을 최소 -2정도로 조정하지 않으면 발산하게 된다.
    # NOTE: 그러므로 가능하다면 log_prob_from_pre_tanh로 계산하는것이 좋고, 그렇지 못하는 경우
    #       log_std_min을 -2 이상으로 조절해야 한다.
    #       또한 batch size를 늘리는 것도 어느정도 도움이 되는 것 같다.
    #       예를 들어 하나가 -inf 가까운 값이 되더라도, 나머지가 클 경우 평균하여 계산되기 때문이다.
    def log_prob(self, value):
        # NOTE: atanh 함수는 pytorch 1.8 이상 부터 지원
        # NOTE: clamp없을 시 -1, 1에서 -inf, inf값 나옴
        # XXX: inf 값은 optimizer에서 무시되므로 몇몇 implementation에서는 inf값을 유지하는게 맞는것 아닌지?
        value = value.clamp(min=-0.999999, max=0.999999)
        return self.log_prob_from_pre_tanh(value.atanh())
    # # inverse tanh 적용
    #     one_plus_x = (1 + value).clamp(min=1e-6)
    #     one_minus_x = (1 - value).clamp(min=1e-6)
    #     normal_value = 0.5*torch.log(one_plus_x / one_minus_x)
    #     return self.log_prob_from_pre_tanh(normal_value)
    
    def sample(self, sample_shape=torch.Size()):
        z = self.normal.sample(sample_shape=sample_shape)
        return z.tanh()
    
    def rsample(self, sample_shape=torch.Size()):
        z = self.normal.rsample(sample_shape=sample_shape)
        return z.tanh()
    
    def to(self, *args, **kwargs):
        return type(self)(self.normal.mean.to(*args, **kwargs), self.normal.stddev.to(*args, **kwargs))
    
    def detach(self):
        return type(self)(self.normal.mean.detach(), self.normal.stddev.detach())
    
    def __repr__(self):
        return f'{self.__class__.__name__}' + \
                f'[mu{tuple(self.normal.mean.shape)}, sigma{tuple(self.normal.stddev.shape)}, {str(self.normal.mean.device)}]'