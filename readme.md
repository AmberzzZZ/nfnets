official repo: https://github.com/deepmind/deepmind-research/tree/master/nfnets


## NFNet family
    * based on SE-ResNeXt-D
    * modified width pattern: 标准版[256, 512, 1536, 1536]，加宽版[384, 768, 2048, 2048]
    * modified depth pattern: [1, 2, 6, 3]*(1~8)
    * train / test resolution: 递增
    * drop rate: 递增

    标准版: F0, F1, ..., F7
    加宽版: F0+, F1+, ..., F7+


## standard conv & activation
    * nonlinearities: magic constants
    * conv: WSConv2D, scaled weight standardization
        ** 先对weight进行标准化 (W-mean)/sqrt(N*var)
        ** 再进行rescale（gain）, operate on channel-dim
        ** WSConv2D要支持groupConv, 发现自定义Layer中assign方法不可导, 换成等号可导
        ** 参数量: 
            - kernel: k*k*c_in*c_out/groups
            - bias: c_out
            - learnable gain: c_out


## stem
    3x3conv, ch16, s2, activation
    3x3conv, ch32, s1, activation
    3x3conv, ch64, s1, activation
    downsamp: 3x3conv, ch[stage2width//2], s2


## resblock
    * transition / non-transition block
        * se-block: scale2
        * stochastic depth: 
            - 不是随机丢block，而是像efficientNet里面那样，Batchwise Dropout
            - only in training
    * std variance reset
        - 每个transition block以后，variance reset为 alpha**2 + 1
        - 每个non-transition block以后，variance更新为 alpha**2 + last_variance**2
        - 每个block的beta是标准差的倒数


## [gelu](https://blog.csdn.net/sinat_36618660/article/details/100088097)
    在激活函数中引入随机正则的思想：对输入不是粗暴的阈值滤波，而是依据正态分布生成的mask
    GELU(x) = x * P(X<=x)
    其中X服从Normal(x)，Normal()可以是标准正态分布N(0,1)，也可以引入可学习参数N(mu, sigma)
    P(X<=x)是对应高斯正态分布的积分
    基于标准正态分布的gelu函数可以通过erf函数/sigmoid函数近似计算


## custom SGD_AGC optimizer









