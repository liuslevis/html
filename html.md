---
layout: title

title: HTML

date: 2017-03-21 15:15:54

tags: 

categories:

---

机器学习笔记

<!--more-->

## 评估指标

### 准确率

查准率

How many relevant items are selected?

$$
accuracy = \frac{TP} {(TP + FN)}
$$

### 召回率

查全率

How many selected items are relevant?

$$
recall = \frac{TP} {(TP + FN)}
$$

### F1 score

准确率和召回率的调和平均值

$$
\frac{2} {F_1} = \frac{1} {accuracy} + \frac{1} {recall}
$$

$$
F_1 = \frac{1} {2 * (\frac{1}{accuracy} + \frac{1}{recall})} = \frac{accuracy + recall}{2 * accuracy * recall}
$$

### ROC/AUC

Receiver Operating Characteristic, 考虑二分类问题, 调整阈值, 绘制 TP 和 FP 坐标曲线如下图. 其中左上角是完美的分类结果, 右下角是最差的分类结果, 曲线越接近左上角越好. 

Area Under Curve, 表示 ROC 曲线下的面积, 越大越好. AUC 是一个概率值, 当你随机挑选一个正样本以及一个负样本, 当前的分类算法根据计算得到的Score值将这个正样本排在负样本前面的概率就是AUC值.  ([Fawcett](http://people.inf.elte.hu/kiss/12dwhdm/roc.pdf), 2006)

ROC 曲线的一个特性: 当测试集中的正负样本的分布变化的时候, ROC曲线能够保持不变;而 Precision-Recall 曲线会随之变化

[ROC](http://alexkong.net/images/Roccurves.png)

### 距离范数

损失函数中, 正则项一般是参数的 Lp 距离. 

L1最优化问题的解是稀疏性的, 其倾向于选择很少的一些非常大的值和很多的insignificant的小值. 而L2最优化则更多的非常少的特别大的值, 却又很多相对小的值, 但其仍然对[最优化解有significant的贡献. ](http://math.stackexchange.com/questions/384003/l1-norm-and-l2-norm#)但从最优化问题解的平滑性来看, L1范数的最优解相对于L2范数要少, 但其往往是最优解, 而L2的解很多, 但更多的倾向于某种局部最优解. 

L0范数本身是特征选择的最直接最理想的方案, 但如前所述, 其不可分, 且很难优化, 因此实际应用中我们使用L1来得到L0的最优凸近似. L2相对于L1具有更为平滑的特性, 在模型预测中, 往往比L1具有更好的预测特性. 当遇到两个对预测有帮助的特征时, L1倾向于选择一个更大的特征. 而L2更倾向把两者结合起来. 

[lp-balls](http://t.hengwei.me/assets/img/post/lp_ball.png)

#### L0-范数

$$
d=\sum_{i=1}^{n}\bigg(x_i^{0}\bigg)^{\frac {1}{0}}
$$

向量中非零元素的个数

在 Sparse Coding 中, 通过最小化 L0 寻找最少最优的稀疏特征. 但难以优化, 一般转化成 L1 L2 

#### L1-范数

曼哈顿距离

$$
d=\sum_{i=1}^{n}\bigg|x_i-y_i\bigg|
$$

如计算机视觉中对比两张图片的不同像素点之和

#### L2-范数

欧几里得距离

$$
d=\sum_{i=1}^{n}\bigg(|x_i-y_i|^{2}\bigg)^{\frac {1}{2}}
$$

#### Lp-范数

$$
d=\sum_{i=1}^{n}\bigg(|x_i-y_i|^{p}\bigg)^{\frac {1}{p}}
$$

#### 无限范数距离

切比雪夫距离

$$
d=\underset {p \to \infty}{\lim}\sum_{i=1}^{n}\bigg(|x_i-y_i|^{p}\bigg)^{\frac {1}{p}} = \max(|x_1-y_1|,…,|x_n-y_n|)
$$

### 损失函数

#### 对数损失函数

Log Loss

$$
L(Y,P(Y|X))=-\log P(Y|X)
$$

对 LR 而言, 把它的条件概率分布方程 

$$
P(Y=0|X)=\frac {1}{1+e^{wx+b}}, P(Y=1|X)=1-P(Y=0|X)
$$

带入上式, 即可得到 LR 的对数损失函数

#### 平方损失函数

Square Loss

$$
L(Y,P(Y|X))=\sum_{i=1}^{n}(Y-f(X))^2
$$

其中 $$Y-f(X)$$ 表示残差, 整个式子表示残差平方和, Residual Sum of Squares

#### 指数损失函数

$$
L(Y,f(X))=e ^ {-Y f(X)}
$$

如 Adaboost, 它是前向分步加法算法的特例, 是一个加和模型, 损失函数就是指数函数. 经过m此迭代之后, 可以得到

$$
f_m(x)=f_{m-1}(x)+\alpha_mG_m(x)
$$

Adaboost 每次迭代的目标是最小化指数损失函数

$$
\underset {\alpha, G}{\arg\ \min}=\sum_{i=1}^{N}e^{[-y_i(f_{m-1}(x_i)+\alpha G(x_i))]}
$$

#### 合页损失

Hinge Loss, 如 SVM

$$
L(Y,f(X))=\max (0,1-Y*f(X)), Y=\{0,1\}
$$

### 正则化

Regulization or Penalization

$$
\underset{f\in F}{min} \frac{1}{N}\sum_{i=l}^{N}L(y_i,f(x_i))+\lambda J(f)
$$

其中第一项是经验风险, 第二项是正则化项

正则化项有不同形式, 如回归问题中, 损失函数是平方损失, 正则化项可可以是参数向量的 $$L_2$$ 范数, 用 $$||w||$$ 表示

$$
L(w)=\frac{1}{N}\sum_{i=1}^{N}(f(x_i;w)-y_i)^2 + \frac{\lambda}{2} ||w||^2
$$

也可以是 $$L_1$$ 范数, 用 $$||w||_1$$ 表示

$$
L(w)=\frac{1}{N}\sum_{i=1}^{N}(f(x_i;w)-y_i)^2 + \lambda ||w||_1
$$

### 泛化能力

#### 泛化误差

用学到的模型 $$\hat f$$ 对未知数据预测的误差, 是所学习到模型的期望风险

$$
R_{exp} (\hat f) = E_p [L(Y,\hat f (X))] = \int_{x \times y} L(y,\hat f(x)) P(x,y) dxdy
$$

#### 泛化误差上界

性质: 它是样本容量的函数, 当样本增加时, 趋于0;他是假设空间容量的函数, 假设空间 (一组函数的集合) 容量越大, 模型学习越困难, 泛化误差上界越大

下面考虑二分类问题, 已知训练集 $$T=\{x_i,y_i\}$$, 它是从联合概率分布 $$P(X,Y)$$ 的独立同分布产生的, $$X\in \mathbb{R}^n, Y \in \{-1,+1\}$$ , 假设空间函数的有限集合 $$F=\{f_1,f_2,…,f_d\}$$, d 是函数个数. 设 $$f$$ 是从 $$F$$ 中选取的函数, 损失函数是 0-1 损失, 关于 $$f$$ 的期望风险和经验风险是

$$
R(f)=E[L(Y,f(X))]
$$

$$
\hat R(f)=\frac{1}{n} \sum_{i=1}^{N}L(y_i,f(x_i))
$$

经验风险最小化函数是

$$
f_N=\underset {f \in F}{\arg\ \min} \ \hat R(f)
$$

我们关心 $$f_N$$ 的泛化能力

$$
R(f_N)=E[L(Y,f_N(x))]
$$

泛化误差上界

$$
R(f) \leq \hat R(f)+\epsilon(d,N,\delta)
$$

其中

$$
\epsilon(d,N,\delta)=\sqrt{\frac{1}{N} (\log d + \log \frac{1}{\delta})}
$$

其中 $$R(f)$$ 是泛化误差, 右端 $$\epsilon(d,N,\delta)$$ 既为泛化误差上界

可以看出, 样本越多, 泛化误差趋于0;空间 $$F$$ 包含的函数越多, 泛化误差越大. 证明见《统计学习方法》 P16. 

以上讨论的只是假设空间包含有限个函数情况下的泛化误差上界, 对一般的假设空间要找到泛化误差上界较复杂. 

### 熵

给定一个离散有限随机变量 $$X$$ 的分布为 $$P(X = x_i)=p_i$$ , i=1,2,...,n, 那么 $$X$$ 的熵定义为

$$
H(X) = \sum_{i} - p_{i}\log p_{i}
$$

熵越大, 随机变量的不确定性越大, 从定义可以验证

$$
0 \le H(X) \le \log n
$$

如无特殊说明, 本文的 $$log$$ 皆为自然对数

### 条件熵

随机变量(X, Y), 其联合概率分布为 $$ P(X,Y)=p_{ij} $$, 随机变量 $$X$$ 给定的条件下随机变量 $$Y$$ 的条件熵 $$H(Y|X)$$, 定义为 $$X$$ 给定条件下 $$Y$$ 的条件概率分布的熵对 $$X$$ 的数学期望

$$
H(Y|X)=\sum_ip_iH(Y|X=x_i)
$$

其中 $$ p_i=P(X=x_i) $$.

由数据估算 (如极大似然估计) 得到时, 称为经验条件熵. 


### 信息增益

信息增益表示得知特征 $$X$$ 的信息而使得类 $$Y$$ 的信息的不确定性减少的程度. 

特征 $$A$$ 对训练数据集 $$D$$ 的信息增益 $$g(D, A)$$, 定义为集合 $$D$$ 的经验熵 $$H(D)$$ 与特征 $$A$$ 给定条件下 $$D$$ 的经验条件熵 $$H(D|A)$$ 之差, 即

$$
g(D, A)=H(D)-H(D|A)
$$


### 信息增益比

信息增益比 $$gR(D,A)$$ 定义为其信息增益 $$g(D, A)$$ 与训练数据集 $$D$$ 关于特征 $$A$$ 的值的熵 $$H_A(D)$$ 之比, 即

$$
gR(D,A)= \frac{|D_i|}{|D|} \log _{2} {\frac{|D_i|}{|D|}}
$$

其中,  $$n$$ 是特征 $$A$$ 取值的个数. 
$$
H_A(D)=−∑i\frac{|Di|}{|D|} \log_{2}{\frac{|Di|}{|D|}}
$$

### 基尼指数 (gini index) 

分类问题中, 假设有 $$K$$ 个类, 样本属于第 $$k$$ 类的概率为 $$p_k$$, 则概率分布的基尼指数定义为: 

$$
Gini(p)=∑p_k(1−p_k)=1−∑p^2_k
$$

对于二分类问题, 若样本点属于第 1 个类的概率是 p, 则概率分布的基尼指数为: 

$$
Gini(p)=2p(1-p)
$$

对于给定的样本集合D, 其基尼指数为: 

$$
Gini(D)=1−∑(|C_k||D|)^2
$$

这里,  $$C_k$$ 是 $$D$$ 中属于第 $$k$$ 类的样本子集, $$k$$ 是类的个数. 

如果样本集合 $$D$$ 根据特征 $$A$$ 是否取到某一可能值 $$a$$ 被分割成 $$D_1$$ 和 $$D_2$$ 两部分, 则在特征 $$A$$ 的条件下, 集合 $$D$$ 的基尼指数定义为: 

$$
Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_1|}{|D|}Gini(D_2)
$$

基尼指数 $$Gini(D)$$ 表示集合 $$D$$ 的不确定性, 基尼指数越大, 样本集合的不确定性也就越大, 这一点与熵相似



## 回归问题

给定一个训练集

$$
T={(x_1,y_1), (x_2,y_2),...,(x_n,y_n)}
$$

构造一个函数 $$Y=f(X)$$. 

对新的输入 $$x_{n+1}$$, 根据学习到的函数 $$Y=f(X)$$ 计算 $$y_{n+1}$$. 

## 分类问题

给定一个训练集

$$
T={(x_1,y_1), (x_2,y_2),...,(x_n,y_n)}
$$

构造一个条件概率分布模型

$$
P(Y^{(1)},Y^{(2)},...,Y^{(n)}) | X^{(1)},X^{(2)},...,X^{(n)})
$$

其中, 每个$$X^{(i)}$$取值取值为所有可能的观测, 每个$$Y^{(i)}$$取值为所有可能的分类结果. 

对新的预测序列 $$x_{n+1}$$, 找到条件概率$$P(y_{n+1}|x_{n+1})$$最大的标记序列 $$y_{n+1}$$, 得到预测结果. 

## 线性回归

对样本 $$T=\{(x_0,y_0),...,(x_N,y_N)\}, x_i \in X \subseteq \mathbb{R}^{m+1}, y_i \in Y \subseteq \mathbb{R}^{1} $$, 其中 $$m$$ 是特征, $$x$$ 的第 $$m+1$$ 是常数 $$1$$

### Least Square

又称最小二乘法, 通过最小化平方和误差得到参数估计

$$
L(w) = \frac{1}{N} ||Y-X w||^2
$$

当 $$X$$ 是满秩的, 即 $$rank(X) = dim(x)$$, 行/列互不线性相关, 有解析解

$$
\hat w = (X^T X)^{-1} X^T Y
$$

但实际问题中 X 往往不是满秩的, 上式的协方差矩阵 $$X^T X$$ 不可逆, 目标函数最小化导数为零时方程有无穷解，没办法求出最优解, 同时存在 overfitting 问题, 这时需要对参数进行限制

### 最小岭回归

加入L2惩罚项

$$
L(w) = \frac{1}{N} ||Y-X\omega||^2 + \frac{\lambda}{2} ||w||^2
$$

### LASSO 

加入L1惩罚项, 把参数约束在 $$L1\ ball$$ 中

更多的系数为0, 可以用来做特征选择, 具有可解释性

$$
L(w) = \frac{1}{N} ||Y-X\omega||^2 + \frac{\lambda}{2} ||w||^1
$$

或优化目标形式

$$
\underset{w}{\min} \frac{1}{2}||y-Xw||^2 \\
\text{s.t.} ||w||^1 < \theta
$$

有解析解

$$
\hat w_R = (X^T X + \lambda I)^{-1} X^T y
$$

### Elastic Net

L1+L2

## LR

源自 Logistic 分布, 是由输入对线性函数表示的输出的对数几率模型

对一个二分类问题，假设样本随机变量 $$X$$ 服从伯努利分布 (0-1分布) , 即概率质量函数为 

$$
f_X(x)=p^x(1-p)^{1-x},x \in \{0,1\}
$$

期望值 $$E[X]=p$$, 方差 $$var[X]=p(1-p)$$ 

二项逻辑斯谛回归模型是一种分类模型, 由条件概率分布 $$P(X|Y)$$ 表示

$$
P(Y=0|x)=\frac {1} {1 + \exp(w \cdot x + b)}
$$

$$
P(Y=1|x)=\frac {\exp(w \cdot x + b)} {1 + \exp(w \cdot x + b)}
$$

训练集: 训练集 D 特征集 A

输入: $$x$$

输出: $$0\le y \le1$$

定义事件发生几率 (Odds) 

$$
\frac {p} {1-p}
$$

对数几率, 或 logit 函数

$$
logit(p)=\log {\frac {p}{1-p}}
$$

对逻辑斯谛回归而言

$$
\log {\frac{P(Y=1|x)}{P(Y=0|x)}}=w \cdot x + b
$$

有时 $$ w \cdot x + b $$ 简记为 $$ w \cdot x $$. 

输出 $$Y=1$$ 的对数几率是输入 $$x$$ 的线性函数, 或者说, 输出 $$Y=1$$ 的对数几率是由输入 $$x$$ 的线性函数表示的模型, 即逻辑斯谛回归模型. 

### 参数估计

为简化推导, 设

$$
P(Y=1|x)=\pi(x), P(Y=0|x)=1-\pi(x)
$$

损失函数 (似然函数) 为

$$
\prod\limits_{i=1}^{N}[\pi(x_i)]^{y_i}[1-\pi(x_i)]^{1-y_i}
$$

对数损失函数为

$$
\begin{align}\\
L(w) &= \sum\limits_{i=1}^{N}[y_i \log \pi (x_i) + (1 - y_i) \log (1-\pi(x_i))]\\
    & =\sum\limits_{i=1}^{N}[y_i \log \frac {\pi (x_i)} {1-\pi (x_i)} + \log (1-\pi(x_i))]\\
    & =\sum\limits_{i=1}^{N}[y_i(w \cdot x_i) - \log (1+e^{w \cdot x_i })]\\
\end{align}
$$

对 $$L(w)$$ 求极大值, 常用迭代尺度法(IIS) / 梯度下降法 / 拟牛顿法, 得到 $$w$$ 的估计值. 

损失函数的一阶、二阶导数为

$$
\frac{\partial L(w)}{\partial w} = X^T (y-\pi) \\
\frac{\partial^2 L(w)}{\partial w \partial w^T} = - X^T W X \\
$$

其中

$$
W_{ij} = \pi(x_i; w)(1 - \pi(x_j; w))\\
\pi_i = \pi(x_i; w)
$$

[ref](http://sites.stat.psu.edu/~jiali/course/stat597e/notes2/logit.pdf)

### 用法

输入特征相互独立 (descrete)

算法把输入问题通过对数转换

## 最大熵模型

熵满足下列不等式

$$
0\leq H(P) \leq \log |X|
$$

其中, $$|X|$$ 是 $$X$$ 的取值个数, 当且仅当 $$X$$ 的分布是均匀时, 等式 $$H(P) = \log|X|$$ 成立. 即当 X 服从均匀分布时, 熵最大

熵最大模型认为, 学习概率模型时, 在所有可能的概率分布模型中, 熵最大的模型, 即等可能分布的模型是最好的模型. 

[The equivalence of logistic regression and maximum entropy models](http://www.win-vector.com/dfiles/LogisticRegressionMaxEnt.pdf)

LR 模型在建模时建模预测 Y|X, 并认为 Y|X 服从伯努利分布, 所以我们只需要知道 P(Y|X);其次我们需要一个线性模型, 所以 P(Y|X) = f(wx). 接下来我们就只需要知道 f 是什么就行了. 而我们可以通过最大熵原则推出的这个 f, 就是sigmoid

给定训练集 $$T=\{(x_1,y_1), ..., (x_N, y_N)\}$$, 可以确定联合分布 $$P(X,Y)$$ 的经验分布

$$
\widetilde{P}(X=x,Y=y)=\frac{v(X=x,Y=y)}{N}
$$

和边缘分布 $$P(X=x)$$的经验分布

$$
\widetilde{P}(X=x)=\frac{v(X=x)}{N}
$$

那么, 特征函数 $$f(x,y)$$ 关于经验分布 $$\widetilde{P}(X,Y)$$ 的期望值, 用 $$E_{\widetilde{P}}(f)$$ 表示

$$
E_{\widetilde{P}}(f)=\sum\limits_{x,y} \widetilde{P}(x,y)f(x,y)
$$

$$
E_{P}(f)=\sum\limits_{x} \widetilde{P}(x)P(y|x)f(x,y)
$$

如果模型可以获得训练数据中的信息, 那么可以假设这两个值相等

$$
E_{\widetilde{P}}(f)=E_{P}(f)
$$

将上式子作为约束条件. 如果有 $$n$$ 个特征函数, 那么就有 $$n$$ 个约束条件

最大熵模型: 假设满足所有约束条件的模型集合为

$$
C \equiv \{P \in P_{all} | E_P(f_i)= E_\widetilde{P}(f_i)\space , i=1,2,...,n\}
$$

定义在条件概率分布 $$P(Y|X)$$ 的条件熵为

$$
H(P)=-\sum\limits_{x,y} \widetilde{P}(x)P(y|x) \log P(y|x)
$$

则模型集合 $$C$$ 中条件熵 $$H(P)$$ 最大的模型, 称为最大熵模型

### 熵最大模型的学习

对给定训练集 $$T=\{(x_1,y_1),...,(x_N,y_N)\}$$ 以及特征函数 $$f_i(x,y)$$，求解最优化问题

$$
\begin{align} \\
\underset{P\in C}{\min} \hspace{1em} & -H(P)=\sum_{x,y} \tilde P(x)P(y|x) \log P(y|x)\\
s.t.                    \hspace{1em} & E_p(f_i) - E_{\tilde P}(f_i) = 0, i = 1,2,...,n \\
                        \hspace{1em}  & \sum_y P(y|x)=1 \\
\end{align} \\
$$

## Softmax

对多分类问题, 假设离散型随机变量 $$Y$$ 的取值集合是 $$\{1,2,…,K\}$$ , 那么逻辑斯谛模型推广到 softmax 有: 

$$
P(Y=k|x)=\frac {\exp(w_k \cdot x )} {1+\sum_{k=1}^{K-1} \exp (w_k \cdot x)}, k=1,2,...,K-1
$$

$$
P(Y=K|x)=\frac{1}{1+\sum_{k=1}^{K-1}\exp(w_k \cdot x)}
$$

这里, $$x \in \mathbb{R}^{n+1},w_k \in \mathbb{R}^{n+1}$$

## k-NN

1968 Cover & Hart

一种高效实现: KD 树

## 决策树

求所有可能的决策树是 NP 完全问题, 常用启发式方法, 近似求解, 得到次优解, 常用学习算法: `ID3` / `C4.5` / `CART`

特征选择的准则, 用`信息增益`、`信息增益`比来衡量: 如果用一个特征进行分类的结果与随机分类的结果没有很大差异, 则认为这个特征是没有分类能力的. 

一般算法: 递归地选择最优特征, 并根据该特征对训练数据进行分割. 

剪枝: 以避免过拟合, 考虑全局最优. 

优点: [wiki](https://zh.wikipedia.org/wiki/决策树学习)

* 易于理解和解释
* 只需很少的数据准备 其他技术往往需要数据归一化
* 即可以处理数值型数据也可以处理类别型数据
* 可以通过测试集来验证模型的性能 
* 强健控制. 对噪声处理有好的强健性
* 可以很好的处理大规模数据

### ID3

极大似然进行概率模型的选择, 用 **信息增益** 来选择特征

输入: 训练集 D, 特征集 A, 阈值 ε

输出: 决策树 T

1. 若 $$D$$ 同属一类 $$C_k$$, 则 $$T$$ 为单节点树, 将 $$C_k$$ 作为该节点的类标记, 返回 $$T$$

2. 若 $$A = \{\}$$ , 则 $$T$$ 为单节点树, 将 $$D$$ 中实例数最大的类 $$C_k$$ 作为该节点的类标记, 返回 $$T$$

3. 否则, 计算 $$A$$ 中特征对 $$D$$ 的信息增益, 选择信息增益最大的特征 $$A_g$$ 

4. 如果 $$A_g$$ 的信息增益小于阈值 $$ε$$, 则置 $$T$$ 为单节点树, 将 $$D$$ 中实例数最大的类 $$C_k$$ 作为该节点的类标记, 返回 $$T$$

5. 否则, 对 $$A_g$$ 的每一可能值 $$a_i$$, 以  $$A_g=a_i$$  (选均值、随机值或者其他) 将 $$D$$ 分割为若干非空子集  $$D_i$$, 将 $$D_i$$ 中实例最大的类作为标记, 构建子节点, 构成树 $$T$$, 返回 $$T$$

6. 对第 $$i$$ 个子节点, 以 $$D_i$$ 为训练集, $$A-{A_g}$$ 为特征集, 递归地调用步骤 1 至 5, 得到子树 $$T_i$$, 返回 $$T_i$$


### C4.5

改进 ID3, 用 **信息增益比** 来选择特征

### 剪枝

一种简单的方法: 通过极小化损失函数来实现, 损失函数定义

$$
C_\alpha(T)=\Sigma_tN_tH_t(T)+\alpha|T|
$$

其中 $$T$$ 是节点个数, t 是叶节点, 有 Nt 个样本点, k 类样本点有 $$N_{tk}$$ 个, $$k=1,2,...,K$$, $$\alpha\ge0$$ 为参数, $$H_t(T)$$ 是叶节点 t 的经验熵, 定义

$$
H_t(T)=-\Sigma_tN_tH_t(T)
$$

模型对训练数据的误差为

$$
C_\alpha(T)=C(T)+\alpha|T|
$$

其中 $$|T|$$ 表示模型复杂度, $$\alpha$$ 越小, 模型越复杂

### CART

分类与回归树, Breiman 1984 提出, 算法由特征选择、树的生成、剪枝组成, 分类、回归任务都可以用. 

#### 最小二乘法生成回归树

输入：训练集 $$T=\{(x_1,y_1),...,(x_N,y_N)\}, x_i \in X, y_i \in Y $$, 其中 $$Y$$ 是连续变量

输出: 回归树 $$f(x)$$

假设将输入空间划分成 $$M$$ 个单元 $$R_1,...,R_M$$, 并且在每个单元 $$R_m$$ 上有固定输出值 $$c_m$$, 于是回归树模型可表示为

$$
f(x)=\sum_{m=1}^{M} c_m I(x \subseteq R_m)
$$

当输入空间的划分确定时, 可用平方误差表示回归树的预测误差, 用平方误差最小的准则求借每个单元上的最优输出值

$$
\hat c_m = ave(y_i|x_i \subseteq R_m)
$$

如何对输入空间划分? 用启发式的方法, 选择第 j 个变量 $$x^{(j)}$$ 和它的取值 $$s$$, 作为切分变量和切分点, 并定义两个区域

$$
R_1(j,s)=\{x|x^{(j)} \leq s\}\\
R_2(j,s)=\{x|x^{(j)} \gt s\}
$$

然后寻找最优切分变量 $$j$$ 和最优切分点 $$s$$, 即求解

$$
\underset{j,s}{\min}\bigg[\underset{c_1}{\min} \sum_{x_i\in R_1(j,s)} (y_i - c_1)^2 + \underset{c_2}{\min}\sum_{x_i\in R_2(j,s)} (y_i-c_2)^2\bigg]
$$

对固定输入变量 $$j$$ 可以找到最优切分点 $$s$$

$$
\hat c_1 = ave(y_i|x_i \in R_1(j,s))\\
\hat c_2 = ave(y_i|x_i \in R_2(j,s))
$$

遍历所有输入变量,找到最优的切分变量 $$j$$, 构成一个对 $$(j,s)$$, 将输入空间划分为两个区域

接着,对每个区域重复上述划分过程, 直到满足停止条件, 得到输入空间的划分空间 $$R_1,R_2,...,R_M$$, 和一颗回归树

$$
f(x)=\sum_{m-1}^M \hat c_m I(x\in R_m)
$$


## 随机森林

是决策树的组合, 适用于分类和回归. 较决策树, 它更容易避免过拟合, 不需要对参数进行正则化预处理, 冗余、相关的参数不会影响结果, 输入特征可以是离散和连续的. 

在训练过程中引入随机性, 如特征的随机选择、训练集的随机抽样, 并行训练多颗树. 多个预测的结合, 有助于降低预测在某棵树上的相关性, 增加在测试集上的性能. 

优点: 

* 对于很多数据集表现良好, 精确度比较高
* 不容易过拟合
* 可以得到变量的重要性排序
* 既能处理离散型数据, 也能处理连续型数据
* 不需要进行归一化处理
* 能够很好的处理缺失数据
* 容易并行化等等

## 随机森林 vs GDBT

Gradient-Boosted Trees 又名 MART(Multiple Additive Regression Tree), GBRT(Gradient Boost Regression Tree), Tree Net, Treelink, 由 Friedman 发明. 

GBT 迭代地训练一组决策树, 每棵树的训练残差结果, 作为下一个带训练的树的输入标签. 

二者的区别有: 

* GBTs 每次训练一颗树, 而 RF 是并行训练多棵树, 前者时间较长
* 增加 GBTs 的 numTrees, 会增加过拟合的可能;增加 RF 的 numTree, 会减少过拟合的可能性
* RF 较容易调参, 因为 numTree 增加会使性能单调提高, 而 GBTs 的 numTree 增加有可能会使性能降低

总体而言, 二者的选择, 取决于你数据集的特性

## Adaboost

在概率近似正确 (PAC) 学习的框架中, 一个类如果存在一个多项式的学习算法能够学习它且正确率[高|仅比随机猜测略好], 称这个类是[强|若]可学习的

Schapire 证明: 在 PAC 学习框架中, 强可学习 $$\leftrightarrows$$ 弱可学习

Adaboost 从一个弱分类学习算法出发, 反复学习, 得到一组若分类器, 然后构成一个强分类器. 在每一轮改变训练数据的权值或概率分布, [提高|降低]前一轮被[错误|正确]分类样本的权值;采取加权多数表决的方法, [加大|减少]分类误差率[小|大]的弱分类器的权值, 

以二分类问题为例, 给定样本 $$ T=\{(x_1,y_1),…, (x_N,y_N\},x\in X, y\in Y =\{-1,+1\}$$ , 其中 $$X$$ 是样本空间, $$Y$$ 是标签集合. 

输入: 训练集 $$T$$ ;弱学习算法

输出: 最终分类器 $$G(x)$$

1) 初始化训练数据的权值分布

$$
D_1(w_{11}, ..., w_{1i}, ..., w_{1N}), w_{1i}=\frac{1}{N},i=1,2,...,N
$$

2) 对 $$m=1,2,…,M$$

a) 使具有权值分布 $$D_m$$ 的训练数据集学习, 得到基本分类器

$$
G_m(x):X \to {-1,1}
$$

b) 计算 $$G_m(x)$$ 在训练集数据上的分类误差率

$$
e_m=P(G_m(x_i)) \neq y_i) = \sum_{i=1}^{N}w_{mi}I(G_m(x_i)\neq y_i)
$$

c) 计算 $$G_m(x)$$ 的系数

$$
\alpha_m=\frac{1}{2}\log{\frac{1-e_m}{e_m}}
$$

d) 更新训练集数据的权值分布

$$
D_{m+i}=(w_{m+1,1} ..., w_{m+1,i},..., w_{m+1,N})
$$

$$
w_{m+1,i}=\frac{w_{mi}}{Z_m} \exp ({-\alpha_m y_i G_m(x_i)})
$$

这里 $$Z_m$$ 是规范化因子

$$
Z_m=\sum_{i=1}^{N}w_{mi}\exp(-\alpha_my_iG_m(x_i))
$$

它使 $$D_m$$ 成为一个概率分布

3) 构建基本分类器的线性组合

$$
f(x)=\sum_{m=1}^{N} \alpha_m G_m(x)
$$

最终得到分类器

$$
G(x)=sign(f(x))
$$

### 算法解释

有一种解释, 可以认为 AdaBoost 是模型为加法模型、损失函数为指数函数, 学习算法为前向分步算法时的二分类学习方法. 

### 加法模型

$$
f(x)=\sum_{m-1}^{M}\beta_mb(x;\gamma_m)
$$

其中 $$b(x;\gamma_m)$$ 是基函数, $$\gamma_m$$ 是基函数的参数, $$\beta_m$$ 是基函数的系数

给定训练集 $$T$$ 和损失函数 $$L(y,f(x))$$, 学习加法模型 $$f(x)$$ 即损失函数的最小化问题

$$
\underset{\beta_m,\gamma_m}{\min}{\sum_{i=1}^{N} L\bigg(y_i,\sum_{m-1}{M}\beta_mb(x;\gamma_m)\bigg)}
$$

### 前向分步算法

学习加法模型时, 从前向后, 每一步只学习一个基函数及其系数, 逐步逼近优化目标, 达到简化计算复杂度

输入: 训练集 $$T=\{(x_i,y_i\}$$;损失函数 $$L(y,f(x))$$;基函数集 $$\{b(x,\gamma\}$$

输出: 加法模型 $$f(x)$$

1) 初始化 $$f_0(x)=0$$

2) 对 $$m=1,2,…,M$$

a) 极小化损失函数

$$
(\beta_m,\gamma_m)=\underset{\beta, \gamma}{\arg\ \min} \sum_{i=1}^{N} L(y_i,f_{m-1}(x_i)+\beta b(x_i;\gamma))
$$

得到参数 $$\beta_m,\gamma_m$$

b) 更新

$$
f_m(x)=f_{m-1}+\beta_mb(x;\gamma_m)
$$

3) 得到加法模型

$$
f(x)=f_M(x)=\sum_{m=1}^M \beta_m b(x;\gamma_m)
$$

这样, 算法将同时求解从 $$m=1$$ 到 $$M$$ 所有参数 $$\beta_m,\gamma_m$$ 的优化问题简化为逐渐求解各个 $$\beta_m,\gamma_m$$ 的优化问题

## Boosting Tree

提升树, 是以分类树或回归树为基本分类器的提升方法. 它采用加法模型与前向分步算法, 以决策树为基函数的提升方法. 对[分类|回归]问题的决策树是二叉[分类|回归]树. 

提升树可以表示为决策树的加法模型

$$
f_M(x)=\sum_{m=1}^{M}T(x;\Theta_m)
$$

其中 $$T$$ 表示决策树, $$\Theta_m$$ 表示决策树的参数, $$M$$ 表示树的个数

提升树用前向分步算法训练

1) 令初试提升树 $$f_0(x)=0$$ 

2) $$m=1,2,…,M$$

$$
f_m(x)=f_{m-1}(x)+T(x;\Theta_M)
$$

其中 $$f_{m-1}$$ 是当前模型, 通过经验风险极小化确定下一棵树的参数 $$\Theta_m$$

$$
\hat \Theta_m = \arg\ \min \sum_{i=1}^{N}L(y_i,f_{m-1}+T(x_i;\Theta_m))
$$

当采用平方误差损失函数时

$$
\begin{align}
L(y,f(x)) &= (y-f(x))^2\\
& =[y-f_{m-1}(x)-T(x;\Theta_m)]^2\\
& =[\gamma-T(x;\Theta_m)]^2
\end{align}
$$

这里

$$
\gamma = y-f_{m-1}(x)
$$

由于树的线性组合可以很好地你和训练数据, 即使数据中的输入与输出关系很复杂, 所以提升树是一个高功能的学习算法

### 回归问题的提升树算法

如果输入空间 $$X$$ 划分为 $$J$$ 个互不相交的区域 $$R_1,R_2,…,R_n$$, 并且在每个区域上输出固定的常量 $$c_j$$ , 那么树可表示为

$$
T(x;\Theta)=\sum_{j=1}^{j}c_jI(x\in \mathbb{R}_j)
$$

其中 $$I(x)=1\ if\ x\ is\ true\ else\ 0$$

输入: 训练集 $$T=\{(x_1,y_1,…,(x_n,y_n)\},x_i \in X \subseteq \mathbb{R}^n,y_i \in Y \subseteq \mathbb{R}$$

输出: 提升树 $$f_M(x)$$

1) 初始化 $$f_0(x)$$

2) 对 $$m=1,2,…,M$$

a) 计算残差

$$
\gamma_{mi}=y_i-f_{m-1}(x_i),i=1,2,...,N
$$

b) 拟合残差 $$\gamma_{mi}$$ 学习一个回归树, 得到 $$T(x;\Theta_m)$$

c) 更新 $$f_m(x)=f_{m-1}+T(x;\Theta_m)$$

3) 得到提升回归树

$$
f_M(x)=\sum_{m=1}^{M}T(x;\Theta_m)
$$

### 梯度提升算法

提升树用加法模型和前向分步算法实现学习的优化过程. 当损失函数是平方损失和指数损失时, 每一步优化很简单;对一半损失函数而言, 有时并不容易. 针对这个问题, Freidman 提出了梯度提升 Gradient Boost 算法. 这是利用最速下降法的近似方法, 关键是利用损失函数的负梯度在当前模型的值

$$
-\bigg[\frac{\partial L(y,f(x_i))}{\partial f(x_i)}\bigg]_{f(x)=f_{m-1}(x)}
$$

作为回归问题提升树算法中的残差的近似值, 拟合一个回归树

输入: 训练集 $$T=\{(x_1,y_1,…,(x_n,y_n)\},x_i \in X \subseteq \mathbb{R}^n,y_i \in Y \subseteq \mathbb{R}$$;损失函数 $$L(y,f(x))$$

输出: 回归树 $$\hat f(x)$$

1) 初始化

$$
f_0(x)=\arg \underset{c}{\min} \sum_{i=1}^{N}L(y_i,c)
$$

2) 对 $$m=1,2,..,M$$

a) 对 $$i=1,2,…,N$$ 计算

$$
\gamma_{mi}=-\bigg[\frac{\partial L(y,f(x_i))}{\partial f(x_i)}\bigg]_{f(x)=f_{m-1}(x)}
$$

b) 对 $$\gamma_{mi}$$ 拟合一个回归树, 得到第 $$m$$ 棵树的叶节点区域 $$R_{mj}$$, $$j=1,2,…,J$$

c) 对 $$j=1,2,…,J$$ 计算

$$
c_{mj}=\arg \underset {c}{\min} \sum_{x_i \in \mathbb{R}_{mj}} L(y_i,f_{m-1}(x_i)+c)
$$

d) 更新

$$
f_m(x)=f_{m-1}(x)+\sum_{j=1}^{J}c_{mj}I(x\in \mathbb{R}_{mj})
$$

3)得到回归树

$$
\hat f(x)=f_M(x)=\sum_{m=1}^{M}\sum_{j=1}^{J} c_{mj}I(x\in \mathbb{R}_{mj})
$$

## 感知机

### 定义

$$
f(x)=sign(w\cdot x + b)
$$

其中, $$sign(x)=1\ if\ x \ge 0\ else\ 0$$

几何解释: $$w\cdot x + b$$ 是特征空间的超平面, 把特征空间划分成两部分. 

### 损失函数

1) 错误分类点总数, 但不是连续可导, 不容易优化

2) 错误分类点到超平面的距离. 对于给定 $$x_0$$ 到超平面的距离是

$$
-\frac {1}{||w||}|w\cdot x_0 + b|
$$

其中 $$||w||$$ 是 L2范式. 那么有损失函数

$$
L(w,b)=-\sum_{x_i\in M}^{}y_i|w\cdot x_0 + b|
$$

其中 $$M$$ 是错误分类点的集合

### 学习方法

随机梯度下降法 stochastic gradient descent

梯度

$$
\nabla_wL(w,b)=-\sum_{x_i\in M}y_ix_i\\
\nabla_bL(w,b)=-\sum_{x_i\in M}y_i
$$

随机取一个错误分类点, 对 $$w,b$$ 进行更新

$$
w\leftarrow w+\eta y_ix_i\\
b\leftarrow b+\eta y_i\\
0 \le \eta \le1
$$

### 收敛性证明

TODO

### 对偶形式

将 $$w,b$$ 表示为实例 $$x_i,y_i$$ 的线形组合形式, 通过求解其系数的到 $$w,b$$

输入: 训练集 $$T={(x_1,y_1),...,(x_N,y_N)}$$ 学习率 $$\eta$$ 

输出: 

$$
\alpha,b,f(x)=sign\bigg(\sum_{j=1}^{N}\alpha_jy_jx_j\bigg)\\
\alpha=(\alpha_1,...,\alpha_N)^T, \alpha_i=n_i\eta
$$

1) $$\alpha \leftarrow 0, b \leftarrow 0$$

2) 在训练集中选取数据 $$x_i,y_i$$

3) 如果

$$
y_i \bigg(\sum_{j=1}^{N}\alpha_j y_j x_j+b\bigg)\le0
$$

那么

$$
\alpha_i \leftarrow \alpha_i + \eta\\
b \leftarrow b + \eta y_i
$$

4) 转 2) 直到没有错误分类数据

## SVM

定义在特征空间上的间隔最大的线性分类器, 以及核技巧 (非线性分类器) 

学习策略是间隔最大化, 可形式化成一个求解凸二次规划 convex quadratic programming 问题

学习方法有: 

* 线性可分 SVM: 当训练数据线性可分, 通过 hard margin maximization 学习
* 线性 SVM: 当训练数据近似线性可分, 通过 soft margin maximization
* 非线性 SVM: 当训练数据线性不可分, 通过 kernel trick 及 soft margin maximization 学习

### 线性可分 SVM

给定线性可分训练集, 通过间隔最大化或等价地求解相应凸二次规划问题而得到的分离超平面

$$
w^* \cdot x=b^*
$$

以及相应的分类决策函数

$$
f(x)=sign(w^* \cdot x + b^*)
$$

称为线性可分支持向量机

#### 函数间隔

对给定训练集 T 和超平面 $$(w,b)$$, 定义超平面关于样本点 $$(x_i,y_i)$$ 的函数间隔为

$$
\hat \gamma_i = y_i (w \cdot x_i + b)
$$

定义超平面关于所有样本点的函数间隔为

$$
\hat \gamma = \underset {i=1,...,N} {\min} \hat \gamma_i
$$

函数间隔用来表示分类的正确性和确信度

但成比例地改变 $$w,b$$ 会使函数间隔改变, 为解决这个问题, 可以对法向量加约束如 $$||w||=1$$, 使间隔确定, 即几何间隔的思想

#### 几何间隔

对给定训练集 T 和超平面 (w,b), 定义超平面关于样本的几何间隔为

$$
\gamma = \underset {i=1,...,N} {\min} \gamma_i\\
\gamma_i = y_i \bigg(\frac{w}{||w||}\cdot x_i + \frac{b}{||w||}\bigg)
$$

函数间隔、几何间隔的关系有

$$
\gamma_i=\frac {\hat \gamma_i}{||w||}\\
\gamma=\frac {\hat \gamma}{||w||}
$$

#### 间隔最大化求解

线性可分问题中, 分离超平面有无穷多个 (感知机) , 但几何间隔最大的分离超平面上唯一的

线性[可分|不可分]间隔最大化又称 [hard|soft] margin maximization

可以表示为最优化问题

$$
\begin{align}
\underset {w,b}{\max}\ &\gamma\\
s.t.\ &y_i\bigg(\frac{w}{||w||}\cdot x_i + \frac{b}{||w||}\bigg) \geq \gamma, i=1,2,...,N
\end{align}
$$

考虑集合间隔和函数间隔的关系, 有

$$
\begin{align}
\underset {w,b}{\max}\ &\hat\gamma\\
s.t.\ &y_i (w\cdot x_i + b) \geq \hat \gamma, i=1,2,...,N
\end{align}
$$

令 $$\hat \gamma = 1$$, 优化问题等价于一个凸二次优化问题

$$
\begin{align}
\underset {w,b}{\min}\ &\frac{1}{2}||w||^2\\
s.t.\ &y_i (w\cdot x_i + b) - 1\geq 0, i=1,2,...,N
\end{align}
$$

凸优化问题指

$$
\begin{align}
\underset {w}{\min}\ &f(w)\\
s.t.\ &g_i(w) \leq 0, i=1,2,...,k\\
&h_i(w)=0, i=1,2,...,l
\end{align}
$$

其中, 目标函数 $$f$$ 和约束函数 $$g$$ 都是 $$\mathbb{R}^n$$ 上的连续可微的凸函数, 约束函数 $$h$$ 是 $$\mathbb{R}^n$$ 上的仿射函数 ( 满足 $$h(x)=a\cdot x + b, a \in \mathbb{R}^n, b \in \mathbb{R},x \in \mathbb{R}^n$$) 

当目标函数 $$f$$ 是二次函数且约束函数 $$g$$ 是仿射函数时, 上述问题称为凸二次规划问题 convex quadratic programming

#### 最大间隔分离超平面存在唯一性

TODO 证明存在性

证明唯一性

#### 对偶算法

TODO 应用拉格朗日对偶性, 推导得到对偶形式《统计学习方法》P103

$$
\begin{align}
\underset{\alpha} {\min}\ & \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j (x_i \cdot x_j) - \sum_{i=1}^{N}\alpha_i\\
s.t.\ & \sum_{i=1}^{N}\alpha_i y_i=0\\
& 0 \le \alpha_i , i = 1,2,...,N
\end{align}
$$

#### 合页损失函数

hinge loss

另一种解释, 最小化以下目标函数

$$
\sum_{i=1}^{N}[1-y_i(w\cdot x_i+b)]_++\lambda ||w||^2
$$

其中第一项是合页损失函数, 定义

$$
[z]_+=\begin{cases}
	z, & z \gt 0 \\
	0, & z \le 0
\end{cases}
$$

### 线性不可分 SVM

对线性不可分问题, 引入松弛变量, 解决某些样本点不能满足函数间隔大于1的约束条件 ($$s.t.\ y_i(w\cdot x_i+b)-1\leq 0$$) , 优化问题变为

$$
\begin{align}
\underset {w,b}{\min}\ &\frac{1}{2}||w||^2 + C\sum_{i=1}^N \xi_i\\
s.t.\ &y_i (w\cdot x_i + b) \geq 1 - \xi_i, i=1,2,...,N\\
& \xi_i \geq 0
\end{align}
$$

可以证明 $$w$$ 时唯一的, $$b$$ 不唯一但存在于一个区间

#### 对偶算法

$$
\begin{align}
\underset{\alpha} {\min}\ & \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j (x_i \cdot x_j) - \sum_{i=1}^{N}\alpha_i\\
s.t.\ & \sum_{i=1}^{N}\alpha_i y_i=0\\
& 0 \le \alpha_i \le C, i = 1,2,...,N
\end{align}
$$

#### 支持向量

线性不可分时, 将对偶问题的解 $$\alpha^*=(\alpha_1^*,...,\alpha_N^*)^T$$ 中对应于 $$\alpha_i^* > 0$$ 的样本点 $$(x_i,y_i)$$ 实例 $$x_i$$ 称为支持向量

### 非线性 SVM

有的二分类问题无法用超平面 (线性模型, 在二维是直线) 分类, 但可用超曲面 (非线性模型, 在二维是椭圆) 将它们正确分开, 那么这个问题是非线性可分问题. 此时可用线性变换, 把非线性问题变换为线性问题

#### 核函数

设 $$X$$ 是输入空间 (欧氏空间 $$\mathbb{R}^n$$ 的子集或离散集合) , 设 $$H$$ 为特征空间 (Hilbert  空间) , 如果存在一个

$$
\phi(x): X \rightarrow H
$$

使得对所有 $$x,z \in X$$ , 函数 $$K(x,z)$$ 满足条件

$$
K(x,z)=\phi(x) \cdot \phi(z)
$$

则称 $$K(x,z)$$ 为核函数, $$\phi(x)$$ 为映射函数, 其中 $$\cdot$$ 表示内积即 $$a \cdot b=\sum_{i=1}^{n} a_ib_i$$

例: 在特征空间 $$\mathbb{R}^2$$, 有

$$
K(x,z)=(x \cdot z)^2\\
\phi(x) = ((x^{(1)})^2,\sqrt{2}x^{(1)}x^{(2)}, (x^{(2)})^2)^T
$$

#### 核技巧

在学习与预测中只定义核函数 $$K(x,z)$$, 而不显式地定义映射函数 $$\phi$$

通常, 直接计算 $$K$$ 比较容易, 通过 $$\phi$$ 计算 $$K$$ 不容易. 把 SVM 的目标函数、决策函数的内积 $$x_i \cdot x_j$$ 用核函数 $$K(x_i,x_j)=\phi(x_i) \cdot \phi(x_j)$$ 代替, 此时对偶问题的目标函数为

$$
W(\alpha)=\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j K(x_i,x_j) - \sum_{i=1}^{N}\alpha_i
$$

分类决策函数为

$$
f(x)=sign\bigg(\sum_{i=1}^{N_s}\alpha_i^* y_i \phi(x_i)\cdot \phi(x)+b^*\bigg)=sign\bigg(\sum_{i=1}^{N_s}\alpha_i^* y_i K(x_i,x)+b^*\bigg)
$$

#### 正定核


## CNN

### BP

[BP](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf) Hinton, 1986

对神经网络的平方代价函数

$$
L = \frac{1}{2N} \sum_{x} ||y(x)-a^L(x)||^2
$$

BP 是一个快速计算神经网络代价函数梯度的方法

1) 输入 $$x$$, 记 $$a^1$$

2) 正向传播, 对 $$l=2,3,...,L$$ 计算 

$$
z^l=w^l a^{l-1} + b^l, a^l=\sigma(z^l)
$$

3) 输出误差 

$$
\delta^L = \nabla_{a} C \odot \delta'(z^L)
$$

其中, $$\nabla_{a} C$$ 可以看做现对于输出激活的 $$C$$ 的改变速率

$$
\nabla_{a} C = \frac{\partial C}{\partial a_j^L}
$$

4) 将误差反向传播, 对每个 $$l=L-1,L-2,...,2$$ 计算 

$$
\delta^l=(w^{l+1}T\delta^{l+1})\odot \partial'(z^l)
$$

5) 输出: 代价函数梯度

$$
\frac{\partial C}{\partial w_{jk}^{l}}=a_{k}^{l-1}\delta_j^l \\
\frac{\partial C}{\partial b_j^l}=\delta_j^l
$$






### LeNet-5

[Gradient-based learning applied to document recognition](yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) Yan LeCun, 1998

[LeNet](https://www.52ml.net/wp-content/uploads/2016/08/lenet.png?_=5821591)

### AlexNet 12'

ImageNet top-5 error 16.4%

* Data Augmentation 数据增强: 水平翻转, 随机剪裁、平移变换, 颜色、光照变换, 防止过拟合
* Dropout , 防止过拟合
* ReLU, 代替 tanh sigmoid, 好处有: 本质是分段线性模型, 前向计算简单;偏导简单, 反向传播梯度, 无需指数或者除法之类操作;不容易发生梯度发散问题, Tanh和Logistic激活函数在两端的时候导数容易趋近于零, 多级连乘后梯度更加约等于0;关闭了右边, 从而会使得很多的隐层输出为0, 即网络变得稀疏, 起到了类似L1的正则化作用, 可以在一定程度上缓解过拟合. 缺点是会导致部分借点永久关闭, 改进如 pReLU randomReLu
* Local Response Normalization 利用临近数据组做归一化, top-5 error -1.2%
* Overlapping Pooling Pooling的步长比Pooling Kernel的对应边要小, top-5 error -0.3%
* GPU

[AlexNet](https://www.52ml.net/wp-content/uploads/2016/08/alexnet2.png?_=5821591)

### VGG

[home page](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) VGG team, Oxford, 2014

ImageNet top-5 error 7.3%

* deepper

[VGG](https://www.52ml.net/wp-content/uploads/2016/08/vgg19.png?_=5821591)

### GoogLeNet

[Going Deeper with Convolutions](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) 2014

ImageNet top-5 error 6.7%

* Inception, network in network 

[Inception](https://www.52ml.net/wp-content/uploads/2016/08/inception.png?_=5821591)

[GoogLeNet](https://www.52ml.net/wp-content/uploads/2016/08/googlenet.png?_=5821591)

### ResNet

[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf), Kaiming He et. al., MSRA

ImageNet top-5 error 3.57%

* Residual network, 解决层次比较深的时候无法训练的问题. 这种借鉴了Highway Network思想的网络相当于旁边专门开个通道使得输入可以直达输出, 而优化的目标由原来的拟合输出H(x)变成输出和输入的差H(x)-x, 其中H(X)是某一层原始的的期望映射输出, x是输入

[Residual Network](https://www.52ml.net/wp-content/uploads/2016/08/residual.png?_=5821591)

[ResNet](https://www.52ml.net/wp-content/uploads/2016/08/resnet.png?_=5821591)

## 无约束最优化问题求解

神经网络中的学习过程可以形式化为最小化损失函数问题, 该损失函数一般是由训练误差和正则项组成

$$
L (w)=train\_err(w) + norm(w)
$$

损失函数的一阶偏导为

$$
\nabla_i L(w)=\frac {df}{dw_i},i=1,2,...,n
$$

损失函数二阶偏导可以使用海塞矩阵 Hessian Matrix $$\mathbf{H}$$ 表示, 其中每个权重向量 $$i$$ 的元素 $$j$$ 的二阶偏导数为

$$
\mathbf{H}_{i,j} L(w) = \frac {d^2 f} {dw_i \cdot dw_j}
$$

求解方法有

### 改进的迭代尺度法

Improved Iterative Scalling, IIS, 想法是: 假设模型向量是 w, 如果能有这样一种向量更新方法 $$\iota:w \rightarrow w + \delta$$ ，那么重复使用这一种方法，直到找到最大值

### 梯度下降

$$
w_{i+1} := w_i - d_i \cdot \eta_i
$$

优点: 使用一阶导数计算, 复杂度小于二阶导数

缺点: 变量没有归一化, 锯齿下降现象, 因为非线性函数局部的梯度方向并不一定就是朝着最优点

### 随机梯度下降

Stochastic Gradient Descent

每次迭代, 选取部分样本进行计算

### 牛顿法

用损失函数的二阶偏导数寻找更好的训练方向. 记

$$
Li = L(w_i), g_i=\nabla_i, \mathbf{H}_i=\mathbf{H}f(w_i)
$$


在 $$w_0$$ 点使用泰勒级数展开式二次逼近损失函数 $$L$$

$$
L=L_0+g_0 \cdot (w-w_0) + 0.5 \cdot (w-w_0)^2 \cdot \mathbf{H}_0
$$

另一阶导数 $$g=0$$ , 有

$$
g=g_0+\mathbf{H}_0 \cdot (w-w_0)=0
$$

参数 $$w$$ 迭代公式为

$$
w_{i+1}=w_i - {\mathbf{H}_i^{-1}} \cdot {g_i}
$$

如果海塞矩阵非正定, 损失函数并不能保证在每一次迭代中都是减少的. 可以通过加上学习速率解决这个问题

$$
w_{i+1}=w_i - {\mathbf{H}_i^{-1}} \cdot {g_i} \cdot \eta_i
$$

优点: 比一阶导数更少迭代

缺点: 计算复杂度比一阶导数更高, 因为对海塞矩阵及其逆的精确求值在计算量复杂度是十分巨大的. 

### 拟牛顿法

Quasi-Newton method

拟牛顿法不直接计算海塞矩阵然后求其矩阵的逆, 而是在每次迭代的时候, 利用一阶偏导矩阵 Jacobian Matrix 或其他方法, 以逼近 Hessian Matrix 的逆.

1) 输入:

$$
L,w_0,\mathbf{H}_0^{-1},\mathbf{QuasiUpdate}
$$

2) 对于 $$n = 0,1,...$$ 更新参数, 直到收敛:

a) 计算搜索方向和步长

$$
\begin{align} \\
d &= \mathbf{H}^{-1} g_n\\
\alpha &\leftarrow \underset{\alpha \geq 0}{\min} L(w_n - \alpha d) \\
w_{n+1} &\leftarrow w_n - \alpha d \\
\end{align}
$$

b) 计算并保存

$$
\begin{align}\\
g_{n+1} &\leftarrow \nabla L(w_{n+1}) \\
s_{n+1} &\leftarrow w_{n+1} - w_{n} \\
L_{n+1} &\leftarrow g_{n+1} - g_{n} \\
\end{align}
$$

c) 更新

$$
\mathbf{H}_{n+1}^{-1} \leftarrow \mathbf{QuasiUpdate}(\mathbf{H}_n^{-1}, s_{n+1}, L_{n+1})
$$


### BFGS

[理解L-BFGS算法](http://mlworks.cn/posts/introduction-to-l-bfgs/)

是一种拟牛顿法

由中值定理我们知道

$$
g_n - g_{n-1} = \mathbf{H}_n(w_n - w_{n-1})
$$

所以有 

$$
\mathbf{H}_n^{-1} L_n = s_n
$$

同时，Hessian矩阵是对称矩阵。在这两个条件的基础上，我们希望 $$\mathbf{H}_n$$ 相对于 $$\mathbf{H}_{n−1}$$ 的变化并不大, 即

$$
\begin{align}\\
\underset{\mathbf{H}^{-1}}{\min}\ &||\mathbf{H}^{-1}_n - \mathbf{H}^{-1}_{n-1}||\\
\text{s.t.} & \mathbf{H}^{-1}y_n=s_n\\
& \mathbf{H}^{-1} \text{is symmetric}\\
\end{align}
$$

其中范式为 Weighted Frobenius Nrom, 这个式子的解, 即 BFGS 的 $$\mathbf{QuasiUpdate}$$ 定义为

$$
\mathbf{H}^{-1}_{n+1}=(I-p_n y_n s_n^T) \mathbf{H}^{-1}_n (I - p_n s_n y_n^T)+p_n s_n s_n^T
$$

其中 

$$
p_n = (y_n^T s_n)^{-1}
$$

下面是完整算法

$$
\begin{align}
& \mathbf{BFGSMultiply}(\mathbf{H}^{-1}_0, \{s_k\}, \{y_k\}, d): \\
& \hspace{1em} r \leftarrow d \\
& \hspace{1em} \mbox{// Compute right product} \\
& \hspace{1em} \mbox{for i=n,...,1}: \\
& \hspace{1em} \hspace{1em} \alpha_i \leftarrow \rho_{i} s^T_i r \\
& \hspace{1em} \hspace{1em} r \leftarrow r - \alpha_i y_i \\
& \hspace{1em} \mbox{// Compute center} \\
& \hspace{1em} r \leftarrow \mathbf{H}^{-1}_0 r \\
& \hspace{1em} \mbox{// Compute left product} \\
& \hspace{1em} \mbox{for i=1,\ldots,n}: \\
& \hspace{1em} \hspace{1em} \beta \leftarrow \rho_{i} y^T_i r \\
& \hspace{1em} \hspace{1em} r \leftarrow r + (\alpha_{n-i+1}-\beta)s_i \\
& \hspace{1em} \mbox{return r}
\end{align}
$$


### L-BFGS

BFGS 每次迭代的复杂度 $$O(n^2)$$, 而L-BFGS, limited-mem BFGS, 是 $$O(nm), m=5 ~ 40$$	

### 共轭梯度法

Conjugate gradient, 可认为是梯度下降法和牛顿法的中间物, 希望能加速梯度下降的收敛速度, 同时避免使用海塞矩阵进行求值、储存和求逆获得必要的优化信息. 每次迭代, 沿着共轭方向 (conjugate directions) 执行搜索的, 所以通常该算法要比沿着梯度下降方向优化收敛得更迅速. 共轭梯度法的训练方向是与海塞矩阵共轭的. 

TODO

### Momentum

[An overview of gradient descent optimization algorithms](https://arxiv.org/pdf/1609.04747.pdf)

为解决 SGD 在沟壑（有一维梯度值特别大）的 Z 字形游走问题，引入动量，减少 Z 字震荡

$$
\begin{align}
v_t &= \gamma v_{t-1} + \eta g_t \\
\theta &= \theta - v_t \\
\gamma &\approx 0.9
\end{align}
$$

### Adagrad

使用了自适应技术来更新学习率：对变化[小|大]的参数进行更[大|小]的更新。epsilon 是一个用于抑制学习率产生变动率的常量。[ref](https://zhuanlan.zhihu.com/p/25950802)

Dean 发现它改进 SGD 的鲁棒性，将其应用在大规模神经网络训练 [NIPS12](http://www.cs.toronto.edu/~ranzato/publications/DistBeliefNIPS2012_withAppendix.pdf)

$$
\begin{align} \\
g_{t,i} &= \Delta_\theta J(\theta_i) \\
\theta_{t+1, i} &= \theta_{t,i} - \eta \cdot g_{t,i} (SGD) \\ 
\theta_{t+1,i} &= \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii} + \epsilon}}\cdot g_{t,i}  (Adagrad) \\
\theta_{t+1} &= \theta_{t} - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t \\
\end{align}
$$

其中 $$G_t \in \mathbb{R}^{d \times d}$$  是对角矩阵，元素 $$i,i$$ 是 $$\theta_i$$ 从 $$t^{0}$$ 到 $$t^{i}$$ 的平方和

Adagrad 有个缺点：随着迭代次数的增多，学习率项 $$\frac{\eta}{\sqrt{G_{t,ii} + \epsilon}}$$ 会急剧递减。Adadelta 和 RMSprop 尝试解决这个问题。

### Adadelta

是 Adagrad 的扩展，减少 Adagrad 快速下降的学习率。把 Adagrad 的梯度平方和 $$G_t$$ 限制在时间窗口内

$$
\Delta\theta_t = - \frac{\eta}{\sqrt{ E[g^2]_t + \epsilon}} * g_t \\
E[g^2]_{t} = \gamma E[g^2]_{t-1} + (1-\gamma) g^2_t \\
E[\Delta\theta^2]_t = \gamma E[\Delta\theta^2]_{t-1} + (1-\gamma) \Delta\theta_t^2 \\
RMS[\Delta\theta]_t = \sqrt{E[\Delta\theta^2]_t + \epsilon} \\
\Delta\theta_t = - \frac{RMS[\Delta\theta]_{t-1}}{RMS[g]_t} g_t \\
\theta_{t+1}=\theta_{t} + \Delta\theta_t \\
$$

### RMSprop

类似 Adadelta，解决 Adagrad 快速降低的学习率

$$
E[g^2]_t = 0.9 E[g^2]_{t-1} + 0.1 g_t^2 \\
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} g_t \\
$$

Hinton 建议 $$\gamma = 0.9, \eta=0.001$$ (Hinton's Lecture)[http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf]

### Adam

Adaptive Moment Estimation, 除了像 Adadelta RMSprop 存储之前迭代的快速下降的梯度平方信息 $$v_t$$，它还存储之前迭代的快速下降的梯度信息 $$m_t$$，类似 momentum

$$
\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat m_t &= \frac{m_t}{1-\beta_1^t} \\
\hat v_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat v_t + \epsilon}} \\
\end{align}
$$

其中，[ $$m_t$$ | $$v_t$$ ] 是第 [1|2] 个时刻的梯度估计 [the mean | uncenter variance]，也是 Adam 名字的由来

### 总结

![opt-update-rule](http://wx1.sinaimg.cn/large/62caff97ly1fdszm10lqjj20c70dg3z5.jpg)

## 特征选取

### 自然语言

借助外部数据集训练模型, 如 [WordNet](http://link.zhihu.com/?target=https%3A//wordnet.princeton.edu/), [Reddit评论数据集](http://link.zhihu.com/?target=https%3A//www.kaggle.com/reddit/reddit-comments-may-2015)

基于字母而非单词的NLP特征

衡量文本在视觉上的相似度, 如逐字符的序列比较 (difflib.SequenceMatcher) 

标注单词的词性, 找出中心词, 计算基于中心词的各种匹配度和距离

将产品标题/介绍中 TF-IDF 最高的一些 Trigram 拿出来, 计算搜索词中出现在这些 Trigram 中的比例;反过来以搜索词为基底也做一遍. 这相当于是从另一个角度抽取了一些 Latent 标识

一些新颖的距离尺度, 比如 [Word Movers Distance](http://link.zhihu.com/?target=http%3A//jmlr.org/proceedings/papers/v37/kusnerb15.pdf)

除了 SVD 以外还可以用上 [NMF](http://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Non-negative_matrix_factorization)



### 视觉

预处理, 二值化、灰度、卷积、sobel边缘

衡量美观、明暗、相似度的指标


### 组合特征

KM 世界杯排名预测第一名, 开发了几个特征, 衡量球队的综合能力

### 筛选

Random Forest 的 Imoprtance Feature (原理TODO) , 根据每个特征对信息增益贡献的大小, 来筛选特征. 

## 调参

### Overfitting

表现

* 准确率在训练集高、测试集低、验证集低
* ​泛化误差, 泛化误差上界

避免方法

* 调整训练集、测试集比例

* 调整模型复杂度参数, 如 RandomForest、Gradient Boosting Tree 的深度, CNN 深度

* 正则项, NN 的 dropout maxpool 层, ReLU 单元. 

* 验证集合连续n个迭代分数没有提高, 停止训练

* 5-Fold Cross Validation

* 利用随机性跳出局部最优解: 遗传算法, 何时重启计算问题[Science: An Economics Approach to Hard Computational Problems](https://link.zhihu.com/?target=http%3A//science.sciencemag.org/content/275/5296/51.full)


## Ensemble Generation

将多个不同的 base model 组合成一个 Ensemble Model, 可以同时降低模型的 bias 和 variance, 提高分数、降低 overfitting 风险 [1](http://link.zhihu.com/?target=http%3A//link.springer.com/chapter/10.1007%252F3-540-33019-4_19). 常见方法有

* Bagging, 使用训练数据的不同随机子集来训练每个 Base Model, 最后进行每个 Base Model 权重相同的 Vote. 也即 Random Forest 的原理

* Boosting, 迭代地训练 Base Model, 每次根据上一个迭代中预测错误的情况修改训练样本的权重. 也即 Gradient Boosting 的原理. 比 Bagging 效果好, 但更容易 Overfit

* Blending, 用不相交的数据训练不同的 Base Model, 将它们的输出取 (加权) 平均. 实现简单, 但对训练数据利用少

* Stacking

[5-fold-stacking](http://pic4.zhimg.com/v2-84dbc338e11fb89320f2ba310ad69ceb_r.jpg)



组装要点: 

* Base Model 的相关性尽可能小. Diversity 越大, Bias 越低
* Base Model 的性能表现不能差距太大

## 附录

### 拉格朗日乘数法

最优化问题中, 寻找多元函数在其变量受到一个或多个条件约束时的极值的方法

这种方法可以将一个有 n 个变量与 k 个约束条件的最优化问题, 转换为一个解有 n + k 个变量的方程组问题

如最优化问题

$$
\max f(x,y)\\
s.t.\ g(x,y)=c
$$

转化为求拉格朗日函数的极值

$$
L(x,y,\lambda)=f(x,y)+\lambda \cdot \bigg(g(x,y)-c\bigg)
$$

其中 $$\lambda$$ 是拉格朗日乘数

### Hilbert Space

在一个复数向量空间 $$H$$ 上的给定的内积 $$<.,.>$$ 可以按照如下的方式导出一个范数 $$||.||$$ 

$$
||x|| = \sqrt{\big<x,x\big>}
$$

此空间称为是一个希尔伯特空间，如果其对于这个范数来说是完备的。这里的完备性是指，任何一个柯西列都收敛到此空间中的某个元素，即它们与某个元素的范数差的极限为
0。任何一个希尔伯特空间都是巴拿赫空间，但是反之未必。

### Rank

一个矩阵 $$A$$ 的[列|行]秩是 $$A$$ 的线性独立的纵[列|行]的极大数目。

行秩 == 列秩，统称矩阵的秩

$$A_{m\cdot n}$$ 的秩最大为 $$min(m,n)$$

### 计算

$$
\begin{align}
\log a + \log b &= \log (a \cdot b)\\
\log a - \log b &= \log (\frac {a} {b})\\
\end{align}
$$


$$
\frac{de^x}{dx}=e^x\\
\frac{d\log_{\alpha}{|x|}}{dx}=\frac{1}{x\ln\alpha}
$$

### 方差

variance, 表示一个特征偏离均值的程度

$$
var(X) = \frac{1}{N-1} \sum_{i=1}^{N} (X_i-\overline{X})^2
$$

### 协方差

表示两个特征正相关/负相关的程度

$$
cov(X) = \frac{1}{N-1} \sum_{i=1}^{N} (X_i-\overline{X})(Y_i-\overline{Y})
$$

协方差矩阵

$$
C_{n \times n}, C_{i,j} = cov(Dim_i, Dim_j)
$$

### 逆矩阵

如果 n 阶方形矩阵 $$A$$ 存在 $$B$$ 且 $$A \cdot B = B \cdot A = \I_n$$, 那么称 $$A$$ 是可逆的, $$B$$ 是 $$A$$ 的逆矩阵, 计作 $$A^{-1}$$. 若 $$A$$ 的逆矩阵存在, 称 $$A$$ 为非奇异方阵, 

$$
rank(A)=rank(B)=n \\
(A^{-1})^{-1}=A \\
(\lambda A)^{-1} = \frac{1}{\lambda} \times A^{-1} \\
(AB)^{-1} = B^{-1} A^{-1} \\
(A^T)^{-1}=(A^{-1})^T \\
det(A^{-1}) = \frac{1}{det(A)}
$$

其中 $$det$$ 为行列式

### 正定矩阵

一个 $$n×n$$ 的实对称矩阵 $$M$$ 是正定的，当且仅当对于所有的非零实系数向量 $$z$$，都有 $$z^T M z > 0$$.

一个 $$n×n$$ 的复数对称矩阵 $$M$$ 是正定的，当且仅当对于所有的非零实系数向量 $$z$$，都有 $$z^* M z > 0$$. 其中 $$*$$ 表示共轭转置


## 引用

《统计学习方法》李航

[Spark MLIB GBDT](https://spark.apache.org/docs/latest/mllib-ensembles.html#gradient-boosted-trees-gbts)

[从梯度下降到拟牛顿法: 详解训练神经网络的五大学习算法](http://www.jiqizhixin.com/article/2448)

[Kaggle入门](https://zhuanlan.zhihu.com/p/25742261)

[机器学习面试的那些事儿](https://zhuanlan.zhihu.com/p/22387312)

[浅谈L0,L1,L2范数及其应用](http://t.hengwei.me/post/浅谈l0l1l2范数及其应用.html)

[机器学习中的范数规则化之 (一) L0、L1与L2范数](http://blog.csdn.net/zouxy09/article/details/24971995/)

[ROC和AUC介绍以及如何计算AUC](http://alexkong.net/2013/06/introduction-to-auc-and-roc/)

[机器学习方法：回归（二）：稀疏与正则约束ridge regression，Lasso](http://blog.csdn.net/xbinworld/article/details/44276389)

[理解L-BFGS算法](http://mlworks.cn/posts/introduction-to-l-bfgs/)

[An overview of gradient descent optimization algorithms](https://arxiv.org/pdf/1609.04747.pdf)

## TODO

<!--
点击预估 CTR, Click through rate

归一化: 映射到0-1, 

标准化: 如 z-score标准化, 使均值为0, 带来处理上的便利, 如做SVD分解等价于在原始数据上做PCA, 机器学习中很多函数如Sigmoid、Tanh、Softmax等都以0为中心左右分布;方差为1, 使每维变量在计算距离的时候重要程度相同.  http://www.zhaokv.com/2016/01/normalization-and-standardization.html

每个算法的 loss, 求解方法, 数据要求 (离散、连续) , 优点, 缺点, 参数意义和调整 TODO

LR loss 对数

EM算法

SVM  loss, 优化方法的BFGS推导 Hinge Loss L(x,y)= max(0,  1-yf(x))

决策树 loss  信息增益 剪枝 loss 

GBDT loss

XGBOOST loss

RF Importance Feature

AlexNet结构, 优点

VGG结构, 优点

SVM在哪个地方引入的核函数?如果用高斯核可以升到多少维?

什么是贝叶斯估计

k折交叉验证中k取值多少有什么关系, 和bias和variance有关系吗?

KDTree

Why LR 模型 work

如何GD 避免过拟合
-->
