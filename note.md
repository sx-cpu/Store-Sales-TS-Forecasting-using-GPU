# 我不理解确定性过程的前因后果

## 1. 时间序列里的成分
时间序列通常由三部分组成：
1. 趋势（Trend）： 长期的上升或下降，比如GDP随时间增长。
2. 季节性（Seasonality）：周期性的波动，比如每年夏天电费更高。
3. 随机波动（Noise）：不可预测的成分，比如突发事件。

👉 在统计建模或预测中，我们通常会先把趋势和季节性解释掉，剩下的才交给更加复杂的模型。

---

## 2. 确定性成分和随机成分
确定性成分是可以用固定公式描述，比如线性趋势、二次趋势、季节性。
随机成分是无法用数学模型解释的部分，比如噪声。不可预测，只能用概率模型来解释。

---

## 3. 为什么要用 DeterministicProcess
在 Python 的` statsmodels `里，`DeterministicProcess` 是一个工具，帮你自动生成时间序列的确定性特征矩阵。

---

## 4. 前因（为什么要做）
- 如果直接对原始序列建模，趋势和季节性会干扰模型。
- 通过 DeterministicProcess，我们能显式建模这些确定性因素，把它们剥离出来。
- 最终只让模型去解释真正的随机部分。

---

## 5. 后果（用完能干啥）
1. 趋势拟合：得到长期走向曲线
2. 未来预测：可以生成未来的 trend 和 seasonality 特征，再用拟合好的模型外推。
3. 残差分析：去掉趋势和季节性后，剩下的残差更接近“平稳序列”，便于用 ARIMA 等方法建模。


---


# 确定性过程（DeterministicProcess）原理与逻辑

## 1. 作用定位

在 Python 的 `statsmodels` 里，`DeterministicProcess` 是一个工具类，用于：

- 自动生成时间序列的 **确定性特征矩阵**（Deterministic Features）  
- 常用于回归建模中，提取 **趋势（trend）**、**截距（const）**、**季节性（seasonality）** 等可预测成分  
- 只处理确定性部分，不涉及随机噪声

公式上，时间序列通常可表示为：
$$ y_t = Trend_t + Seasonality_t + Noise_t $$


- `DeterministicProcess` 负责生成 `Trend_t + Seasonality_t` 的特征  
- `Noise_t` 留给后续随机模型处理（如 ARIMA 或神经网络）

---

## 2. 原理

1. **输入**：时间索引 `index`  
2. **参数**：
   - `constant=True/False` → 是否生成截距列（全为 1）  
   - `order=n` → 生成 n 阶趋势（线性 n=1，二次 n=2，…)  
   - `seasonal=True/False` → 是否生成季节性虚拟变量  
   - `additional_terms=[...]` → 可加入傅里叶项或自定义确定性特征  
   - `drop=True` → 自动去掉线性相关的列，避免共线性  
3. **输出**：
   - `in_sample()` → 样本内确定性特征矩阵  
   - `out_of_sample(steps=n)` → 样本外预测特征矩阵，可用于未来外推

---

## 3. 使用逻辑

1. **生成确定性特征矩阵**

```python
from statsmodels.tsa.deterministic import DeterministicProcess
dp = DeterministicProcess(index=y.index, constant=True, order=1)
X_in = dp.in_sample()  # 样本内特征
```

---

# DeterministicProcess 特征矩阵详解

## 1. 什么是特征矩阵

在统计建模或机器学习里：

- **特征矩阵（X）**：每一行对应一个观测点，每一列对应一个可解释变量（feature）  
- 在时间序列里，特征矩阵由 **确定性成分** 构成，例如截距、趋势、季节性、傅里叶项等  

公式表示：
$$ y_t ≈ β0 * const + β1 * trend + β2 * season1 + β3 * season2 + ... + ε_t $$

- `const`, `trend`, `season1`, `season2` 等就是特征矩阵的列

---

## 2. 特征矩阵组成

假设时间序列按月采样，有 12 个月：

| 日期       | const | trend | season_Jan | season_Feb | ... | season_Dec |
|------------|-------|-------|------------|------------|-----|------------|
| 2020-01-01 | 1     | 0     | 1          | 0          | ... | 0          |
| 2020-02-01 | 1     | 1     | 0          | 1          | ... | 0          |
| 2020-03-01 | 1     | 2     | 0          | 0          | ... | 0          |
| ...        | ...   | ...   | ...        | ...        | ... | ...        |

- **const**：截距列，全 1  
- **trend**：趋势列（0,1,2,...），由 `order` 参数控制  
- **season_* **：季节性虚拟变量，由 `seasonal=True` 自动生成  
- **其他列**：`additional_terms` 可以加入傅里叶项、假日指标等自定义列  

每一行对应时间序列的一个观测点，每一列对应一个可解释因素。

---

## 3. 样本内 vs 样本外特征矩阵

- **样本内 (`in_sample()`)**  
  - 行数 = 时间序列长度  
  - 用于回归拟合趋势/季节性  

- **样本外 (`out_of_sample(steps=n)`)**  
  - 行数 = `steps`（未来时间点数）  
  - 列与样本内一致，用于预测未来趋势/季节性  

---

## 4. 特征矩阵的作用

1. **提取趋势和季节性**  
   - X 矩阵包含了时间规律性  
   - 通过线性回归拟合 β 系数，得到趋势和季节性贡献  

2. **未来预测**  
   - 使用 `out_of_sample()` 生成的矩阵，回归模型直接预测未来趋势  

3. **分离随机成分**  
   - 拟合确定性成分后，残差 ε_t 更平稳，便于 ARIMA、神经网络等建模  

---

## 5. Python 示例

```python
import pandas as pd
from statsmodels.tsa.deterministic import DeterministicProcess

# 时间序列索引
dates = pd.date_range(start='2020-01-01', periods=12, freq='M')

# 构造确定性过程
dp = DeterministicProcess(index=dates, constant=True, order=1, seasonal=True)

# 样本内特征矩阵
X_in = dp.in_sample()

# 样本外特征矩阵，预测未来3步
X_out = dp.out_of_sample(steps=3)

print("样本内特征矩阵：\n", X_in.head())
print("\n样本外特征矩阵：\n", X_out)

```

---

# Periodogram
Periodogram（周期图/谱图）。其实就是把时间序列拆开，看里面不同周期成分的强弱。

## 1. 时间序列 = 趋势 + 季节性/周期性 + 随机波动。
举例：
- 每年圣诞节，零售额都会上升 → 年周期
- 每周末，超市客流量会增加 → 周周期
- 股票每天都有高频波动 → 高频成分

我们想知道 这些周期性 是否存在，以及有多强。
这时就用到 频率分析（Frequency An   alysis）。

## 2. 频率分析
高频 → 短周期（变化快）
低频 → 长周期（变化慢）

## 3. 计算方法 
### 1. 时间序列和傅里叶分解
假设我们有一个长度为 \(N\) 的离散时间序列 \(\{x_t\}, t=0,1,\dots,N-1\)。  

任何序列都可以表示为一堆 **正弦波和余弦波** 的叠加（傅里叶级数）：  

\[
x_t \approx \sum_{k=0}^{N-1} A_k \cos(2\pi f_k t) + B_k \sin(2\pi f_k t)
\]

其中：
- \(f_k = \frac{k}{N}\)：频率（每步采样能产生多少个周期）。  
- 系数 \(A_k, B_k\)：表示该频率分量的“强度”。  

---

### 2. 傅里叶变换和周期图
使用 **离散傅里叶变换 (DFT)**：

\[
X(f_k) = \sum_{t=0}^{N-1} x_t \, e^{-i 2\pi f_k t}
\]

这里：
- \(X(f_k)\)：频率 \(f_k\) 的复数系数（包含幅度和相位）。  

---

### 3. 功率谱（Periodogram）
**周期图 = 功率谱估计：**

\[
I(f_k) = \frac{1}{N} \, \big|X(f_k)\big|^2
\]

解释：
- \(|X(f_k)|^2\)：频率 \(f_k\) 的能量（幅度平方）。  
- 除以 \(N\)：让它和原序列的方差对应。  

👉 **功率谱强度 (spectral power)** = 这个频率分量对整体方差的贡献。  

---

### 4. 方差分解
时间序列的总方差可写为所有频率的谱强度之和：

\[
\text{Var}(x_t) \approx \sum_{k=0}^{N-1} I(f_k)
\]

这说明：  
- **周期图展示了方差如何分布在不同频率上。**  
- 峰值高 → 该周期性成分最重要。  

---