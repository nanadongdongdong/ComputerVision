## Competition Proposal: Defect Detection
### baseline
#### MLP
class Net(nn.Module):  
def __init__(self):  
super(Net,self).__init__()  
self.conv1=nn.Conv2d(1,6,5)  
self.pool=nn.MaxPool2d(2,2)  
self.conv2=nn.Conv2d(6,16,5)  
self.fc1=nn.Linear(16*4*4,120)  
self.fc2=nn.Linear(120,84)  
self.fc3=nn.Linear(84,10)  

def forward(self,x)  
x=self.pool(F.relu(self.conv1(x)))  
x=self.pool(F.relu(self.conv2(x)))  
x=x.view(-1,16*4*4)  
x=F.relu(self.fc1(x))  
x=F.relu(self.fc2(x))  
x=self.fc3(x)  
return x  
#### resnet50
from torchvision import datasets,models,transforms  
model_resnet50 =models.rsnet50(pretrained=True)  
#### Faster R-CNN  Mask R-CNN  Cascade
pytorch官方模型    预处理图像    边界框   
![](<https://github.com/nanadongdongdong/ComputerVision/blob/main/ODimg/OD.png>)

## Competition Proposal: Image Enhancement
### baseline
#### SRCNN
def srcnn():  
    inputs = keras.layers.Input(shape=(1080, 1920, 3))   
    cnn = keras.layers.Conv2D(64, 9, padding='same', activation='relu')(inputs)  
    cnn = keras.layers.Conv2D(32, 1, padding='same', activation='relu')(cnn)  
    outputs = keras.layers.Conv2D(3, 5, padding='same')(inputs)  
    model = keras.models.Model(inputs=[inputs], outputs=[outputs])  
    model.compile(optimizer=tf.optimizers.Adam(1e-1), loss=tf.losses.mse, metrics=['mse'])  
return model  
#### FSRCNN 
deconvolution
#### ESPCN 
sub-pixel convolution
#### SRGAN
生成网络 VGG 判别网络

## Competition Proposal: User Attributes and Behaviors
### baseline
缺失/异常/冗余，分布一致性，提取重要性特征  
数据预处理：异常值/缺失值/内存优化；  
特征变换：数值特征的归一化/标准化/log变换/cbox-cox变换/连续变量离散化，类别特征的自然数编码/one-hot编码，不规则特征例如身份证；  
特征提取：类别特征的统计比如目标编码/count/nunique/ratio/交叉组合，数值特征的统计比如交叉组合/与类别的交叉组合/行统计，时间特征比如分离维度/时间差，多值特征比如one-hot/countvectorizer/TF_IDF/嵌入表示；  
特征选择：最优子集，相关性分析corr，重要性例如使用树模型评估，封装例如启发式/递归消除，null importance  

## Competition Proposal: Time Series
### baseline
强相关性特征（不同天同一时段，同一时段的前后时段（考虑到有波动性）等）
趋势性特征（差分，差分的差分）
特征强化：操作就是交叉组合或者聚合统计
站点相关特征
历史平移特征：N-1个时间单位内数量
窗口统计特征：mean/meidan/max/min/std

## Competition Proposal: Computational Advertising
### baseline
点击与转化

## Competition Proposal: Regression
### baseline
pandas库和seaborn库探索变量  
常用的回归库文件：  
from sklearn.linear_model import LinearRegression  #线性回归  
from sklearn.neighbors import KNeighborsRegressor  #K近邻回归  
from sklearn.tree import DecisionTreeRegressor     #决策树回归  
from sklearn.ensemble import RandomForestRegressor #随机森林回归  
from sklearn.svm import SVR  #支持向量回归  
import lightgbm as lgb #lightGbm模型  
from sklearn.metrics import mean_squared_error #评价指标  

## Competition Proposal: Binary Classification
### baseline
from sklearn.linear_model import LogisticRegression #逻辑回归  
from sklearn.neighbors import KNeighborsClassifier #k近邻分类  
from sklearn.naive_bayes import GaussianNB #高斯贝叶斯分类  
from sklearn import tree ... tree.DecisionTreeClassifier() #决策树分类  
from sklearn.preprocessing import StandardScaler #归一化  
sklearn.metrics.roc_auc_score #评价指标  
此外：集成学习分类模型：Bagging；Boosting；集成学习投票法；随机森林；LightGBM；极端随机数（ExtraTree） 

## Competition Proposal: Multi-class Classification
### baseline
评估指标：Logloss注重评估的准确性，AUC注重正样本排到前面的能力  

## Proposal: ECG Classification
### 来源
[ECG](<https://blog.csdn.net/qq_15746879/article/details/80329711?spm=1001.2101.3001.6650.11&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-11.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-11.pc_relevant_default&utm_relevant_index=17>)
特征波:P波;QRS波;T波   
预处理：使用Pan-Tompkins算法突出QRS波；滤波去除基线漂移和工频干扰；低通差分滤波器；平方操作使QRS波非线性放大  
调整信号阈值与噪声阈值：T波与QRS波辨别：心拍计数/QRS计数3个以上；R-R间期:心拍计数/QRS计数9个以上  






