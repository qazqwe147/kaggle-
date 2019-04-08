
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.set_printoptions(precision=4, suppress=True)


# In[2]:


df_train=pd.read_csv(r'E:\train.csv')
df_test=pd.read_csv(r'E:\test.csv')


# In[3]:


en_train=df_train.copy()


# In[4]:


df_train.head()


# In[5]:


df_test.head()


# In[6]:


dic={'Id':'Id','SalePrice':'房屋售价','MSSubClass': '建筑的等级','MSZoning': '区域分类','LotFrontage': '距离街道的直线距离','LotArea': '地皮面积','Street': '街道类型','Alley':' 巷子类型','LotShape': '房子整体形状','LandContour': '平整度级别','Utilities': '公共设施类型','LotConfig': '房屋配置','LandSlope': '倾斜度','Neighborhood': '市区物理位置','Condition1': '主干道或者铁路便利程度1','Condition2':'主干道或者铁路便利程度2','BldgType':'住宅类型','HouseStyle': '住宅风格','OverallQual': '整体材料和饰面质量','OverallCond': '总体状况评价','YearBuilt': '建筑年份','YearRemodAdd': '改建年份','RoofStyle': '屋顶类型','RoofMatl': '屋顶材料','Exterior1st': '住宅外墙1st','Exterior2nd': '住宅外墙2nd','MasVnrType': '砌体饰面类型','MasVnrArea': '砌体饰面面积','ExterQual': '外部材料质量','ExterCond': '外部材料的现状','Foundation': '地基类型','BsmtQual': '地下室高度','BsmtCond': '地下室概况','BsmtExposure': '花园地下室墙','BsmtFinType1': '地下室装饰质量type1','BsmtFinSF1': '地下室装饰面积SF1','BsmtFinType2': '地下室装饰质量type2','BsmtFinSF2': '地下室装饰面积SF2','BsmtUnfSF': '未装饰的地下室面积','TotalBsmtSF': '地下室总面积','Heating': '供暖类型','HeatingQC': '供暖质量和条件','CentralAir': '中央空调状况','Electrical': '电力系统','1stFlrSF': '首层面积','2ndFlrSF': '二层面积','LowQualFinSF': '低质装饰面积','GrLivArea': '地面以上居住面积','BsmtFullBath': '地下室全浴室','BsmtHalfBath': '地下室半浴室','FullBath': '高档全浴室','HalfBath': '高档半浴室','BedroomAbvGr': '地下室以上的卧室数量','KitchenAbvGr': '厨房数量','KitchenQual': '厨房质量','Functional': '房屋功用性评级','Fireplaces': '壁炉数量','FireplaceQu': '壁炉质量','GarageType': '车库位置','GarageYrBlt': '车库建造年份','GarageFinish': '车库内饰','GarageCars': '车库车容量大小','GarageArea': '车库面积','GarageQual': '车库质量','GarageCond': '车库条件','PavedDrive': '铺的车道情况','WoodDeckSF': '木地板面积','OpenPorchSF': '开放式门廊区面积','EnclosedPorch': '封闭式门廊区面积','3SsnPorch': '三个季节门廊面积','ScreenPorch': '纱门门廊面积','PoolArea': '泳池面积','PoolQC':'泳池质量','Fence': '围墙质量','MiscFeature': '其他特征','MiscVal': '其他杂项特征值','MoSold': '卖出月份','YrSold': '卖出年份','SaleType': '交易类型','SaleCondition': '交易条件'}


# In[7]:


col_names=pd.Series(df_train.columns)
col_names.isin(dic.keys())


# In[8]:


df_train.rename(columns=dic,inplace=True)


# In[9]:


df_train.head()


# In[10]:


df_test.rename(columns=dic,inplace=True)


# In[11]:


df_test.head()


# In[12]:


df_train.info()


# In[13]:


nu=df_train.isnull().sum()[df_train.isnull().sum()>0]


# In[14]:


nu_train=df_train[nu.index]
nu_train.describe(include='all')


# In[15]:


nu


# In[16]:


print('填补缺失值策略：用已有值的分布来填补缺失值，而不是用单一个值')


# In[17]:


#距离街道的距离可能与‘市区物理位置相关’
df_train['距离街道的直线距离']=df_train.groupby(df_train['市区物理位置'])['距离街道的直线距离'].transform(
    lambda x: x.fillna(x.median()))


# In[18]:


#巷子类型与街道类型和市区位置都相关,且缺失过多，删除
df_train=df_train.drop(' 巷子类型',axis=1)


# In[19]:


df_train['其他特征']=df_train['其他特征'].fillna('None')


# In[20]:


#有关车库的5个特征的缺失值数量是相同的，推测缺失值是因为没有车库
df_train['车库条件']=df_train['车库条件'].fillna('None')
df_train['车库质量']=df_train['车库质量'].fillna('None')
df_train['车库内饰']=df_train['车库内饰'].fillna('None')
df_train['车库建造年份']=df_train['车库建造年份'].fillna('None')
df_train['车库位置']=df_train['车库位置'].fillna('None')


# In[21]:


df_train['泳池面积'].value_counts()


# In[22]:


#泳池面积为0的数量与泳池质量缺失值的数量一样，所以缺失值代表没有泳池。建议把此特征改为有无泳池
df_train['泳池质量']=df_train['泳池质量'].fillna('None')


# In[23]:


#与泳池特征类似
df_train['围墙质量']=df_train['围墙质量'].fillna('None')


# In[24]:


#壁炉这里与泳池质量类似
df_train['壁炉质量']=df_train['壁炉质量'].fillna('None')


# In[25]:


df_train['电力系统']=df_train['电力系统'].fillna(df_train['电力系统'].mode()[0])


# In[26]:


df_train['地下室高度'].describe()


# In[27]:


#地下室高度，地下室概况，花园地下室墙，地下室装饰质量type2，地下室装饰质量type2 ，这5个关于地下室的特征的缺失值数量都是37或38，
#可猜测没有地下室的数量是37或38
df_train['地下室装饰质量type2']=df_train['地下室装饰质量type2'].fillna('None')
df_train['地下室装饰质量type1']=df_train['地下室装饰质量type1'].fillna('None')
df_train['花园地下室墙']=df_train['花园地下室墙'].fillna('None')
df_train['地下室概况']=df_train['地下室概况'].fillna('None')
df_train['地下室高度']=df_train['地下室高度'].fillna('None')


# In[28]:


df_train['砌体饰面面积'].isnull().value_counts()


# In[29]:


#砌体饰面类型，砌体饰面面积  ，情况与地下室类似，缺失都为8
df_train['砌体饰面类型']=df_train['砌体饰面类型'].fillna('None')
df_train['砌体饰面面积']=df_train['砌体饰面面积'].fillna(0)


# In[30]:


df_train.info()


# In[31]:


print('查看各特征的分布')


# In[32]:


import seaborn as sns


# In[33]:


sns.distplot(df_train['房屋售价'])


# In[34]:


df_train['房屋售价']=np.log(df_train['房屋售价']+1)


# In[35]:


sns.distplot(df_train['房屋售价'])


# In[36]:


from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] 
mpl.rcParams['axes.unicode_minus'] = False


# In[37]:


sns.pairplot(df_train)


# In[38]:


print('各特征与目标变量的散点图分析：')
print('1.地皮面积：其直方图90%度的数据集中在第一面元，散点图类似一条垂直于x轴的直线，看上去这一特征属于低方差变量，但也许是因为其离散值大，导致面元太大，应对其离散值做处理')
print('2.Id：明显的，其变化不会引起目标变量的任何变化，可以舍弃')
print('3.建筑等级：类似倒U形的散点图，中间高，两边低，可以做一个二次的转换')
print('4.距离街道的直线距离：分布类似于Id，但从概念上理解，它与目标变量间应该有关系，所以需要进一步分析')
print('5.整体材料和饰面质量、总体状况评价、地下室总面积、首层面积、二层面积、地面以上居住面积、TotRmsAbvGrd、壁炉数量、车库容量大小、车库面积：这些特征与目标变量明显线性相关')
print('6.卖出月份、卖出年份：明显的，其变化不会引起目标变量的任何变化，可以舍弃')
print('7.泳池面积：绝大多数都为0，可以舍弃。概念上与目标变量关系挺大的，但数据上95%的都为0，没有分析的意义')
print('8.其他杂质特征值、纱门门廊面积、三个季节门廊面积：分布上90%以上为0，但还是转换为有无此特征，先别舍弃')
print('9.木地板面积、开放式门廊面积、封闭式门廊面积、地下室以上的卧室数量：从下方看是线性的，但从顶部看，它们与目标变量关系是一条横线，即它们取小值时，目标变量有小值，也有大值，但它们取大值时，目标变量只取大值，没有小值。因此，肯定还有其他特征在影响，可能需要交互')
print('10.关于浴室的几个特征：都是相关的，随着浴室数量的增加，售价增加')
print('11.厨房数量：数量为2的最低价明显高于数量为1的，所以还是有关系的，也许从平均值或中位数来看，是无关的')


# In[39]:


#地皮面积
sns.distplot(df_train['地皮面积'])


# In[40]:


sns.boxplot(df_train['地皮面积'])


# In[41]:


dpmj_sub=df_train['地皮面积'][df_train['地皮面积']<30000]


# In[42]:


sns.distplot(dpmj_sub)


# In[43]:


log_dpmj=np.log(1+df_train['地皮面积'])
log_dpmj_sub=np.log(1+dpmj_sub)
sns.distplot(log_dpmj)


# In[44]:


sns.distplot(log_dpmj_sub)


# In[45]:


#经过log转换后，完整的‘地皮面积’特征数据集分布就已经类似于正态了
df_train['地皮面积']=log_dpmj
sns.regplot(x=df_train['地皮面积'],y=df_train['房屋售价'],data=df_train)


# In[46]:


print('可见，地皮面积与房屋售价更有相关性了，不像之前的，几乎一条直线')


# In[47]:


#删除Id，卖出月份，卖出年份
df_train.drop(columns='Id',inplace=True)


# In[48]:


df_train.drop(columns=['卖出月份','卖出年份'],inplace=True)


# In[49]:


#建筑的等级,从平均值来看，也确实有先上升再下降的趋势
sns.boxplot(x=df_train['建筑的等级'],y=df_train['房屋售价'],data=df_train)


# In[50]:


df_train['fec_建筑的等级']=-np.square(df_train['建筑的等级'])
sns.boxplot(x=df_train['fec_建筑的等级'],y=df_train['房屋售价'],data=df_train)


# In[51]:


df_train['zec_建筑的等级']=np.square(df_train['建筑的等级'])
sns.boxplot(x=df_train['zec_建筑的等级'],y=df_train['房屋售价'],data=df_train)


# In[52]:


print('可见，进行二次转换没有任何效果，先别处理‘建筑的等级’。但为什么建筑等级高，售价反而会下降呢？')


# In[53]:


#把其他杂质特征值、纱门门廊面积、三个季节门廊面积、泳池面积、低质装饰面积 转为二元变量，0表示无，1表示有
er=['其他杂项特征值','纱门门廊面积','三个季节门廊面积','泳池面积','低质装饰面积']
for h in er:
    df_train[df_train[h]==0][h]=0
    df_train[df_train[h]!=0][h]=1


# In[54]:


#距离街道的直线距离
sns.distplot(df_train['距离街道的直线距离'])


# In[55]:


sns.distplot(np.log(df_train['距离街道的直线距离']))


# In[56]:


df_train['log_距离街道的直线距离']=np.log(df_train['距离街道的直线距离'])
sns.regplot(x='log_距离街道的直线距离',y='房屋售价',data=df_train)


# In[57]:


sns.regplot(x='距离街道的直线距离',y='房屋售价',data=df_train)


# In[58]:


df_train['房屋售价'].corr(df_train['log_距离街道的直线距离'])


# In[59]:


df_train['房屋售价'].corr(df_train['距离街道的直线距离'])


# In[60]:


print('log转换并没有增强距离街道的直线距离与目标间的相关性，所以不转换')


# In[61]:


df_train.drop(columns='log_距离街道的直线距离',inplace=True)


# In[62]:


#相关系数矩阵
plt.subplots(figsize=(16,16))
sns.heatmap(df_train.corr(),vmax=0.9,annot=True,square=True)


# In[63]:


#热力图分析
print('1.两两特征间相关系数较高的组：')
print('地面以上居住面积vs TotRmsAbvGrd，地下室总面积vs首层面积，车库车容量大小vs车库面积')
print('-----------------------------')
print('2.与目标变量相关性较大的特征，前10个：')
print('整体材料和饰面质量，地面以上居住面积，车库车容量大小，车库面积 ，地下室总面积，首层面积，高档全浴室，建筑年份，改建年份，TotRmsAbvGrd')
print('-----------------------------')
print('3.从第2点分析得出，房价最重要的两个因素：质量和面积。')
print('与现实矛盾之处：地理位置才是最重要的')
print('-----------------------------')
print('4.矛盾原因：相关系数只能算数值型特征，关于地理位置的几个特征是非数值型的。')


# In[64]:


#数据类型转换


# In[65]:


df_train.info()


# In[66]:


#第一步，找出非数值型特征，对它们转换
obj_col=[]
for c in df_train.columns:
    if (df_train[c].values.dtype!='int64') & (df_train[c].values.dtype!='float64'):
        obj_col.append(c)


# In[67]:


obj_col


# In[68]:


df_obj=df_train.reindex(columns=obj_col)


# In[69]:


df_train['车库建造年份'][df_train['车库建造年份']=='None']=0


# In[70]:


df_train['车库建造年份']=df_train['车库建造年份'].astype('int64')


# In[71]:


df_obj['区域分类'].unique()


# In[72]:


df_obj['区域分类'][df_obj['区域分类']=='RL']=0
df_obj['区域分类'][df_obj['区域分类']=='RM']=1
df_obj['区域分类'][df_obj['区域分类']=='C (all)']=2
df_obj['区域分类'][df_obj['区域分类']=='FV']=3
df_obj['区域分类'][df_obj['区域分类']=='RH']=4


# In[73]:


df_train['区域分类'][df_train['区域分类']=='RL']=0
df_train['区域分类'][df_train['区域分类']=='RM']=1
df_train['区域分类'][df_train['区域分类']=='C (all)']=2
df_train['区域分类'][df_train['区域分类']=='FV']=3
df_train['区域分类'][df_train['区域分类']=='RH']=4


# In[74]:


df_obj['区域分类'].unique()


# In[75]:


df_obj['区域分类'].unique()


# In[76]:


obj_uniq={}
for c in obj_col:
    obj_uniq[c]=df_train[c].unique()


# In[77]:


obj_uniq


# In[78]:


obj_uniq.pop('车库建造年份')


# In[79]:


del obj_uniq['区域分类']


# In[80]:


obj_uniq


# In[81]:


for key,value in obj_uniq.items():
    i=0
    for v in value:
        df_train[key][df_train[key]==v]=i
        i=i+1


# In[82]:


df_train['供暖类型'].unique()


# In[83]:


for c in obj_col:
    df_train[c]=df_train[c].astype('int64')


# In[84]:


sns.barplot(x='区域分类',y='房屋售价',data=df_train)


# In[85]:


fig,axis=plt.subplots(nrows=len(obj_col), ncols=1,figsize=(5,5*len(obj_col)))
i=0
for c in obj_col:
    sns.barplot(x=c,y='房屋售价',data=df_train,ax=axis[i])
    i=i+1


# In[86]:


#分类变量的柱状图分析


# In[87]:


std={}
for c in obj_col:
    ser=df_train.groupby(c)['房屋售价'].mean()
    std[c]=ser.std()


# In[88]:


df_std=pd.Series(std)


# In[89]:


df_std.sort_values(ascending=False)


# In[90]:


print('1.各类之间区别较大的特征,即对房屋售价影响较大的特征：')
print('外部材料质量，厨房质量，地下室高度，地下室概况，主干道或者铁路便利程度2，区域分类 ')
print('-----------------------------------')
print('2.各类之间区别较小的特征,即对房屋售价影响较小的特征：')
print('住宅类型,房子整体形状,围墙质量,公共设施类型,房屋配置,倾斜度')


# In[91]:


print('总结，1.重要的元素：地理位置，材料质量，地下室。')


# In[92]:


#对重要的几个特征，联合起来，更深层的分析


# In[93]:


big_col=['整体材料和饰面质量','地面以上居住面积','车库车容量大小','车库面积' ,'地下室总面积','首层面积','高档全浴室','建筑年份','改建年份','TotRmsAbvGrd','外部材料质量','厨房质量','地下室高度','地下室概况','主干道或者铁路便利程度2','区域分类']


# In[94]:


big_col


# In[95]:


#材料质量分析


# In[96]:


sns.regplot(x=df_train['整体材料和饰面质量'],y=df_train['房屋售价'])


# In[97]:


sns.regplot(x=df_train['外部材料质量'],y=df_train['房屋售价'])


# In[98]:


sns.boxplot(x=df_train['外部材料质量'],y=df_train['房屋售价'])


# In[99]:


print('结论：1.材料质量越高，售价越高，从均值看，质量最好的比最差的高：30%（粗略估计）。2.可推测，外部材料质量，高到底：2>0>1>3')


# In[100]:


#分析地下室


# In[101]:


sns.boxplot(x=df_train['地下室高度'],y=df_train['房屋售价'])


# In[102]:


sns.boxplot(x=df_train['地下室概况'],y=df_train['房屋售价'])


# In[103]:


sns.regplot(x=df_train['地下室总面积'],y=df_train['房屋售价'])


# In[104]:


df_train.drop(df_train[df_train['地下室总面积']>6000].index,inplace=True)


# In[105]:


sns.regplot(x=df_train['地下室总面积'],y=df_train['房屋售价'])


# In[106]:


g=sns.FacetGrid(df_train,row='地下室概况',col='地下室高度')
g.map(plt.scatter,'地下室总面积','房屋售价')


# In[107]:


print('结论：')
print('1.受面积影响较小的细分类：地下室概况=3且地下室高度=1，地下室概况=3且地下室高度=4。（从图中看，地下室概况=3的类别，似乎都这样）')
print('2.多数类别都随着面积增大，售价增大，增的最快的是：地下室概况=0且地下室高度=1')
print('3.平均（整体）售价最高的类别：地下室概况=0且地下室高度=2')


# In[108]:


#面积与位置分析


# In[109]:


sns.regplot(x=df_train['地下室总面积'],y=df_train['地面以上居住面积'])


# In[110]:


print('1.竖直的那条线表示没有地下室。2.同一x下y的最小值构成了一条整齐的斜线，这可能与房屋建造和政府规定相关。3.总体而言，二者正相关')
print('----------------------------')
print('4.既然二者正相关，所以研究一个就可以了')


# In[111]:


g=sns.FacetGrid(df_train,row='主干道或者铁路便利程度2',col='区域分类')
g.map(plt.scatter,'地面以上居住面积','房屋售价')


# In[112]:


print('1.集中性：对主干道或者铁路便利程度2，房屋集中在主干道或者铁路便利程度2=0的类别。对区域分类，较为分散，但最多的是区域分类=0')
print('--------------------------')
print('2.斜率：主干道或者铁路便利程度2=0且区域分类=4，此类斜率比其他类别低。而此类的平均房屋售价属于中等，说明位置不差。那为何面积的影响收到压制，受谁的压制？')
print('--------------------------')
print('3.平均房屋售价：明显地，主干道或者铁路便利程度2=0且区域分类=3整体较高，主干道或者铁路便利程度2=0且区域分类=2整体较低。可推测地理位置的好坏')


# In[113]:


g=sns.FacetGrid(df_train,row='主干道或者铁路便利程度2',col='区域分类')
g.map(plt.scatter,'首层面积','地面以上居住面积')


# In[114]:


print('1.前一个分析的第3点是错误的，从此图看出，面积分布：主干道或者铁路便利程度2=0且区域分类=3整体较高，主干道或者铁路便利程度2=0且区域分类=2整体较低。因此售价是面积的影响，而不仅地理位置。')
print('------------------')
print('2.主干道或者铁路便利程度2=0且区域分类=0，这里的楼层整体较高，因为散布图较厚')
print('------------------')
print('3.主干道或者铁路便利程度2=0且区域分类=3，此类的散布图是两条斜率不同的直线，可推测这里由两种类型的房子构成')


# In[115]:


#房屋时间分析


# In[116]:


fig,axis=plt.subplots(1,1,figsize=(16,5))
sns.boxplot(x=df_train['建筑年份'],y=df_train['地面以上居住面积'])


# In[117]:


fig,axis=plt.subplots(1,1,figsize=(16,5))
sns.boxplot(x=df_train['建筑年份'],y=df_train['房屋售价'])


# In[118]:


print('1.建筑年份对房屋售价的影响，很可能是地面以上居住面积在背后作用。看第一第二图的首尾两部分，地面以上居住面积都比中间部分高，相应房屋售价也比中间部分高')


# In[119]:


#车库


# In[120]:


sns.boxplot(x=df_train['车库车容量大小'],y=df_train['车库面积'])


# In[121]:


fig,axis=plt.subplots(1,3,figsize=(16,5))
sns.barplot(x='区域分类',y='车库面积',data=df_train,ax=axis[0])
sns.barplot(x='区域分类',y='房屋售价',data=df_train,ax=axis[1])
sns.barplot(x='区域分类',y='地面以上居住面积',data=df_train,ax=axis[2])


# In[122]:


print('1.车库对售价的影响小于房屋面积的影响。2.区域分类=2，其房屋面积小，但车库面积大，很奇怪。')


# In[123]:


#对某些数值特征进行log转换


# In[124]:


df_num=df_train.drop(columns=obj_col)


# In[125]:


df_num.info()


# In[126]:


obj_in_num=['建筑的等级','整体材料和饰面质量','总体状况评价','建筑年份','改建年份','地下室全浴室','地下室半浴室','高档全浴室','高档半浴室']


# In[127]:


fig,axis=plt.subplots(nrows=len(df_num.columns), ncols=1,figsize=(5,5*len(df_num.columns)))
i=0
for c in df_num.columns:
    sns.distplot(df_num[c],ax=axis[i])
    i=i+1


# In[128]:


col_log=['距离街道的直线距离','地皮面积','地下室总面积','首层面积','地面以上居住面积','车库面积']


# In[129]:


print('选择要转换的特征：看上去是正态分布，但是有偏斜的。（有些特征，0处有很大的分布，0之外的地方像正态分布，这些特征怎么办？）')


# In[130]:


for c in col_log:
    df_train[c]=np.log(1+df_train[c])


# In[131]:


fig,axis=plt.subplots(nrows=len(col_log), ncols=1,figsize=(5,5*len(col_log)))
i=0
for c in col_log:
    sns.distplot(df_train[c],ax=axis[i])
    i=i+1


# In[132]:


#对分类变量编码


# In[133]:


obj_col


# In[134]:


df_dum=pd.get_dummies(df_train,columns=obj_col,prefix=obj_col)


# In[135]:


df_dum.info()


# In[136]:


df_dum=df_dum.drop(columns='房屋售价')
X=np.array(df_dum)
y=np.array(df_train['房屋售价'])


# In[137]:


#模型训练


# In[138]:


from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel


# In[139]:


#线性模型：Ridge
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
clf=linear_model.Ridge()
model=clf.fit(x_train,y_train)
s=model.score(x_test,y_test)
print('参数：',model.get_params)
print('分数:{:.2f}'.format(s))


# In[140]:


y_pre=model.predict(x_test)
msr=metrics.mean_squared_error(y_test,y_pre)
print('均方误差:{:.2f}'.format(msr))


# In[141]:


clf=linear_model.Ridge()
score_ridge=cross_val_score(clf,X,y,cv=5,scoring='r2')
print('ridge分数：{:.2f}'.format(score_ridge.mean()))


# In[142]:


#非线性模型：SVR;  RBF kernel
clf=svm.SVR(kernel='rbf')
score_svr=cross_val_score(clf,X,y,cv=5,scoring='neg_mean_squared_error')
print('分数:{:.2f}'.format(score_svr.mean()))


# In[143]:


clf=svm.SVR(kernel='rbf')
score_svr=cross_val_score(clf,X,y,cv=5,scoring='r2')
print('分数:{:.2f}'.format(score_svr.mean()))


# In[144]:


model_svr=clf.fit(X,y)
print('linearsvr参数：',model_svr.get_params)


# In[145]:


from sklearn.model_selection import cross_val_predict


# In[146]:


clf=svm.SVR(kernel='rbf')
y_pre=cross_val_predict(clf,X,y,cv=5)
score_2r=metrics.r2_score(y,y_pre)
print('分数:{:.2f}'.format(score_2r))


# In[147]:


clf=svm.SVR(kernel='rbf')
model_svr=clf.fit(X,y)
print('svr参数：',model_svr.get_params)


# In[148]:


clf=svm.LinearSVR()
score_linearsvr=cross_val_score(clf,X,y,cv=5,scoring='r2')
print('linearsvr分数:{:.2f}'.format(score_linearsvr.mean()))


# In[149]:


clf=svm.LinearSVR()
model_linearsvr=clf.fit(X,y)
print('linearsvr参数：',model_linearsvr.get_params)


# In[150]:


print('用SVM模型的分数都很低，不管是线性还是非线性；但是为何与用ridge模型相差那么大？')


# In[151]:


#集成
clf=RandomForestRegressor(n_estimators=50,max_features=80,min_samples_split=2)
score_RFR=cross_val_score(clf,X,y,cv=5,scoring='r2')
print('RFR分数：{:.2f}'.format(score_RFR.mean()))


# In[152]:


clf=RandomForestRegressor(n_estimators=50,max_features=80,min_samples_split=2)
model_RFR=clf.fit(X,y)
print('RFR参数：',model_RFR.get_params)


# In[153]:


clf=GradientBoostingRegressor()
score_GBR=cross_val_score(clf,X,y,cv=5,scoring='r2')
print('GBR分数：{:.2f}'.format(score_GBR.mean()))


# In[154]:


clf=GradientBoostingRegressor()
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
model_GBR=clf.fit(x_train,y_train)
score=model_GBR.score(x_test,y_test)
print('GBR参数：',model_GBR.get_params)
print('GBR分数:{:.2f}'.format(score))


# In[155]:


#对SVM再分析：1.对数据标准化；2.对参数调节


# In[156]:


#把C调大
clf=svm.SVR(kernel='rbf',C=10)
score_svr=cross_val_score(clf,X,y,cv=5,scoring='r2')
print('SVR(C=10)分数:{:.2f}'.format(score_svr.mean()))


# In[157]:


clf=svm.LinearSVR(C=10)
score_linearsvr=cross_val_score(clf,X,y,cv=5,scoring='r2')
print('linearsvr(C=10)分数:{:.2f}'.format(score_linearsvr.mean()))


# In[158]:


#把C调小
clf=svm.SVR(kernel='rbf',C=0.1)
score_svr=cross_val_score(clf,X,y,cv=5,scoring='r2')
print('SVR(C=0.1)分数:{:.2f}'.format(score_svr.mean()))


# In[159]:


clf=svm.LinearSVR(C=0.1)
score_linearsvr=cross_val_score(clf,X,y,cv=5,scoring='r2')
print('linearsvr(C=0.1)分数:{:.2f}'.format(score_linearsvr.mean()))


# In[160]:


clf=svm.LinearSVR(C=0.08)
score_linearsvr=cross_val_score(clf,X,y,cv=5,scoring='r2')
print('linearsvr(C=0.08)分数:{:.2f}'.format(score_linearsvr.mean()))


# In[161]:


clf=svm.LinearSVR(C=1)
score_linearsvr=cross_val_score(clf,X,y,cv=5,scoring='r2')
print('linearsvr(C=1)分数:{:.2f}'.format(score_linearsvr.mean()))


# In[162]:


clf=svm.LinearSVR(C=1)
model_linearsvr=clf.fit(X,y)
print('linearsvr参数：',model_linearsvr.get_params)


# In[163]:


print('意外发现：linearsvr它每次运行会得到不同的分数，且相差极大（很奇怪）。而SVR是正常的。')


# In[164]:


#对数据标准化
scaler=preprocessing.StandardScaler().fit(X)
X_scaler=scaler.transform(X)


# In[165]:


clf=svm.SVR(kernel='rbf',C=1)
score_svr=cross_val_score(clf,X_scaler,y,cv=5,scoring='r2')
print('SVR(C=1)分数:{:.2f}'.format(score_svr.mean()))


# In[166]:


print('结论：对rbf的kernel，数据标准化非常重要')


# In[167]:


#选参数


# In[168]:


#对于Ridge，选alpha 
parameters={'alpha':[0.1,0.5,0.8,1,3,5,7,9]}
ridge=linear_model.Ridge()
clf=GridSearchCV(ridge,param_grid=parameters,cv=5,scoring='r2')
clf.fit(X,y)


# In[169]:


print('最佳参数：',clf.best_params_)
print('最佳分数：',clf.best_score_)


# In[170]:


#试一下RidgeCV
ridgecv=linear_model.RidgeCV(alphas=[0.1,0.5,0.8,1,3,5,7,9])
ridgecv.fit(X,y)


# In[171]:


ridgecv.alpha_


# In[172]:


#绘制验证曲线。（涉及到过拟合，并不是分数越高越好，显然，alpha越大，拟合能力越强，分数自然越高）(错了：因为现在的方数就是验证分数)
trian_scores,valid_scores=validation_curve(linear_model.Ridge(),X,y,param_name='alpha',param_range=[0.1,0.5,0.8,1,3,5,7,9],cv=5,scoring='r2')


# In[173]:


fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot([0.1,0.5,0.8,1,3,5,7,9],trian_scores.mean(axis=1),'bo-',label='trian_scores')
ax.plot([0.1,0.5,0.8,1,3,5,7,9],valid_scores.mean(axis=1),'rs-',label='valid_scores')
plt.legend()
ax.set_xlabel('alpha')
ax.set_ylabel('score')


# In[174]:


print('从图中可看出，还存在更大的alpha，使得验证分数更高')


# In[175]:


#调大alpha值
trian_scores,valid_scores=validation_curve(linear_model.Ridge(),X,y,param_name='alpha',param_range=[9,11,13,15,17,19,22,25],cv=5,scoring='r2')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot([9,11,13,15,17,19,22,25],trian_scores.mean(axis=1),'bo-',label='trian_scores')
ax.plot([9,11,13,15,17,19,22,25],valid_scores.mean(axis=1),'rs-',label='valid_scores')
plt.legend()
ax.set_xlabel('alpha')
ax.set_ylabel('score')


# In[176]:


print('结论：最优alpha=10')


# In[177]:


#对SVM调节参数（它有两个参数：C；gamma）
param_grid={'C':[1,3,5,7,10,13,16],'gamma':[0.001,0.005,0.01,0.03,0.07,0.1]}
svr=svm.SVR()
clf=GridSearchCV(svr,param_grid=param_grid,cv=5,scoring='r2')
clf.fit(X_scaler,y)


# In[178]:


print('SVR最佳分数：{:.2f}'.format(clf.best_score_))
print('SVR最佳参数：',clf.best_params_)


# In[179]:


train_scores,valid_scores=validation_curve(svm.SVR(gamma=0.001),X_scaler,y,cv=5,scoring='r2',param_name='C',param_range=[0.3,0.5,0.8,1,3,5,7,10])
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot([0.3,0.5,0.8,1,3,5,7,10],train_scores.mean(axis=1),'bo-',label='train_scores')
ax.plot([0.3,0.5,0.8,1,3,5,7,10],valid_scores.mean(axis=1),'rs-',label='valid_scores')
ax.set_xlabel('C')
ax.set_ylabel('score')
plt.legend()


# In[180]:


print('在gamma=0.001是，最优的C大概为0.4')


# In[181]:


train_scores,valid_scores=validation_curve(svm.SVR(C=0.4),X_scaler,y,cv=5,scoring='r2',param_name='gamma',param_range=[0.0001,0.0003,0.0007,0.001,0.005])
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot([0.0001,0.0003,0.0007,0.001,0.005],train_scores.mean(axis=1),'bo-',label='train_scores')
ax.plot([0.0001,0.0003,0.0007,0.001,0.005],valid_scores.mean(axis=1),'rs-',label='valid_scores')
ax.set_xlabel('gamma')
ax.set_ylabel('score')
plt.legend()


# In[182]:


print('从图中看出，在C=0.4时，最优的gamma为0.003.')
print('结论，参数最优范围：C=[0.3,0.4,0.5,0.8,1],gamma=[0.0001,0.0003,0.0005,0.0007,0.001]')


# In[183]:


param_grid={'C':[0.5,0.8,1,1.3,1.5,2],'gamma':[0.0001,0.0003,0.0005,0.0007,0.001]}
svr=svm.SVR()
clf=GridSearchCV(svr,param_grid=param_grid,cv=5,scoring='r2')
clf.fit(X_scaler,y)
print('SVR最佳分数：{:.2f}'.format(clf.best_score_))
print('SVR最佳参数：',clf.best_params_)


# In[184]:


print('显然，上一步的结论是错误的。对于要调两个及两个以上参数的模型，应以一组组参数对为单位，而不是单个参数')
print('-----------------------')
print('此例中，C和gamma的作用有互补性：大的C，导致高方差；小的gamma，也导致高方差')


# In[185]:


#对RandomForestRegressor,要调的参数是‘n_estimators’和‘max_features’


# In[186]:


param_grid={'n_estimators':[50,80,100],'max_features':[80,100,120]}
RFR=RandomForestRegressor(min_samples_split=2)
clf=GridSearchCV(RFR,param_grid=param_grid,cv=5,scoring='r2')
clf.fit(X,y)
print('RFR最佳分数：{:.2f}'.format(clf.best_score_))
print('RFR最佳参数：',clf.best_params_)


# In[187]:


param_grid={'n_estimators':[100,120,150],'max_features':[120,150,180]}
RFR=RandomForestRegressor(min_samples_split=2)
clf=GridSearchCV(RFR,param_grid=param_grid,cv=5,scoring='r2')
clf.fit(X,y)
print('RFR最佳分数：{:.2f}'.format(clf.best_score_))
print('RFR最佳参数：',clf.best_params_)


# In[188]:


print('为何分数都是0.88，之前n_estimators=50和max_features=80时，分数；也是0.88')


# In[189]:


RFR=RandomForestRegressor(n_estimators=50,min_samples_split=2)
train_scores,valid_scores=validation_curve(RFR,X,y,cv=5,scoring='r2',param_name='max_features',param_range=[40,60,80,100,120])
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot([40,60,80,100,120],train_scores.mean(axis=1),'bo-',label='train_scores')
ax.plot([40,60,80,100,120],valid_scores.mean(axis=1),'rs-',label='valid_scores')
ax.set_xlabel('max_features')
ax.set_ylabel('score')
ax.set_title('RandomForestRegressor: n_estimators=50')
plt.legend()


# In[190]:


RFR=RandomForestRegressor(max_features=80,min_samples_split=2)
train_scores,valid_scores=validation_curve(RFR,X,y,cv=5,scoring='r2',param_name='n_estimators',param_range=[10,30,50])
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot([10,30,50],train_scores.mean(axis=1),'bo-',label='train_scores')
ax.plot([10,30,50],valid_scores.mean(axis=1),'rs-',label='valid_scores')
ax.set_xlabel('n_estimators')
ax.set_ylabel('score')
ax.set_title('RandomForestRegressor: max_features=80')
plt.legend()


# In[191]:


param_grid={'n_estimators':[20,30,50],'max_features':[60,80,100]}
RFR=RandomForestRegressor(min_samples_split=2)
clf=GridSearchCV(RFR,param_grid=param_grid,cv=5,scoring='r2')
clf.fit(X,y)
print('RFR最佳分数：{:.2f}'.format(clf.best_score_))
print('RFR最佳参数：',clf.best_params_)


# In[192]:


param_grid={'n_estimators':[20,30,50],'max_features':[60,80]}
RFR=RandomForestRegressor(min_samples_split=2)
clf=GridSearchCV(RFR,param_grid=param_grid,cv=5,scoring='r2')
clf.fit(X,y)
print('RFR最佳分数：{:.2f}'.format(clf.best_score_))
print('RFR最佳参数：',clf.best_params_)


# In[193]:


param_grid={'n_estimators':[20,30,50],'max_features':[60]}
RFR=RandomForestRegressor(min_samples_split=2)
clf=GridSearchCV(RFR,param_grid=param_grid,cv=5,scoring='r2')
clf.fit(X,y)
print('RFR最佳分数：{:.2f}'.format(clf.best_score_))
print('RFR最佳参数：',clf.best_params_)


# In[194]:


param_grid={'n_estimators':[20,30,50],'max_features':[30,40]}
RFR=RandomForestRegressor(min_samples_split=2)
clf=GridSearchCV(RFR,param_grid=param_grid,cv=5,scoring='r2')
clf.fit(X,y)
print('RFR最佳分数：{:.2f}'.format(clf.best_score_))
print('RFR最佳参数：',clf.best_params_)


# In[195]:


print('说明一点：没毛病，不是一直0.88')


# In[196]:


#用标准化的数据试试
param_grid={'n_estimators':[20,30,50],'max_features':[60,70,80]}
RFR=RandomForestRegressor(min_samples_split=2)
clf=GridSearchCV(RFR,param_grid=param_grid,cv=5,scoring='r2')
clf.fit(X_scaler,y)
print('RFR最佳分数：{:.2f}'.format(clf.best_score_))
print('RFR最佳参数：',clf.best_params_)


# In[197]:


print('看来标准化数据对随机森林没用')


# In[198]:


#对于GBRT，调参数‘learning_rate’和‘n_estimators’，‘max_depth’
param_grid={'n_estimators':[100,150,200],'max_depth':[2,3,4],'learning_rate':[0.05,0.1,0.15]}
GBRT=GradientBoostingRegressor()
clf=GridSearchCV(GBRT,param_grid=param_grid,cv=5,scoring='r2')
clf.fit(X,y)
print('GBRT最佳分数：{:.2f}'.format(clf.best_score_))
print('GBRT最佳参数：',clf.best_params_)


# In[199]:


clf=GradientBoostingRegressor()
score_GBR=cross_val_score(clf,X,y,cv=5,scoring='r2')
print('GBR分数：{:.2f}'.format(score_GBR.mean()))


# In[200]:


#可见默认的参数已经很好了。现在只调节估计器的个数
train_scores,valid_scores=validation_curve(GradientBoostingRegressor(),X,y,cv=5,scoring='r2',param_name='n_estimators',param_range=[80,100,150,180,200])
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot([80,100,150,180,200],train_scores.mean(axis=1),'bo-',label='train_scores')
ax.plot([80,100,150,180,200],valid_scores.mean(axis=1),'rs-',label='valid_scores')
ax.set_xlabel('n_estimators')
ax.set_ylabel('score')
ax.set_title('GradientBoostingRegressor：其他参数默认')
plt.legend()


# In[201]:


print('最优估计器数量为150个')


# In[202]:


print('调参结论：1.分数都达到0.8-0.9，最终选Ridge（alpha=10），因为它简单，分数也高')
print('----------------------')
print('2.此案例，某些模型（Ridge，随机森林，GBRT）调参的作用并不大，用默认的参数所得分数都很高，接近调参后的结果')
print('----------------------')
print('3.对SVM，数据标准化对分数的影响很大。调参的作用很大，分数从默认参数的0.76上升到0.9')
print('----------------------')
print('4.问题：SVR用kernel=‘linear’，电脑一直运行，但出不来结果；LinearSVR每次Run都会得到不同的结果，很奇怪')


# In[203]:


#补充，标准化数据对L2正则化的线性模型有影响，实验一下
trian_scores,valid_scores=validation_curve(linear_model.Ridge(),X_scaler,y,param_name='alpha',param_range=[0.1,0.5,0.8,1,3,5,7,9],cv=5,scoring='r2')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot([0.1,0.5,0.8,1,3,5,7,9],trian_scores.mean(axis=1),'bo-',label='trian_scores')
ax.plot([0.1,0.5,0.8,1,3,5,7,9],valid_scores.mean(axis=1),'rs-',label='valid_scores')
ax.set_title('Ridge：使用标准化数据')
ax.set_xlabel('alpha')
ax.set_ylabel('score')
plt.legend()


# In[204]:


print('图中的形式是，典型的高方差。尝试调大alpha')


# In[205]:


trian_scores,valid_scores=validation_curve(linear_model.Ridge(),X_scaler,y,param_name='alpha',param_range=[8,10,12,14,16,18,20],cv=5,scoring='r2')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot([8,10,12,14,16,18,20],trian_scores.mean(axis=1),'bo-',label='trian_scores')
ax.plot([8,10,12,14,16,18,20],valid_scores.mean(axis=1),'rs-',label='valid_scores')
ax.set_title('Ridge：使用标准化数据，更大的alpha')
ax.set_xlabel('alpha')
ax.set_ylabel('score')
plt.legend()


# In[206]:


trian_scores,valid_scores=validation_curve(linear_model.Ridge(),X_scaler,y,param_name='alpha',param_range=[10,15,20,25,30,35],cv=5,scoring='r2')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot([10,15,20,25,30,35],trian_scores.mean(axis=1),'bo-',label='trian_scores')
ax.plot([10,15,20,25,30,35],valid_scores.mean(axis=1),'rs-',label='valid_scores')
ax.set_title('Ridge：使用标准化数据，更大大的alpha')
ax.set_xlabel('alpha')
ax.set_ylabel('score')
plt.legend()


# In[207]:


trian_scores,valid_scores=validation_curve(linear_model.Ridge(),X_scaler,y,param_name='alpha',param_range=[20,30,40,50,60,70,80],cv=5,scoring='r2')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot([20,30,40,50,60,70,80],trian_scores.mean(axis=1),'bo-',label='trian_scores')
ax.plot([20,30,40,50,60,70,80],valid_scores.mean(axis=1),'rs-',label='valid_scores')
ax.set_title('Ridge：使用标准化数据，更大大大的alpha')
ax.set_xlabel('alpha')
ax.set_ylabel('score')
plt.legend()


# In[208]:


trian_scores,valid_scores=validation_curve(linear_model.Ridge(),X_scaler,y,param_name='alpha',param_range=[80,90,100,110,120,130,140],cv=5,scoring='r2')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot([80,90,100,110,120,130,140],trian_scores.mean(axis=1),'bo-',label='trian_scores')
ax.plot([80,90,100,110,120,130,140],valid_scores.mean(axis=1),'rs-',label='valid_scores')
ax.set_title('Ridge：使用标准化数据，更大大大大的alpha')
ax.set_xlabel('alpha')
ax.set_ylabel('score')
plt.legend()


# In[209]:


trian_scores,valid_scores=validation_curve(linear_model.Ridge(),X_scaler,y,param_name='alpha',param_range=[100,150,200,250,300,350,400,450,500],cv=5,scoring='r2')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot([100,150,200,250,300,350,400,450,500],trian_scores.mean(axis=1),'bo-',label='trian_scores')
ax.plot([100,150,200,250,300,350,400,450,500],valid_scores.mean(axis=1),'rs-',label='valid_scores')
ax.set_title('Ridge：使用标准化数据，更大大大大大的alpha')
ax.set_xlabel('alpha')
ax.set_ylabel('score')
plt.legend()


# In[210]:


print('结论：Ridge用标准化数据，对比使用原始数据，最优alpha从10增到300。最优分数都是0.9多一点点。')
print('----------------------')
print('数据标准化对L2正则化（Ridge模型中）的影响没有对‘rbf’kernel（SVR模型）的大。')


# In[211]:


#绘制学习曲线，探索数据量
ridge=linear_model.Ridge(alpha=10)
train_size,train_scores,valid_scores=learning_curve(ridge,X,y,train_sizes=[200,400,600,800,1000,1200],cv=10,scoring='r2')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(train_size,train_scores.mean(axis=1),'bo-',label='train_scores')
ax.plot(train_size,valid_scores.mean(axis=1),'rs-',label='valid_scores')
ax.set_xlabel('数据量')
ax.set_ylabel('分数')
ax.set_title('数据量探索')
plt.legend()


# In[212]:


print('从图中可看出：1.验证分数正处于上升趋势，增加数据量可以提高验证分数')
print('-----------------------')
print('2.训练分数的下降形式已经很弱了，就是说，对于此模型，增加数据量最多能使验证分数提高到0.94（但两种分数不可能会完全相等）')


# In[213]:


#特征选取，即删除个别无用特征
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


# In[214]:


#用SelectFromModel，基于树和'L1’
lass=linear_model.LassoCV(alphas=[0.1,1,5,10,15],cv=5)
lass.fit(X,y)


# In[215]:


lass.alpha_


# In[216]:


lass=linear_model.Lasso(alpha=0.1)
lass.fit(X,y)
SFM=SelectFromModel(lass,prefit=True,threshold='0.5*mean')
X_new=SFM.transform(X)


# In[217]:


ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new,y,cv=5,scoring='r2')
print('0.5*mean新数据分数：{:.2f}'.format(score.mean()))


# In[218]:


lass=linear_model.Lasso(alpha=0.1)
lass.fit(X,y)
SFM=SelectFromModel(lass,prefit=True,threshold=0.0001)
X_new=SFM.transform(X)
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new,y,cv=5,scoring='r2')
print('0.01新数据分数：{:.2f}'.format(score.mean()))
print('选取的特征数：',X_new.shape[1])


# In[219]:


lass.coef_


# In[220]:


lass=linear_model.Lasso(alpha=0.0001)
lass.fit(X,y)
lass.coef_


# In[221]:


lass=linear_model.Lasso(alpha=0.0001)
lass.fit(X,y)
SFM=SelectFromModel(lass,prefit=True,threshold='0.5*mean')
X_new=SFM.transform(X)
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new,y,cv=5,scoring='r2')
print('alpha=0.0001的新数据分数：{:.2f}'.format(score.mean()))


# In[222]:


print('结论：同样的0.91分，新数据的特征数为：{:}，原数据的特征数为：{:}'.format(X_new.shape[1],X.shape[1]))


# In[223]:


#不重要特征
ind=SFM.get_support()
no_imp_L1=df_dum.columns[~ind]
no_imp_L1.shape


# In[224]:


no_imp_L1


# In[225]:


print('总结：1.用L1正则化来选特征，要注意‘alpha’应该足够小，否则会让很多特征的系数变成0')
print('------------------')
print('2.SelectFromModel好像不好使用，因为其阀值难以确定，还不如直接从模型中表示特征重要性的属性上，选取')


# In[226]:


#作为有‘coef_’属性的Ridge模型，尝试通过它来进行特征选取
ridge=linear_model.Ridge(alpha=10)
model_ridge=ridge.fit(X,y)
model_ridge.coef_


# In[227]:


print('直接在‘coef_’上选，也不好选，它需要创建一个Series，索引为对应的特征名，然后再对值从小到大排列，还是用RFE，它可以明确要删除几个')


# In[228]:


df_dum.columns


# In[229]:


f_import=pd.Series(data=model_ridge.coef_,index=df_dum.columns)


# In[230]:


f_import=f_import.abs()
f_import=f_import.sort_values(ascending=True)


# In[231]:


no_imp=f_import[:10]


# In[232]:


no_imp.index


# In[233]:


df_new=df_dum.drop(columns=no_imp.index)
X_new=np.array(df_new)


# In[234]:


ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new,y,cv=5,scoring='r2')
print('新数据(from ridge)分数：{:.2f}'.format(score.mean()))


# In[235]:


ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X,y,cv=5,scoring='r2')
print('未删除特征的数据分数：{:.2f}'.format(score.mean()))


# In[236]:


print('结论：删除最不重要的10个特征后，分数没增没减，这是特例还是规律。（按理说，删除特征，并未增加拟合能力，所以分数不增是正常的）')


# In[237]:


print('******特征选取的诉求是不是：在保证分数不降低的前提下，尽可能多地删除不重要特征*****')


# In[238]:


# Ridge.coef_属性与L1正则化对特征选取作用的对比
ridge=linear_model.Ridge(alpha=10)
model_ridge=ridge.fit(X,y)
f_import=pd.Series(data=model_ridge.coef_,index=df_dum.columns)
f_import=f_import.abs()
f_import=f_import.sort_values(ascending=True)
no_imp=f_import[:238]
df_new=df_dum.drop(columns=no_imp.index)
X_new=np.array(df_new)
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new,y,cv=5,scoring='r2')
print('新数据(from ridge，去掉238个)分数：{:.2f}'.format(score.mean()))


# In[239]:


gbr=GradientBoostingRegressor(n_estimators=150)
gbr.fit(X,y)
sfm=SelectFromModel(gbr,prefit=True,threshold='0.5*mean')
X_new_tree=sfm.transform(X)


# In[240]:


ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new_tree,y,cv=5,scoring='r2')
print('新数据(from tree)分数：{:.2f}'.format(score.mean()))


# In[241]:


X_new_tree.shape


# In[242]:


gbr=GradientBoostingRegressor(n_estimators=150)
gbr.fit(X,y)
sfm=SelectFromModel(gbr,prefit=True,threshold='0.8*mean')
X_new_tree=sfm.transform(X)
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new_tree,y,cv=5,scoring='r2')
print('新数据(from tree；0.8*mean)分数：{:.2f}'.format(score.mean()))
print('选取特征数：',X_new_tree.shape[1])


# In[243]:


gbr=GradientBoostingRegressor(n_estimators=150)
gbr.fit(X,y)
sfm=SelectFromModel(gbr,prefit=True,threshold='mean')
X_new_tree=sfm.transform(X)
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new_tree,y,cv=5,scoring='r2')
print('新数据(from tree；1*mean)分数：{:.2f}'.format(score.mean()))
print('选取特征数：',X_new_tree.shape[1])


# In[244]:


gbr=GradientBoostingRegressor(n_estimators=150)
gbr.fit(X,y)
sfm=SelectFromModel(gbr,prefit=True,threshold='1.2*mean')
X_new_tree=sfm.transform(X)
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new_tree,y,cv=5,scoring='r2')
print('新数据(from tree；1.2*mean)分数：{:.2f}'.format(score.mean()))
print('选取特征数：',X_new_tree.shape[1])


# In[245]:


gbr=GradientBoostingRegressor(n_estimators=150)
gbr.fit(X,y)
sfm=SelectFromModel(gbr,prefit=True,threshold='1.5*mean')
X_new_tree=sfm.transform(X)
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new_tree,y,cv=5,scoring='r2')
print('新数据(from tree；1.5*mean)分数：{:.2f}'.format(score.mean()))
print('选取特征数：',X_new_tree.shape[1])


# In[246]:


gbr=GradientBoostingRegressor(n_estimators=150)
gbr.fit(X,y)
sfm=SelectFromModel(gbr,prefit=True,threshold='2*mean')
X_new_tree=sfm.transform(X)
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new_tree,y,cv=5,scoring='r2')
print('新数据(from tree；2*mean)分数：{:.2f}'.format(score.mean()))
print('选取特征数：',X_new_tree.shape[1])


# In[247]:


gbr=GradientBoostingRegressor(n_estimators=150)
gbr.fit(X,y)
sfm=SelectFromModel(gbr,prefit=True,threshold='3*mean')
X_new_tree=sfm.transform(X)
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new_tree,y,cv=5,scoring='r2')
print('新数据(from tree；3*mean)分数：{:.2f}'.format(score.mean()))
print('选取特征数：',X_new_tree.shape[1])


# In[248]:


gbr=GradientBoostingRegressor(n_estimators=150)
gbr.fit(X,y)
sfm=SelectFromModel(gbr,prefit=True,threshold='3.5*mean')
X_new_tree=sfm.transform(X)
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new_tree,y,cv=5,scoring='r2')
print('新数据(from tree；3.5*mean)分数：{:.2f}'.format(score.mean()))
print('选取特征数：',X_new_tree.shape[1])


# In[249]:


gbr=GradientBoostingRegressor(n_estimators=150)
gbr.fit(X,y)
sfm=SelectFromModel(gbr,prefit=True,threshold='4*mean')
X_new_tree=sfm.transform(X)
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new_tree,y,cv=5,scoring='r2')
print('新数据(from tree；4*mean)分数：{:.2f}'.format(score.mean()))
print('选取特征数：',X_new_tree.shape[1])


# In[250]:


gbr=GradientBoostingRegressor(n_estimators=150)
gbr.fit(X,y)
sfm=SelectFromModel(gbr,prefit=True,threshold='6*mean')
X_new_tree=sfm.transform(X)
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new_tree,y,cv=5,scoring='r2')
print('新数据(from tree；6*mean)分数：{:.2f}'.format(score.mean()))
print('选取特征数：',X_new_tree.shape[1])


# In[251]:


print('*******很震惊，特征数从300多降到25，分数仍为0.9******')


# In[252]:


#用条件判断，来达到保证分数不降低，自动选取最少特征数的目的
shape=[]
for n in range(1,100):
    n=0.2*n
    gbr=GradientBoostingRegressor(n_estimators=150)
    gbr.fit(X,y)
    sfm=SelectFromModel(gbr,prefit=True,threshold=n*gbr.feature_importances_.mean())
    X_new_tree=sfm.transform(X)
    ridge=linear_model.Ridge(alpha=10)
    score=cross_val_score(ridge,X_new_tree,y,cv=5,scoring='r2')
    shape.append(X_new_tree.shape[1])
    if score.mean()<0.91:
        break
print('阀值：{:}*mean'.format(n))    
print('分数：{:3f}'.format(score.mean()))
print('分数不低于0.91的最少特征数：',shape[-2])


# In[253]:


col_imp=df_dum.columns[sfm.get_support()]
df_new=df_dum.reindex(columns=col_imp)


# In[254]:


df_new.info()


# In[255]:


print('我感觉有的不是很重要啊。但选择他们的依据是其重要性大于某一阀值。***关键是那个关于重要性的数组是不是固定不变的***')
print('------------------------')
print('***因为SelectFromModel是一次性删除不重要特征，可能会不稳定；用RFE递归地一次删除一个，选到的特征也许更重要***')


# In[256]:


from sklearn.feature_selection import RFECV


# In[257]:


#用RFE，模型仍用GBRT
gbr=GradientBoostingRegressor(n_estimators=150)
rfe=RFECV(gbr,cv=5,scoring='r2',step=5,n_jobs=-1)
rfe.fit(X,y)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (判定系数)")
plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
plt.show()
print("最优特征数: %d" % rfe.n_features_)


# In[258]:


print('***本例中，无用特征数的增加，不会减低分数，这与前面的特征选取情况吻合****')
print('--------------------------------')
print('前面用SelectFromModel，最重要的60个特征就能保证最高分，而这里用RFE递归地选，要90个。***关键是前面的分数是来自Ridge模型，而这里的分数应该是来自GBRT模型***')


# In[259]:


#尝试用Ridge模型，而不是GBRT
ridge=linear_model.Ridge(alpha=10)
rfe=RFECV(ridge,cv=5,scoring='r2',step=1,n_jobs=-1)
rfe.fit(X,y)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (判定系数)")
plt.title('用Ridge模型，而不是GBRT模型')
plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
plt.show()
print("最优特征数: %d" % rfe.n_features_)


# In[260]:


print('从图形来看，最优特征数60多，但其n_features_属性却是384，****所以用RFECV时画图很重要，不能只看n_features_属性***')
print('-----------------------------')
print('***到底删除无用特征能不能提高分数，如果不能，就没必要用RFECV，它计算量太大，且易受微小分数的影响，因为它要选择最高分数（如分数只高一小点，特征数却要大很多，仍选）****')


# In[261]:


# 基于单变量的统计测试来进行特征选取，即用SelectKBest和 f_regression
sk=SelectKBest(f_regression,k=100)
X_new_sk=sk.fit_transform(X,y)
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new_sk,y,cv=5,scoring='r2')
print('选取特征数：',X_new_sk.shape[1])
print('分数：{:.2f}'.format(score.mean()))


# In[262]:


sk=SelectKBest(f_regression,k=120)
X_new_sk=sk.fit_transform(X,y)
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new_sk,y,cv=5,scoring='r2')
print('选取特征数：',X_new_sk.shape[1])
print('分数：{:.2f}'.format(score.mean()))


# In[263]:


s=[]
ns=[]
for n in range(2,20):
    n=10*n
    sk=SelectKBest(f_regression,k=n)
    X_new_sk=sk.fit_transform(X,y)
    ridge=linear_model.Ridge(alpha=10)
    score=cross_val_score(ridge,X_new_sk,y,cv=5,scoring='r2')
    s.append(score.mean())
    ns.append(n)
plt.figure()
plt.plot(ns,s,'ro-')
plt.xlabel('选取的最重要特征数')
plt.ylabel('分数（‘r2’）')
plt.title('用f_regression选取特征，用Ridge训练模型')
plt.show()


# In[264]:


print('可见在特征数为170到达最高分，然后稳定')
print('----------------------------')
print('之前都是基于在训练好的模型内，关于特征重要性的属性，来特征选取；现在是基于单特征与目标变量的统计测试（线性相关程度）来选的（这个应该接近于用相关系数矩阵来选吧）')
print('----------------------------')
print('*****不同的选择方式和标准，选取的特征数量差别很大：60(SelecFromModel,一次性选取)，90(RFE，递归地选取)，170(基于f_regression统计量)****')
print('----------------------------')
print('重要的一点是，哪一种选取方式的***泛化能力***强。显然选取的特征数越多的，越强，但如何得到***特征数不多，分数不低，泛化能力还强***的选取方式？')


# In[265]:


print('接下来，把选取的特征数据拿出了，看能不能对他们做一些更有用的加工处理')


# In[266]:


#用RFE选取65个最重要的特征，然后对他们从新分析
gbr=GradientBoostingRegressor(n_estimators=150)
rfe=RFE(gbr,n_features_to_select=65,step=1)
X_new=rfe.fit_transform(X,y)


# In[267]:


#验证一下分数
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_new,y,cv=5,scoring='r2')
print('分数：{:.2f}'.format(score.mean()))


# In[268]:


col=df_dum.columns[rfe.support_]
df_new=df_dum.reindex(columns=col)


# In[269]:


df_new['房屋售价']=df_train['房屋售价']


# In[270]:


df_new.info()


# In[271]:


plt.figure(figsize=(30,30))
sns.heatmap(df_new.corr(),vmax=0.9,annot=True,square=True)


# In[272]:


print('***可见，模型的特征重要性与相关系数，不完全吻合，存在多个相关系数很小的，只有0.03的特征，但它们仍被认为重要性很强。***')
print('-----------------------------')
print('所选的这些特征中，仍有多对特征，之间相关系数很大。相关系数大，作用一样，那就是多余的')


# In[273]:


#尝试删除4个可能是多余的特征，看看分数会不会下降
col_dr=['车库面积','地下室装饰面积SF1','二层面积','地皮面积']


# In[274]:


df_dr=df_new.drop(columns=col_dr)
df_dr=df_dr.drop(columns='房屋售价')
X_dr=np.array(df_dr)
y=np.array(df_train['房屋售价'])


# In[275]:


ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_dr,y,cv=5,scoring='r2')
print('分数：{:.2f}'.format(score.mean()))


# In[276]:


print('****分数没降，但这是特例还是规律呢****')


# In[277]:


#再随便删除几个特征试试
X_dr_dr=X_dr[:,:50]
ridge=linear_model.Ridge(alpha=10)
score=cross_val_score(ridge,X_dr_dr,y,cv=5,scoring='r2')
print('分数：{:.2f}'.format(score.mean()))


# In[278]:


print('删了最后的10多个特征后，分数仍然是0.91，很奇怪，看来上一步不是规律')

