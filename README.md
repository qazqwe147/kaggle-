# kaggle房价预测-
本项目主要分为两个部分：                                                                                                                     
第一部分，对数据预处理，填补缺失值，特征转换，对重要特征用图形进行简单的分析；                                                                   
第二部分，对处理好的数据用机器学习模型训练，调节参数，特征选取，数据量分析

一、数据预处理
   1.	把特征的名字用中文替换，这样做的好处是能轻易地通过名字发现特征之间的关系。
   
   2.	填补缺失值。                                                                                                            
   策略有：（1）寻找相关的另一个特征，通过相关特征对待填补特征分组，然后用各组的均值或中位数来填补缺失值；                                          
          （2）进行缺失值分析，探索是什么因素造成了缺失值，如上面的有关车库的5个特征的缺失值数量是相同的，推测缺失值是因为没有车库，所以对缺失值填充                  为‘None’，与地下室相关的特征，与砌体饰面相关的特征等都属于此类型；                                                               
          （3）当特征是分类类型时，可以用特征的众数来填充
          
   3.	用sns.pairplot(df_trian)和热力图矩阵分析各特征与目标特征的相关性。结果如下：                                                           
     （1）两两特征间相关系数较高的组：                                                                                                     
         地面以上居住面积vs TotRmsAbvGrd，地下室总面积vs首层面积，车库车容量大小vs车库面积                                                   
     （2）与目标变量相关性较大的特征，前10个：                                                                                             
          整体材料和饰面质量，地面以上居住面积，车库车容量大小，车库面积 ，地下室总面积，首层面积，高档全浴室，建筑年份，改建年份，TotRmsAbvGrd

   4.	对这些重要的特征用分布图、柱状图、盒图、分面网格等进行图形化分析。

   5.	进行特征转换，对于数值特征，若其发布属于正态，但有偏斜，就用log转换为无偏斜的正态分布；对于分类特征，用pd.get_dummies()转化为虚拟变量；
                                                                                             
二、 模型训练
   1.  用Ridge、SVR、RandomForestRegressor、GradientBoostingRegressor分别对数据进行训练，                                                    
       用GridSearchCV和validation_curve进行参数调节，                                                                                       
       4个模型的验证得分都在0.9左右，最后选择了最简单的Ridge(alpha=10)模型。                                                                   
       过程中发现，数据标准化对SVR的影响很大，其分数从0.7提高到了0.9；
       
   2.  用了多种方法进行特征选择，有基于L1正则化、基于树、基于单变量的统计测试f_regression，                                                      
       用了一次性选取SelectKBest和递归地选取RFE。                                                                                             
       最后发现用不同的方法选取的特征数量差别很大，最少的是用基于树和SelectKBest选了26个特征，分数在0.9；
       
   3.  用learning_curve进行数据量分析，结论是增加数据量可以提高验证分数，极限是提高到0.94，                                                   
       因为验证分数处于上升趋势，而训练分数处于平稳趋势，位置在0.94；

