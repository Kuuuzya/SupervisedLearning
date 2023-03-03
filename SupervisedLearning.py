#!/usr/bin/env python
# coding: utf-8

# **Ссылка на репозиторий:** https://github.com/Kuuuzya/SupervisedLearning

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка данных</a></span></li><li><span><a href="#Исследование-задачи" data-toc-modified-id="Исследование-задачи-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Исследование задачи</a></span></li><li><span><a href="#Борьба-с-дисбалансом" data-toc-modified-id="Борьба-с-дисбалансом-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Борьба с дисбалансом</a></span></li><li><span><a href="#Тестирование-модели" data-toc-modified-id="Тестирование-модели-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Тестирование модели</a></span></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li></ul></div>

# # Отток клиентов

# Из «Бета-Банка» стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.
# 
# Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Вам предоставлены исторические данные о поведении клиентов и расторжении договоров с банком. 
# 
# Постройте модель с предельно большим значением *F1*-меры. Чтобы сдать проект успешно, нужно довести метрику до 0.59. Проверьте *F1*-меру на тестовой выборке самостоятельно.
# 
# Дополнительно измеряйте *AUC-ROC*, сравнивайте её значение с *F1*-мерой.
# 
# Источник данных: [https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling)

# ## Подготовка данных

# Передо мной поставлена задача об анализе исторических данных клиентов Бета-банка. Необходимо проанализировать датасет и с помощью машинного обучения спрогнозировать, уйдёт клиент из банка или нет, чтобы в дальнейшем руководители подразделений смогли предпринять те или иные шаги. Для начала загрузим данные и изучим их.

# In[1]:


#Все импорты, а их будет много, собираю тут, перезапуская ячейку при необходимости
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix,f1_score, recall_score, precision_score, precision_recall_curve,roc_curve, roc_auc_score 
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


# In[2]:


#попробуем считать датасет из файла и положить в переменную. Выдадим ошибку, если это не получится.
try:
    data = pd.read_csv('./Churn.csv')
    print('Датасет загружен в переменную data')
except:
    print('Не удалось загрузить датасет')


# In[3]:


#Посмотрим на размер датасета
data.shape[0]


# 10 тысяч записей. Не идеально, но неплохо, есть, с чем работать. 
# Посмотрим, что там с колонками и все ли данные у нас есть

# In[4]:


data.info()


# Неплохо, бывали датасеты в этом курсе и похуже. 
# Проблема только с колонкой Tenure — сколько лет клиент был в банке.
# Посмотрим на датасет глазами.

# In[5]:


data.head(50)


# Всё чистенько. Но пропуски надо заполнить. Логичным здесь будет заполнение средним значением по колонке.
# Можно, конечно, подставлять среднее по городам или по сегментам зарплат, или по полу. Но это менее 10%,
# так что в данном случае можно пренебречь. Проверим, что у нас там со средним.

# In[6]:


data['Tenure'].mean()


# Почти 5 лет клиент банка. Выглядит логичным, на работу моделей сильно повлиять не должно.
# Округлим до 5, чтобы данные были однородными, а то остальные-то все целые.

# In[7]:


data['Tenure'] = data['Tenure'].fillna(5)


# In[8]:


data.info()


# In[9]:


data['Tenure'].mean()


# И пропусков нет, и среднее почти не изменилось. Зато теперь можно работать с датасетом дальше.

# У нас в датасете есть 3 колонки категориальные. А модели машинного обучения (в нашем случае) с таким работать не умеют. Поэтому переведём их в дамми признаки.
# 
# А фамилии и номера строк вообще удалим, они нам для модели вообще не нужны!

# In[10]:


data = data.drop(columns = ['Surname'], axis=1)
data = data.drop(columns = ['RowNumber'], axis=1)


# In[11]:


data_ohe = pd.get_dummies(data, drop_first=True)


# In[12]:


data_ohe.head()


# Теперь мы получили довольно много колонок (привет, фамилии), зато теперь машинное обучение с этим справится. Давайте попробуем несколько методов и посмотрим, получится ли у нас предсказать результат.

# P.S. Ах да, у нас ещё есть потрясающая колонка CustomerID, которая ничего особо не даёт, но алгоритмы на неё точно будут ориентироваться, ведь это число. Да, по нему можно вычислить, насколько давно с нами клиент, но у нас есть параметр Tenure. Так что, думаю, что от номера клиента точно можно избавиться без сожаления.

# In[13]:


data_ohe = data_ohe.drop(columns = ['CustomerId'], axis=1)


# <div class="alert alert-info"> <b>🎓 Комментарий студента:</b> Спасибо, интересное решение, но мне надо вчитаться и понять.</div>

# In[14]:


data_ohe.head()


# Вот теперь всё должно быть хорошо. Кроме дисбаланса, но его мы проверим чуть позже и придумаем, как с ним быть.

# ## Исследование задачи

# Что ж, мы получили чистенький датасет, попробуем загнать его в модель решающего дерева и посмотреть, вдруг уже всё хорошо и ничего делать не нужно. Для этого разобьём датасет на обучающую и валидационную выборки. Целевой признак у нас — это Exited, ушёл клиент из банка или нет (1 - ушёл, 0 - не ушел)

# In[15]:


#разбиваем датасет на выборку и целевой признак
target = data_ohe['Exited']
features = data_ohe.drop(['Exited'] , axis=1)

#разбиваем выборку и целевой признак на две части 75/25
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.2, random_state=12345)

features_train, features_test, target_train, target_test = train_test_split(
    features_train, target_train, test_size=0.25, random_state=12345)


# In[16]:


print('Размер train:',features_train.shape[0])
print('Размер valid:',features_valid.shape[0])
print('Размер test:',features_test.shape[0])


# Разбили выборки на 60,20 и 20%.

# Т.к. у признаков разный масштаб, а именно, у нас есть возраст и зарплата, например. У них разные порядки. Чтобы привести всё к единому виду, сделаем масштабирование признаков.

# In[17]:


data_ohe.head(5)


# In[18]:


numeric = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']


# In[19]:


scaler = StandardScaler()
scaler.fit(features_train[numeric])


# In[20]:


pd.options.mode.chained_assignment = None

features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])
features_test[numeric] = scaler.transform(features_test[numeric])


# In[21]:


features_train.head(5)


# Конечно, возраст не может быть отрицательным, как и количество лет в банке и зарплата. Но теперь все наши признаки отмасштабированы и они будут корректно влиять на модель.

# In[22]:


#Начнём с логистической регресси
try:
    model_LR = LogisticRegression(random_state=12345, solver='liblinear')
    model_LR.fit(features_train, target_train)
    print('Модель логистической регрессии обучена!')
except:
    print('Модель не обучена, ищи ошибки!')


# In[23]:


#Проверим, как модель предсказывает результаты
predicted_valid_LR = model_LR.predict(features_valid)

# print("R2 на предсказаниях: ", r2_score(target_valid, predicted_valid_LR))
# print("R2 на обучающей выборке: ", model_LR.score(features_train, target_train))
# print("R2 на валидационной выборке: ", model_LR.score(features_valid, target_valid))

print("F1: ", f1_score(target_valid, predicted_valid_LR))


#  ~~Метрика R2 отрицательная, хоть и около 0, но это всё равно говорит об очень низком качестве модели. А вот на обучающей и валидационной выборке всё ок. Попробуем модель решающего дерева и случайного леса. ~~

# Метрика F1 низкая. Значит, эта модель нам не подойдёт. Попробуем дерево.

# In[24]:


try:
    model_DTC = DecisionTreeClassifier(random_state=12345)
    model_DTC.fit(features_train, target_train)
    print('Модель обучающего дерева обучена!')
except:
    print('Модель не обучена, ищи ошибки!')


# In[25]:


#Проверим, как модель предсказывает результаты
predicted_valid_DTC = model_DTC.predict(features_valid)

# print("R2 на предсказаниях: ", r2_score(predicted_valid_DTC, target_valid))
# print("R2 на обучающей выборке: ", model_DTC.score(features_train, target_train))
# print("R2 на валидационной выборке: ", model_DTC.score(features_valid, target_valid))

print("F1: ", f1_score(target_valid, predicted_valid_DTC))


# А вот тут F1 уже неплохая, больше 0.5, при целевом 0.59 это очень неплохое значение.

#  ~~Лучше, но всё равно отрицательное значение, что очень плохо. Попробуем случайный лес, причём сразу с перебором параметров. ~~

# In[26]:


# model_RFR = RandomForestRegressor(random_state=12345, n_estimators=2, max_depth=5)
# model_RFR.fit(features_train, target_train)
# predicted_valid_RFR = model_RFR.predict(features_valid)

    
# print("F1: ", f1_score(target_valid, predicted_valid_RFR))


# In[43]:


get_ipython().run_cell_magic('time', '', 'best_depth = 0\nbest_est = 0\n#best_r2 = -1\nbest_f1 = 0\n\nfor depth in range(1,20,1):\n    for est in range(20,100,10):\n        model_RFC = RandomForestClassifier(random_state=12345, n_estimators=est, max_depth=depth)\n        model_RFC.fit(features_train, target_train)\n        predicted_valid_RFC = model_RFC.predict(features_valid)\n       \n#if f1_score(target_valid,predicted_valid_RFC) > best_f1 :\n#         #best_r2 = r2_score(predicted_valid_RFR, target_valid)\n#         best_f1 = f1_score(target_valid,predicted_valid_RFC)\n#         best_depth = depth\n#         best_est = est\n#         model_RFC = RandomForestClassifier(random_state=12345, n_estimators=best_est, max_depth=best_depth)\n#         model_RFC.fit(features_train, target_train)\n#         predicted_valid_RFC = model_RFC.predict(features_valid)\n        \n    if f1_score(target_valid,predicted_valid_RFC) > best_f1 :\n        best_f1 = f1_score(target_valid,predicted_valid_RFC)\n        best_depth = depth\n        best_est = est\n        best_model = model_RFC\n        best_predict_val = predicted_valid_RFC\n        print(best_f1)\n\nprint(\'Модель случайного леса\')\nprint(\'Лучшая глубина: \', best_depth)\nprint(\'Лучшее кол-во: \', best_est)\n# print(\'R2 на предсказаниях: \', best_r2)\n# print("R2 на обучающей выборке: ", model_RFR.score(features_train, target_train))\n# print("R2 на валидационной выборке: ", model_RFR.score(features_valid, target_valid))\nprint(\'F1: \', best_f1)')


# In[44]:


f1_score(target_valid,best_predict_val)


# Получили 0.57, что уже очень и очень неплохо. Дальше, значит, надо работать со случайным лесом на глубине 17 и 80 деревьями.

# ~~Выше я закомментировал код, т.к. он долго выполняется. Чтобы вы не тратили время на проверку, вот результаты, которые у меня получились.~~

# ~~
# Модель случайного леса
# Лучшая глубина:  9
# Лучшее кол-во:  60
# R2 на предсказаниях:  -0.7142392066291492
# R2 на обучающей выборке:  0.5304141028148517
# R2 на валидационной выборке:  0.37344763273842085
# CPU times: user 2min 34s, sys: 2.48 s, total: 2min 37s
# Wall time: 2min 37s \
# ~~

# ~~Результаты неутешительные. Самые хорошие результаты получились у модели решающего дерева и логистической регрессии. Проверим, что у них с F1-мерой, выберем лучший и пойдём дальше.~~

# In[28]:


print('Дерево')
print('Матрица ошибок')
print(confusion_matrix(target_valid, predicted_valid_DTC))
print('Полнота: ', recall_score(target_valid, predicted_valid_DTC))
print('Точность: ', precision_score(target_valid, predicted_valid_DTC))
print('F1-мера: ', f1_score(target_valid, predicted_valid_DTC))
print('')

print('Логистическая регрессия')
print('Матрица ошибок:')
print(confusion_matrix(target_valid, predicted_valid_LR))
print('Полнота: ', recall_score(target_valid, predicted_valid_LR))
print('Точность: ', precision_score(target_valid, predicted_valid_LR))
print('F1-мера: ', f1_score(target_valid, predicted_valid_LR))
print('')

#добавил лес
print('Случайный лес')
print('Матрица ошибок:')
print(confusion_matrix(target_valid, predicted_valid_RFC))
print('Полнота: ', recall_score(target_valid, predicted_valid_RFC))
print('Точность: ', precision_score(target_valid, predicted_valid_RFC))
print('F1-мера: ', f1_score(target_valid, predicted_valid_RFC))


# Модель леса устраивает нас больше всех. Тут и полнота с точностью приняли хорошие значения значения, и F1-мера, метрика качества классификации, гораздо ближе к требуемому значению в 0.59. С этим можно поработать.

# ## Борьба с дисбалансом

# К сожалению, у нас несбалансированы значения в датасете. Мы это выяснили ещё на старте, но по заданию с дисбалансом нужно разбираться только здесь. Для начала давайте попробуем взвесить классы в модели логистической регрессии.

# In[29]:


#Логистическая регрессия с балансом
try:
    model_LR_b = LogisticRegression(class_weight='balanced', random_state = 12345, solver='liblinear')
    model_LR_b.fit(features_train, target_train)
    predicted_valid_LR_b = model_LR_b.predict(features_valid)
    print('Модель логистической регрессии с балансом обучена!')
except:
    print('Модель не обучена, ищи ошибки!')


# In[30]:


#и то же самое с деревом
try:
    model_DTC_b = DecisionTreeClassifier(class_weight='balanced', random_state=12345)
    model_DTC_b.fit(features_train, target_train)
    predicted_valid_DTC_b = model_DTC_b.predict(features_valid)
    print('Модель обучающего дерева c балансом обучена!')
except:
    print('Модель не обучена, ищи ошибки!')


# In[31]:


#ну и лес тоже теперь
try:
    model_RFC_b = RandomForestClassifier(random_state=12345, n_estimators=50, max_depth=18, class_weight='balanced')
    model_RFC_b.fit(features_train, target_train)
    predicted_valid_RFC_b = model_RFC_b.predict(features_valid)
    print('Модель случайного леса с балансом обучена!')
except:
    print('Модель не обучена!')


# In[32]:


print('Дерево с весами')
print('Матрица ошибок')
print(confusion_matrix(target_valid, predicted_valid_DTC_b))
print('Полнота: ', recall_score(target_valid, predicted_valid_DTC_b))
print('Точность: ', precision_score(target_valid, predicted_valid_DTC_b))
print('F1-мера: ', f1_score(target_valid, predicted_valid_DTC_b))
print('')

print('Логистическая регрессия с весами')
print('Матрица ошибок:')
print(confusion_matrix(target_valid, predicted_valid_LR_b))
print('Полнота: ', recall_score(target_valid, predicted_valid_LR_b))
print('Точность: ', precision_score(target_valid, predicted_valid_LR_b))
print('F1-мера: ', f1_score(target_valid, predicted_valid_LR_b))
print('')
#добавил лес
print('Случайный лес')
print('Матрица ошибок:')
print(confusion_matrix(target_valid, predicted_valid_RFC_b))
print('Полнота: ', recall_score(target_valid, predicted_valid_RFC_b))
print('Точность: ', precision_score(target_valid, predicted_valid_RFC_b))
print('F1-мера: ', f1_score(target_valid, predicted_valid_RFC_b))


# Логистической регрессии балансировка этим методом помогла приблизиться к 0.5 для F1-меры, но дерево всё равно лучше, а лес ещё лучше. У него уже параметр F1 – 0.5681

# Мы опробуем два метода - апсэмплинг и даунсэмплинг выборок. Начнём с увеличения.

# In[33]:


#функция увеличения выборки
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled


# In[34]:


target_train.value_counts()


# Положительных значений - 1219, нулевых - 4781. Значит, нам нужно умножить положительные на 4, чтобы сбалансировать выборку.

# In[35]:


features_upsampled, target_upsampled = upsample(features_train, target_train, 4)


# In[36]:


target_upsampled.value_counts()


# Теперь количество нулей и единиц соотносится, и они у нас разбросаны случайно. Можно работать дальше!

# In[37]:


print(features_upsampled.shape)
print(target_upsampled.shape)

print(features_train.shape)
print(target_train.shape)


# In[39]:


# try:
#     model_DTC_b = DecisionTreeClassifier(random_state=12345, class_weight='balanced')
#     model_DTC_b.fit(features_upsampled,target_upsampled)
#     predicted_valid_DTC_b = model_DTC_b.predict(features_valid)
#     print('Модель обучающего дерева c увеличенной выборкой обучена!')
# except:
#     print('Модель не обучена, ищи ошибки!')


# In[40]:


# print("F1 up:", f1_score(target_valid, predicted_valid_DTC_b))


# In[1]:


#обучим модель на апсемплинговой выборке
try:
    model_RFC_b = RandomForestClassifier(random_state=12345, n_estimators=70, max_depth=18, class_weight='balanced')
    model_RFC_b.fit(features_upsampled,target_upsampled)
    predicted_valid_RFC_b = model_RFC_b.predict(features_valid)
    print('Модель случайного леса с балансом и апсемплингом обучена!')
except:
    print('Модель не обучена!')


# In[48]:


print("F1 up:", f1_score(target_valid, predicted_valid_RFC_b))


# Ого! Получили 0.59. Кажется, это то, чно нам нужно.
# Попробуем ещё другое изменение выборки, вдруг будет лучше. А потом попробуем даунсэмплинг. Data science – это же эээксперименты!

# In[43]:


# best_f1 = 0
# best_rep = 0
# #перебрал разные параметры, эти оказались лучшими
# for rep in range(2,20,1):
#     features_upsampled, target_upsampled = upsample(features_train, target_train, rep)
#     try:
#         model_DTC_b = DecisionTreeClassifier(class_weight='balanced',random_state=12345)
#         model_DTC_b.fit(features_upsampled,target_upsampled)
#         predicted_valid_DTC_b = model_DTC_b.predict(features_valid)
#         #print(rep, 'Модель обучающего дерева c увеличенной выборкой обучена!')
#     except:
#         print('Модель не обучена, ищи ошибки!')
#     if f1_score(target_valid, predicted_valid_DTC_b) > best_f1:
#         best_rep = rep
#         best_f1 = f1_score(target_valid, predicted_valid_DTC_b)
#         print("F1 ", best_rep," up:", f1_score(target_valid, predicted_valid_DTC_b))
        
# print("Лучший F1 ", best_rep," up:", best_f1)


# In[49]:


best_f1 = 0
best_rep = 0
#перебрал разные параметры, эти оказались лучшими
for rep in range(2,15,1):
    features_upsampled, target_upsampled = upsample(features_train, target_train, rep)
    try:
        model_RFC_b = RandomForestClassifier(random_state=12345, n_estimators=80, max_depth=14, class_weight='balanced')
        model_RFC_b.fit(features_upsampled,target_upsampled)
        predicted_valid_RFC_b = model_RFC_b.predict(features_valid)
        #print(rep, 'Модель обучающего дерева c увеличенной выборкой обучена!')
    except:
        print('Модель не обучена, ищи ошибки!')
    if f1_score(target_valid, predicted_valid_RFC_b) > best_f1:
        best_rep = rep
        best_f1 = f1_score(target_valid, predicted_valid_RFC_b)
        print("F1 ", best_rep," up:", f1_score(target_valid, predicted_valid_RFC_b))
        
print("Лучший F1 ", best_rep," up:", best_f1)


# По сравнению со сбалансированной выборкой, можем получить лучший результат на 8 повторениях вместо 4. Но потеряем в скорости выполнения. Поэтому, оставим сбалансированный вариант с апсемплингом в 4 раза, это и логичнее, и результат хороший.

# Мы примем итоговый результат в 0.62238 на 4 повторениях. Посмотрим, что можно сделать с даунсемплингом.

# In[50]:


#функция уменьшения выборки
def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])
    
    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345)
    
    return features_downsampled, target_downsampled


# In[51]:


target_train.value_counts()


# Не знаю, зачем я ещё раз проверяю. Нулей у нас в 4 раза больше. Значит, нужно сделать их в 4 раза меньше, чтобы сбалансировать выборку. Коэффициент будет 0.25.

# In[52]:


features_downsampled, target_downsampled = downsample(features_train, target_train, 0.25)


# In[53]:


target_downsampled.value_counts()


# Так-то лучше!

# In[54]:


# try:
#     model_DTC_b = DecisionTreeClassifier(random_state=12345, class_weight='balanced')
#     model_DTC_b.fit(features_downsampled,target_downsampled)
#     predicted_valid_DTC_b = model_DTC_b.predict(features_valid)
#     print('Модель обучающего дерева c уменьшенной выборкой обучена!')
# except:
#     print('Модель не обучена, ищи ошибки!')


# In[55]:


# print("F1 down:", f1_score(target_valid, predicted_valid_DTC_b))


# In[56]:


#обучим модель на даунсемплинговой выборке
try:
    model_RFC_b = RandomForestClassifier(random_state=12345, n_estimators=80, max_depth=14, class_weight='balanced')
    model_RFC_b.fit(features_downsampled,target_downsampled)
    predicted_valid_RFC_b = model_RFC_b.predict(features_valid)
    print('Модель случайного леса с балансом и даунсемплингом обучена!')
except:
    print('Модель не обучена!')


# In[57]:


print("F1 up:", f1_score(target_valid, predicted_valid_RFC_b))


# И здесь получили подходящий задаче параметр F1 в 0.59. Попробуем поиграться с параметром. Ну так, для практики.

# In[53]:


# #попробуем тоже поиграть с параметрами. Стартуем сразу с F1 = 0.5, меньше нам вообще неинтересно
# best_f1 = 0.5
# best_fr = 0

# #перебрал разные параметры, эти оказались лучшими
# for fr in np.arange(0.1,0.5,0.005):
#     features_downsampled, target_downsampled = downsample(features_train, target_train, fr)
#     try:
#         model_DTC_b = DecisionTreeClassifier(random_state=12345)
#         model_DTC_b.fit(features_downsampled,target_downsampled)
#         predicted_valid_DTC_b = model_DTC_b.predict(features_valid)
#         #print(round(fr,2),'Модель обучающего дерева c уменьшенной выборкой обучена!')
#     except:
#         print('Модель не обучена, ищи ошибки!')
        
#     if f1_score(target_valid, predicted_valid_DTC_b) > best_f1:
#         best_fr = fr
#         best_f1 = f1_score(target_valid, predicted_valid_DTC_b)
#         print("F1 ", round(fr,3)," up:", f1_score(target_valid, predicted_valid_DTC_b))

# print("Лучший F1 ", round(best_fr,3)," up:", best_f1)


# In[58]:


#попробуем тоже поиграть с параметрами. Стартуем сразу с F1 = 0.59, меньше нам вообще неинтересно
best_f1 = 0.59
best_fr = 0

#перебрал разные параметры, эти оказались лучшими
for fr in np.arange(0.25,0.5,0.005):
    features_downsampled, target_downsampled = downsample(features_train, target_train, fr)
    try:
        model_RFC_b = RandomForestClassifier(random_state=12345, n_estimators=80, max_depth=14, class_weight='balanced')
        model_RFC_b.fit(features_downsampled,target_downsampled)
        predicted_valid_RFC_b = model_RFC_b.predict(features_valid)
        #print(round(fr,2),'Модель обучающего дерева c уменьшенной выборкой обучена!')
    except:
        print('Модель не обучена, ищи ошибки!')
        
    if f1_score(target_valid, predicted_valid_RFC_b) > best_f1:
        best_fr = fr
        best_f1 = f1_score(target_valid, predicted_valid_RFC_b)
        print("F1 ", round(fr,3)," up:", f1_score(target_valid, predicted_valid_RFC_b))

print("Лучший F1 ", round(best_fr,3)," up:", best_f1)


# Шикарно. Достигли показателя F1 - 0.6312 при уменьшении количества нулей в выборке почти в 2 раза вместо 4 раз. Это улучшение, т.к. мы не так сильно изменяем исходные данные, значит, за итоговый результат возьмём следующее:**
# - Лучшая модель: Случайный лес
# - Глубина: 14
# - Количество деревьев: 80
# - Баланс: с помощью даунсэмплинга в 2 раза (0.455)
# - F1: 0.632302405

# ~~Мне не удалось достичь требуемого по заданию показателя в 0.59. Возможно, я не учёл все параметры балансировки или не вспомнил о каких-то методах, которые позволят это сделать. Но результат 0.5838 считаю хорошим, поэтому сделаю тестирование этой модели и буду надеяться, что ревьювер подскажет, где я ошибся (хотя в реальном проекте такого не будет, но мы же здесь только учимся, правда?).~

# ## Тестирование модели

# Посмотрим на результаты тестирования модели различными методами. Сначала только ещё раз перезапустим модель, чтобы точно работать с тем, что мы выбрали.

# In[60]:


get_ipython().run_cell_magic('time', '', "try:\n    features_downsampled, target_downsampled = downsample(features_train, target_train, 0.49)\n    model = RandomForestClassifier(random_state=12345, n_estimators=80, max_depth=14, class_weight='balanced')\n    model.fit(features_downsampled,target_downsampled)\n    predicted_test = model.predict(features_test)\n    print('Итоговая модель обучена, предсказания сделаны')\nexcept:\n    print('Что-то пошло не так!')\n\nprint('F1 test:', f1_score(target_test, predicted_test))")


# In[63]:


print('F1:',f1_score(target_test, predicted_test))
print('Точность:',precision_score(target_test, predicted_test))
print('Полнота:',recall_score(target_test, predicted_test))


# Неидеально, конечно. Мы предсказываем верно, что человек уйдёт, в 61% случаях. Идеально было бы 100%, но это скорее всего невозможно.

# In[64]:


predictions_train = model.predict(features_train)
predictions_valid = model.predict(features_valid) 

print("MAE на обучающей выборке: ", mean_absolute_error(target_train,predictions_train))
print("MAE на валидационной выборке: ", mean_absolute_error(target_valid, predictions_valid))
print("MAE на тестовой выборке: ", mean_absolute_error(target_test, predicted_test))


# В среднем, мы ошибаемся на 0.16. Т.к. наши значения это 0 и 1, то можно сказать, что мы ошибаемся и в случае 0, и в случае 1, то есть, суммарно около 32%, что как раз соответствует полноте предсказаний.

# In[65]:


probabilities_test = model.predict_proba(features_test)
precision, recall, thresholds = precision_recall_curve(target_test, probabilities_test[:, 1])

plt.figure(figsize=(5, 5))
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Кривая Precision-Recall')
plt.show() 


# In[66]:


probabilities_test = model.predict_proba(features_test)
probabilities_one_test = probabilities_test[:, 1]

fpr, tpr, thresholds = roc_curve(target_test, probabilities_one_test) 

plt.figure()

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
# ROC-кривая случайной модели (выглядит как прямая)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.show();


# In[60]:


probabilities_test = model.predict_proba(features_test)
probabilities_one_test = probabilities_test[:, 1]

auc_roc = roc_auc_score(target_test, probabilities_one_test)

print(auc_roc);


# Наша модель отличается от случайной и это прекрасно. Значит, она предсказывает иначе. И т.к. она лежит сверху от случайного графика, то и предсказывает она лучше. Площадь под графиком нашей моделью равна 0.84, а у случайной модели — 0.5. Значит, мы на 34% лучше предсказываем результат. Неидеально, но для теста, на мой взгляд, весьма неплохо. 
# 
# А если нам удастся поработать с 60% клиентов, кто хочет уйти и хотя бы 50% оставить (мы не знаем процент конверсии, но пусть будет такой), это очень и очень неплохо, согласитесь!

# ## Чек-лист готовности проекта

# Поставьте 'x' в выполненных пунктах. Далее нажмите Shift+Enter.

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке исполнения
# - [x]  Выполнен шаг 1: данные подготовлены
# - [x]  Выполнен шаг 2: задача исследована
#     - [x]  Исследован баланс классов
#     - [x]  Изучены модели без учёта дисбаланса
#     - [x]  Написаны выводы по результатам исследования
# - [x]  Выполнен шаг 3: учтён дисбаланс
#     - [x]  Применено несколько способов борьбы с дисбалансом
#     - [x]  Написаны выводы по результатам исследования
# - [x]  Выполнен шаг 4: проведено тестирование
# - [x]  Удалось достичь *F1*-меры не менее 0.59
# - [x]  Исследована метрика *AUC-ROC*
