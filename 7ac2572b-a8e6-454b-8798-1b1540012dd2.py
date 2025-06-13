# # Разработка персонализизации предложения постоянным клиентам, с посследующим анализом

# ## Цель работы: разработать решение, которое позволит персонализировать предложения постоянным клиентам, чтобы увеличить их покупательскую активность.
# 
# *Интернет магазин выявил проблема-активность клиентов начала снижаться.Нам необходимо разработать модель МО, для предсказания снижения покупательской активности и своевременного принятия решения , чтоб это не допустить.Для работы у нас даны 3 датасета, в которых находится информация о покупателях:активность,выручка,прибыль за 3 месяца*
# 
# 
# ***Ход работы***: 
# - *Загрузить данные*
# - *Выполнить предобработку данных*
# - *Провести исследовательский анализ*
# - *Объединить таблицы*
# - *Провести корреляционный анализ*
# - *Создать модель МО*
# - *Выполнить предсказания с последующим анализом*
# - *Написать вывод*

# ## Загрузка данных

# In[1]:





# In[2]:





# In[3]:




# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import phik
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,RobustScaler,MinMaxScaler,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import shap
from sklearn.inspection import permutation_importance


# In[5]:


def load_files(file_paths):
    dataframes={}
    for file in file_paths:
        try:
            df=pd.read_csv(file,sep=None,engine='python')
            dataframes[file]=df
        except Exception as e:
            print(f"Ошибка:{e}")
    return dataframes

files=['/datasets/market_file.csv','/datasets/market_money.csv','/datasets/market_time.csv','/datasets/money.csv']
dataframes=load_files(files)

market_file=dataframes['/datasets/market_file.csv']
market_money=dataframes['/datasets/market_money.csv']
market_time=dataframes['/datasets/market_time.csv']
money=dataframes['/datasets/money.csv']

# In[6]:


df=[market_file,market_money,market_time,money]

for i in df:
    display(i.head())
    print('-'*50)
    display(i.info())
    display(i.describe().T)


# *Данные соответствуют описанию,пропуски отсутствуют*

# ## Предобработка данных

# *Удалим возможные дубликаты,пустые значения и приведём названия столбцов к общему виду*

# In[7]:


def preprocess_data(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.columns=df.columns.str.lower().str.replace(" ","-")
    return df

df=[market_file,market_money,market_time,money]


for i in range(len(df)):
    preprocess_data(df[i])
    display(df[i].head())
    print('-'*60)


# *В таблице morket_money есть явный выброс в столбце "выручка", заменим его на медианное значение*

# In[8]:


market_money.loc[market_money['выручка']>100000]=market_money['выручка'].median()


# *Заменим тип данных в столбце "прибыль",датафрейма money*

# In[9]:


money['прибыль']=money['прибыль'].str.replace(',','.')
money['прибыль']=pd.to_numeric(money['прибыль'])


# In[10]:


market_money['id']=market_money['id'].astype(int)


# *Займёмся неявными дупликатами*

# In[11]:


for col_name,col_data in market_file.items():
    print(f'Столбец:{col_name}')
    print(col_data.unique())
    print("-"*40)


# *Исправим значения в столбце "тип сервиса"*

# In[12]:


market_file['тип-сервиса']=market_file['тип-сервиса'].str.replace('стандартт','стандарт')


# In[13]:


for col_name,col_data in market_money.items():
    print(f'Столбец:{col_name}')
    print(col_data.unique())
    print("-"*40)



# In[14]:


for col_name,col_data in market_time.items():
    print(f'Столбец:{col_name}')
    print(col_data.unique())
    print("-"*40)


# In[15]:


market_time['период']=market_time['период'].str.replace('предыдцщий_месяц','предыдущий_месяц')


# ## Исследовательский анализ

# In[16]:


data=market_file.groupby('покупательская-активность')['id'].count()
ax=data.plot.pie(figsize=(10,10), autopct='%.2f%%')


# *Доля клиентов со сниженным уровнем активности составляет 38.31%-есть над чем работать*


# In[17]:


x_columns=market_file.select_dtypes(include=['number']).columns.to_list()
x_columns.remove('id')

for x in x_columns:
    plt.figure(figsize=(8,5))
    sns.histplot(data=market_file,x=x,hue='покупательская-активность',bins=20,kde=True)
    plt.title(f'Распределение {x} в зависимости от активности')
    plt.show()



# *Исходя из данных распределений можно отметить что преобладает колличество признаков с нормальным распределением , но есть и бимодальные распределения. Маркетинговая активность на пике с конца 3 месяца по начало 4 ,средний просмотр по 2-4 категории,количество просмотренных страниц за визит у клиентов чья активность снизилась-5* 

# *Исследуем данные клиентов чья активность больше или равна 3 месяцам*

# In[18]:


market_file=market_file[market_file['акционные_покупки']>0]
data=market_file.groupby('покупательская-активность')['id'].count()
ax=data.plot.pie(figsize=(10,10), autopct='%.2f%%')


# In[19]:


for x in x_columns:
    plt.figure(figsize=(8,5))
    sns.histplot(data=market_file,x=x,hue='покупательская-активность',bins=20,kde=True)
    plt.title(f'Распределение {x} в зависимости от активности')
    plt.show()


# *Сильных изменений нету , это может говорить о том, что покупательская активность снижается раньше 3 месяцев*

# *Проверим категориальные признаки*

# In[20]:


y_columns=market_file.select_dtypes(exclude=['number']).columns.to_list()
y_columns.remove('покупательская-активность')

for y in y_columns:
    plt.figure(figsize=(8,5))
    sns.barplot(data=market_file,x=y,y=market_file['покупательская-активность'].map({'Снизилась':0,'Прежний уровень':1}))
    plt.title(f'Распределение {y} в зависимости от активности')
    plt.xticks(rotation=45)
    plt.show()


# *Акивных пользователей больше в категории "Мелокой бытовой техники", остальные категории распределены примерно 60% на 40% в пользу актьивных пользователей* 

# **Вывод: Маркетинговая активность снижается у покупателей со сниженной покупательской активностью с 4 месяца,так же покупательская активность снижается после 800 дней с момента регистрации на сайте.Клиенты с низкой покупательской активностью больше покупают товары по акции,просматривают 2-3 категории,6 страниц за визит и за последние 3 месяца в корзине находится до 5 неоплаченых товаров**

# Исследовательский анализ мы делаем для того, чтобы понять, какие закономерности заложены в наших данных. Здесь важно посмотреть на распределения признаков в разрезе целевого признака (снижения активности). 
#     
# Выше ты верно строишь графики, но не делаешь ни одного вывода. А выводы здесь будут важнее всего.
# 
# ***
#     
# По итогам исследовательского анализа (кроме красивой таблицы) у нас должен быть вывод с портретом покупателя, который снижает активность: какие значения каких признаков для него будут характерны. Сколько страниц он в среднем просматривает, больше или меньше это, чем у активных пользователей, также с акционными покупками, временем на сайте и другими признаками. 
#     
# Фактически основной костяк выводов проекта делаем уже здесь. Модель нам нужна для перевода категориального признака (факт снижения активности) в вероятностный (вероятность снижения), чтобы дальше чуть более пластично исследовать выбранный сегмент.

# ## Объединение таблиц

# In[21]:


market_money=market_money.pivot_table(index='id',columns='период',values='выручка',aggfunc='sum').fillna(0)


# In[22]:


market_money=market_money.drop(columns=[4957.5])


# In[23]:


market_money=market_money[(market_money !=0).any(axis=1)]


# In[24]:


market_time=market_time.pivot_table(index='id',columns='период',values='минут',aggfunc='sum').fillna(0)


# In[25]:


market_file=market_file.set_index('id')


# In[26]:


market_file=market_file.merge(market_money,on='id',how='left').merge(market_time,on='id',how='left')


# In[27]:


market_file

# Как видим, у нас во всех таблицах по 1300 уникальных пользователей. В таблице с минутами у нас два периода и 2600 значений (то есть дважды по 1300), в таблице с выручкой три периода и 3900 строк соответственно. После того, как разворачиваем таблицы, убирая периоды в отдельные колонки, в каждой из таблиц будет ровно по 1300 значений.
#     
# Минус три неактивных пользователя, в итоге должно при верном объединении таблицы выйти 1297 строк. Этот момент нужно будет при доработке проверить. Если получается иное количество строк, значит что-то делаешь неправильно.
#     
# У нас в id должны быть уникальные значения. Не должно быть двух строк с одинаковым id. Сейчас они у тебя затраиваются.


# ## Корреляционный анализ

# In[28]:


phik_matrix=market_file.phik_matrix()


# In[29]:


plt.figure(figsize=(10,8))
sns.heatmap(data=phik_matrix,annot=True,cmap='coolwarm',fmt=".2f")
plt.title("Матрица корреляции Phik")
plt.show()


# *Основная мультиколлинеарность нашей целевой категории со столбцом "id", его не стоит исползовать , чтоб модель не путалась, дальше идет "страниц_за_визит"=0,75-но это не критическое значение ,поэтому можно оставить*

# ## Использование пайплайнов

# *Для начала проверим данные на присутствие дисбаланса*

# In[30]:


class_counts=market_file['покупательская-активность'].value_counts()

print(class_counts)
print(class_counts/class_counts.sum())


# *Дисбаланс присутствует поэтому будем использовать метрики классификации f1_score либо roc_auc_score, мы выберем 2 вариант , так как она не чувствительна к дисбалансу*


# In[31]:


RANDOM_STATE=42
TEST_SIZE=0.25

X_train,X_test,y_train,y_test=train_test_split(
    market_file.drop(['покупательская-активность','маркет_актив_6_мес'],axis=1),
    market_file['покупательская-активность'],
    random_state=RANDOM_STATE,
    test_size=TEST_SIZE,
    stratify=market_file['покупательская-активность']
)

ohe_columns=['тип-сервиса','разрешить-сообщать','популярная_категория']

num_columns=([
    'маркет_актив_тек_мес',
    'длительность',
    'акционные_покупки',
    'средний_просмотр_категорий_за_визит',
    'неоплаченные_продукты_штук_квартал',
    'ошибка_сервиса',
    'страниц_за_визит',
    'предыдущий_месяц_x',
    'препредыдущий_месяц',
    'текущий_месяц_x',
    'предыдущий_месяц_y',
    'текущий_месяц_y'
])


# In[32]:


ohe_pipe=Pipeline(
   [('simpleImputer_ohe', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
     ('ohe', OneHotEncoder(drop='first',handle_unknown='ignore', sparse=False))
    ]
    )



# In[33]:


data_preprocessor=ColumnTransformer(
    [('ohe',ohe_pipe,ohe_columns),
    ('num',MinMaxScaler(),num_columns)
    ],
    remainder='passthrough'
)


# In[34]:


final_pipe=Pipeline([
    ('preprocessor',data_preprocessor),
    ('models',DecisionTreeClassifier(random_state=RANDOM_STATE))
])


# In[35]:


param_grid=[
    {'models': [DecisionTreeClassifier(random_state=RANDOM_STATE)],
    'models__max_depth': range(2,5),
    'models__max_features': range(2,5),
    'preprocessor__num':[StandardScaler(),MinMaxScaler(),RobustScaler(),'passthrough']
    },
    
    {'models': [KNeighborsClassifier()],
     'models__n_neighbors': range(2,5),
     'preprocessor__num':[StandardScaler(),MinMaxScaler(),RobustScaler(),'passthrough']
    },
    
    {'models':[LogisticRegression(
        random_state=RANDOM_STATE,
        solver='liblinear',
        penalty='l1'
    )],
     'models__C': range(2,5),
     'preprocessor__num':[StandardScaler(),MinMaxScaler(),RobustScaler(),'passthrough']
    },
    
    {'models':[SVC(
        kernel='poly',
        random_state=RANDOM_STATE,
        degree=2
    )],
    'preprocessor__num':[StandardScaler(),MinMaxScaler(),RobustScaler(),'passthrough']   
    }
    
]


# In[36]:


randomized_search=RandomizedSearchCV(
    final_pipe,
    param_grid,
    cv=5,
    scoring='roc_auc',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

randomized_search.fit(X_train,y_train)

print('Лучшая модель и ее параметры\n\n',randomized_search.best_estimator_)
print('Метрика лучшей модели на тренировочной выборке:',randomized_search.best_score_)



# In[37]:


y_pred=randomized_search.predict_proba(X_test)[:,1]
print(f'Метрика ROC_AUC на тестовой выборке:{roc_auc_score(y_test,y_pred)}')


# **Лучшая модель "LogisticRegression" с параметрами C=4, penalty='l1', random_state=42,solver='liblinear'**

# ## Анализ важности признаков

# *Оценим важность признаков*

# In[38]:


best_model=randomized_search.best_estimator_


# In[39]:


linear_model=best_model.named_steps['models']


# In[40]:


permutation=permutation_importance(best_model,X_test,y_test,scoring='roc_auc')


# In[41]:


feature_importance = pd.DataFrame({'Feature': X_test.columns, 'Importance': permutation['importances_mean']})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
sns.set_style('white')
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 8));



# *Построим график важности*

# In[42]:


X_train_2=final_pipe.named_steps['preprocessor'].fit_transform(X_train)

explainer=shap.Explainer(randomized_search.best_estimator_.named_steps['models'],X_train_2)

X_test_2=final_pipe.named_steps['preprocessor'].transform(X_test)

feature_names=final_pipe.named_steps['preprocessor'].get_feature_names_out()

X_test_2=pd.DataFrame(X_test_2,columns=feature_names)

shap_values=explainer(X_test_2)

shap.plots.beeswarm(shap_values, max_display=16)


# In[43]:


shap.plots.bar(shap_values, max_display=22)


# *Основные значимые признаки 'предыдущий_месяц_y','страниц_за_визит','акционные_покупки','средний_просмотр_категорий_за_визит','текущий_месяц_y','популярная_категория',чем больше этих данных тем ваыше влияние на предсказание*


# ## Сегментация признаков

# *Выполним сегментацию по группе клиентов с высокой вероятностью снижения покупательской активности и наиболее высокой прибыльностью.*

# In[44]:


X_full=pd.concat([X_test,X_train])


# In[45]:


X_full['predict']=randomized_search.predict_proba(X_full)[:,1]


# In[46]:


X_full['общая_выручка']=X_full['предыдущий_месяц_x']+X_full['препредыдущий_месяц']+X_full['текущий_месяц_x']
X_full['покупательская-активность']=market_file['покупательская-активность']


# In[47]:


sns.histplot(X_full['predict'],bins=30,kde=True)
plt.title('Распределение вероятности снижения активности')
plt.xlabel('Вероятность снижения активности')
plt.ylabel('Число клиентов')
plt.show()

sns.histplot(X_full['общая_выручка'],bins=30, kde=True)
plt.title('Распределение прибыли клиентов')
plt.xlabel('Прибыль')
plt.ylabel('Число клиентов')
plt.show()


# *Для точного анализа возьмём клиентов с вероятностью снижения активности от 60% и по прибыли тоже возьмем клиентов выше 60%*

# In[48]:


threshold_proba=X_full['predict'].quantile(0.7)
threshold_profit=X_full['общая_выручка'].quantile(0.7)

selected_segment=X_full[(X_full['predict']>=threshold_proba)&(X_full['общая_выручка']>=threshold_profit)]

print(f'Выбрано {len(selected_segment)} клиентов')



# In[49]:


plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=X_full, 
    x="общая_выручка", 
    y="predict",
    hue="покупательская-активность",
    alpha=0.7,
    palette="coolwarm"
)
plt.axhline(threshold_proba, color='red', linestyle='--', label="Порог вероятности 60%")
plt.axvline(threshold_profit, color='blue', linestyle='--', label="Порог прибыли 60%")
plt.title("Выделение сегмента клиентов по прибыли и вероятности снижения активности")
plt.xlabel("Прибыль")
plt.ylabel("Вероятность снижения активности")
plt.legend()
plt.show()


# *Для исследования возьмём топ 5 по важности признаков Shap, ведь они больше всего влияют на модель*

# In[50]:


top_features=['предыдущий_месяц_y','страниц_за_визит','акционные_покупки','средний_просмотр_категорий_за_визит','текущий_месяц_y','популярная_категория']


# In[51]:


selected_segment


# In[52]:


for x in top_features:
    plt.figure(figsize=(10,8))
    data=selected_segment.groupby(x)['predict'].mean().reset_index()
    sns.barplot(data=data,x=x, y='predict')
    plt.xticks(rotation=45)
    plt.show()


# In[53]:


selected_segment.groupby('популярная_категория')['предыдущий_месяц_x'].mean().reset_index()


# In[54]:


selected_segment.groupby(['популярная_категория'])['препредыдущий_месяц'].mean().reset_index()


# In[66]:


data=selected_segment.groupby(['популярная_категория'])[
    'текущий_месяц_x',
    'препредыдущий_месяц',
    'предыдущий_месяц_x'
].mean().reset_index()



x = np.arange(len(data['популярная_категория']))
width = 0.25 

fig, ax = plt.subplots(figsize=(20, 6))

ax.bar(x - width, data['текущий_месяц_x'], width, label='текущий_месяц_x')
ax.bar(x, data['препредыдущий_месяц'], width, label='препредыдущий_месяц')
ax.bar(x + width, data['предыдущий_месяц_x'], width, label='предыдущий_месяц_x')

# Настройки осей
ax.set_xlabel('Категории товаров')
ax.set_ylabel('Выручка')
ax.set_title('Выручка по категориям товаров за три периода')
ax.set_xticks(x)
ax.set_xticklabels(data['популярная_категория'])
ax.legend()

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# *Препредыдущий месяц был самым плохим , а по выручке всегда больше всего в категории "Кухонная посуда", а проседает категория "Техника для красоты и здоровья"*

# **На основе анализа важных признаком можно сделать вывод о том, что :Чем дольше клиеннт находится на сайте,чем больше категорий просматривает и чем меньше доля акционных товаров у клиента . тем ниже риск снижения клиентской активности. Нужно в первые минуты привлечь внимаение клиента**
 
# `8.1 Выполните сегментацию покупателей. Используйте результаты моделирования и данные о прибыльности покупателей.`
# 
# Попробую описать, что мы должны сделать:
#     
# Самое важное − мы должны по заданию выбрать некоторый сегмент пользователей, обосновать выбор сегмента, обосновать то, как мы этот сегмент определяем (почему выбираем такие значения признаков для отбора пользователей в сегмент), а дальше исследовать только этот сегмент.
#     
# 1) Под результатами моделирования здесь мы понимаем предсказания нашей модели, то есть, вероятности классов. Мы можем ранжировать клиентов по вероятности снижения активности и таким образом использовать эту информацию как одну из осей скаттерплота (для примера). Второй осью тогда будет какая-то категория, которая будет логичной после выбора сегмента. Например, если выбираем сегмент с высокой вероятностью снижения активности и высокой выручкой, то второй шкалой в скаттерплоте будет выручка.
# 
#     
# 2) Важно аргументировать выбор границ и для вероятности снижения активности, и для прибыльности. Как раз это будет удобно сделать, солавшись на график (думаю, скаттерплот тут в качестве типа визуализации будет выигрывать).
#     
#     
# Дальше следует провести исследование для выбранного сегмента: посмотреть на данные в разрезе периодов, в разрезе категорий товаров, акций итд. Нужно выявить факторы, которые сильнее всего влияют на снижение активности и предложить решения для минимизации негативного воздействия этих факторов.
# 


# ## Итоговый отчет по  разработка персонализизации предложения постоянным клиентам, с посследующим анализом

# *Цель работы: разработать решение, которое позволит персонализировать предложения постоянным клиентам, чтобы увеличить их покупательскую активность*
# 
# *Ключевые этапы разработки*
# 1. Маркетинговая активность снижается у покупателей со сниженной покупательской активностью с 4 месяца,так же покупательская активность снижается после 800 дней с момента регистрации на сайте.Клиенты с низкой покупательской активностью больше покупают товары по акции,просматривают 2-3 категории,6 страниц за визит и за последние 3 месяца в корзине находится до 5 неоплаченых товаров
# 2. Основная мультиколлинеарность нашей целевой категории со столбцом "id", его не стоит исползовать , чтоб модель не путалась, дальше идет "страниц_за_визит"=0,75-но это не критическое значение ,поэтому можно оставить
# 3. Лучшая модель "LogisticRegression" с параметрами C=4, penalty='l1', random_state=42,solver='liblinear'
# 4. Основные значимые признаки 'предыдущий_месяц_y','страниц_за_визит','акционные_покупки','средний_просмотр_категорий_за_визит','текущий_месяц_y','популярная_категория',чем больше этих данных тем ваыше влияние на предсказание
# 5. Чем дольше клиеннт находится на сайте,чем больше категорий просматривает. Нужно в первые минуты привлечь внимаение клиента
# 6. Препредыдущий месяц был самым плохим , а по выручке всегда больше всего в категории "Кухонная посуда", а проседает категория "Техника для красоты и здоровья"
# 
# #### Рекоммендации для бизнеса:
# 1. Персонализация для "слабых" клиентов (акции + напоминания).  
# 2. Стимулы для долгосрочных пользователей (бонусы, эксклюзивы).  
# 3. Оптимизация первых минут на сайте (быстрое вовлечение).  
# 4. Уменьшение зависимости от акций через альтернативные мотиваторы.

