import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

crop=pd.read_csv("static/Crop_recommendation.csv")

crop_items={
        'rice'          :1,
        'maize'         :2,
        'jute'          :3,
        'cotton'        :4,
        'coconut'       :5,
        'papaya'        :6,
        'orange'        :7,
        'apple'         :8,
        'muskmelon'     :9,
        'watermelon'    :10,
        'grapes'        :11,
        'mango'         :12,
        'banana'        :13,
        'pomegranate'   :14,
        'lentil'        :15,
        'blackgram'     :16,
        'mungbean'      :17,
        'mothbeans'     :18,
        'pigeonpeas'    :19,
        'kidneybeans'   :20,
        'chickpea'      :21,
        'coffee'        :22
    }
crop['crop_num']=crop['label'].map(crop_items)

crop.drop('label',axis=1,inplace=True)

x=crop.drop('crop_num',axis=1)
y=crop['crop_num']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

mScaler=MinMaxScaler()

mScaler.fit(x_train)
x_train=mScaler.transform(x_train)
x_test=mScaler.transform(x_test)


sScaler=StandardScaler()

sScaler.fit(x_train)
x_train=sScaler.transform(x_train)
x_test=sScaler.transform(x_test)

#taining

dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
ypred=dtc.predict(x_test)


def recommendation(data):
        
        prediction=dtc.predict(data).reshape(1,-1)

        return prediction[0]
    
    
        
N=127
P=27
K=208
temperature=28
humidity=89
pH=6
rainfall=242

data = np.array([[N,P, K, temperature, humidity, pH, rainfall]])



data=mScaler.transform(data)


data=sScaler.transform(data)





predict=recommendation(data)



crop_dict={
        1:'rice',
        2:'maize',
        3:'jute',
        4:'cotton',
        5:'coconut',
        6:'papaya',
        7:'orange',
        8:'apple',
        9:'muskmelon',
        10:'watermelon',
        11:'grapes',
        12:'mango',
        13:'banana',
        14:'pomegranate',
        15:'lentil',
        16:'blackgram',
        17:'mungbean',
        18:'mothbeans',
        19:'pigeonpeas',
        20:'kidneybeans',
        21:'chickpea',
        22:'coffee'        
    }

if predict[0] in crop_dict:
        crop=crop_dict[predict[0]]
        success="{} is the best crop to grow".format(crop)
        print(success)
else:
        sorry="Sorry,we are unable to recommend a crop based on the given data"
        print(sorry)