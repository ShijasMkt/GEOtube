from django.shortcuts import redirect, render,HttpResponse
from django.contrib import messages


from app1.models import users

# Create your views here.

def IndexPage(request):
    if 'user_id' in request.session:
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.preprocessing import StandardScaler  
            from sklearn.tree import DecisionTreeClassifier
            import warnings

            warnings.filterwarnings("ignore", category=UserWarning)

            crop=pd.read_csv("static/Crop_recommendation.csv")
            features = crop[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
            target = crop['label']

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

            x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=23)

            mScaler=MinMaxScaler()

            mScaler.fit(x_train,y_train)
            x_train=mScaler.transform(x_train)
            x_test=mScaler.transform(x_test)

            sScaler=StandardScaler()

            sScaler.fit(x_train)
            x_train=sScaler.transform(x_train)
            x_test=sScaler.transform(x_test)


            
            #taining

            dtc=DecisionTreeClassifier(criterion='entropy',random_state=23)
            dtc.fit(x_train,y_train)
            

    
            if request.method=='POST':
                if 'recommend' in request.POST:
                    n=float(request.POST.get('N'))
                    p=float(request.POST.get('P'))
                    k=float(request.POST.get('K'))
                    temp=float(request.POST.get('temp'))
                    humid=float(request.POST.get('humid'))
                    ph=float(request.POST.get('ph'))
                    rain=float(request.POST.get('rain'))
        
                    N=n
                    P=p
                    K=k
                    temperature=temp
                    humidity=humid
                    pH=ph	
                    rainfall=rain

                    data = np.array([[N,P, K, temperature, humidity, pH, rainfall]])

                    data=mScaler.transform(data)
                    data=sScaler.transform(data)

                    def recommendation(data):
        
                        prediction=dtc.predict(data).reshape(1,-1)

                        return prediction[0]

                    predict=recommendation(data)

                    
                

                    if predict[0] in crop_items:
                        output="{} is the best crop to grow".format(predict[0])
                        return render(request,'results.html',{'output':output})
                    else:
                        output="Sorry,we are unable to recommend a crop based on the given data"
                        return render(request,'results.html',{'output':output})
                elif 'logout' in request.POST:
                    del request.session['user_id']
                    return redirect('login')
                
    

        

    else:
        return redirect('login')    
    return render(request,'index.html')
    


def LoginPage(request):
    
    
    if request.method=='POST':
        name=request.POST.get('uname')
        pass1=request.POST.get('pass')
        
        user=users.objects.filter(uname=name,password=pass1)
        
        if user :
            user_id=user[0].id
            request.session['user_id']=user_id
            
            return redirect('index')
        else:
            messages.error(request,"The User Name or password is incorrect")
            return redirect("login")
    

    return render(request,'login.html')


def RegisterPage(request):
    if request.method=='POST':
        name=request.POST.get('name')
        email=request.POST.get('email')
        pass1=request.POST.get('pass')
        passRE=request.POST.get('passRE')
        if name and email and pass1 and passRE!='':
            if pass1==passRE:
                users(uname=name, email=email, password=pass1).save()
                messages.success(request, "User successfully registered")
                return redirect("login")
            else:
                messages.error(request,"Enter the correct password")
                return redirect("register")
        else:
            messages.error(request,"Please enter all fields")
            return redirect("register")
        
    return render(request,'register.html')

def ResultPage(request):
    if 'user_id' in request.session:
        if request.method=="POST":
            return redirect('index')

    return render(request,'results.html')    


# def DataAnalysis(n,p,k,temp,humid,ph,rain):
    
#     import numpy as np
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import MinMaxScaler  
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.tree import DecisionTreeClassifier

#     crop=pd.read_csv("static/Crop_recommendation.csv")

#     crop_items={
#         'rice'          :1,
#         'maize'         :2,
#         'jute'          :3,
#         'cotton'        :4,
#         'coconut'       :5,
#         'papaya'        :6,
#         'orange'        :7,
#         'apple'         :8,
#         'muskmelon'     :9,
#         'watermelon'    :10,
#         'grapes'        :11,
#         'mango'         :12,
#         'banana'        :13,
#         'pomegranate'   :14,
#         'lentil'        :15,
#         'blackgram'     :16,
#         'mungbean'      :17,
#         'mothbeans'     :18,
#         'pigeonpeas'    :19,
#         'kidneybeans'   :20,
#         'chickpea'      :21,
#         'coffee'        :22
#     }
#     crop['crop_num']=crop['label'].map(crop_items)

#     crop.drop('label',axis=1,inplace=True)

#     x=crop.drop('crop_num',axis=1)
#     y=crop['crop_num']

#     x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#     mScaler=MinMaxScaler()

#     mScaler.fit(x_train,y_train)
#     x_train=mScaler.transform(x_train)
#     x_test=mScaler.transform(x_test)


#     sScaler=StandardScaler()

#     sScaler.fit(x_train,y_train)
#     x_train=sScaler.transform(x_train)
#     x_test=sScaler.transform(x_test)

#     #taining

#     dtc=DecisionTreeClassifier()
#     dtc.fit(x_train,y_train)
#     ypred=dtc.predict(x_test)


#     def recommendation(data):
        
#         prediction=dtc.predict(data).reshape(1,-1)

#         return prediction[0]
    
    
        
#     N=n
#     P=p
#     K=k
#     temperature=temp
#     humidity=humid
#     pH=ph	
#     rainfall=rain

#     data = np.array([[N,P, K, temperature, humidity, pH, rainfall]])



#     data=mScaler.transform(data)


#     data=sScaler.transform(data)





#     predict=recommendation(data)



#     crop_dict={
#         1:'rice',
#         2:'maize',
#         3:'jute',
#         4:'cotton',
#         5:'coconut',
#         6:'papaya',
#         7:'orange',
#         8:'apple',
#         9:'muskmelon',
#         10:'watermelon',
#         11:'grapes',
#         12:'mango',
#         13:'banana',
#         14:'pomegranate',
#         15:'lentil',
#         16:'blackgram',
#         17:'mungbean',
#         18:'mothbeans',
#         19:'pigeonpeas',
#         20:'kidneybeans',
#         21:'chickpea',
#         22:'coffee'        
#     }

#     if predict[0] in crop_dict:
#         crop=crop_dict[predict[0]]
#         success="{} is the best crop to grow".format(crop)
#         return success
#     else:
#         sorry="Sorry,we are unable to recommend a crop based on the given data"
#         return sorry
    



