from django.shortcuts import render, HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView
from django.http import HttpResponseRedirect
from django.urls import reverse
import pickle
import pandas as pd


def homePageView(request):
    return render(request, 'home.html', {
        'mynumbers':[1,2,3,4,5,6,7,8,9,10],
        'firstName': 'Pat'})


def homePost(request):
    try:
        age = request.POST['age']

        pollution = request.POST['pollution']
        alcohol = request.POST['alcohol']
        allergies = request.POST['allergies']
        hazards = request.POST['hazards']
        obesity = request.POST['obesity']
        passive_smoke = request.POST['passive_smoke']
        chest_pain = request.POST['chest_pain']
        blood_cough = request.POST['blood_cough']
        fatigue = request.POST['fatigue']
        wheezing = request.POST['wheezing']
        clubbing = request.POST['clubbing']
        frequent_cold = request.POST['frequent_cold']
        dry_cough = request.POST['dry_cough']
        snoring = request.POST['snoring']

    except:
        return render(request, 'home.html', {'errorMessage': '*** The data submitted is invalid. Please try again.'})
    else:
        return HttpResponseRedirect(reverse('results',
                                            kwargs={'pollution': pollution, 'alcohol': alcohol, 'allergies': allergies,
                                                    'hazards': hazards, 'obesity': obesity,
                                                    'passive_smoke': passive_smoke, 'chest_pain': chest_pain,
                                                    'blood_cough': blood_cough, 'fatigue': fatigue,
                                                    'wheezing': wheezing, 'clubbing': clubbing,
                                                    'frequent_cold': frequent_cold, 'dry_cough': dry_cough,
                                                    'snoring': snoring, 'age': age}))





def results(request, pollution, alcohol, allergies, hazards, obesity, passive_smoke, chest_pain, blood_cough, fatigue,
            wheezing, clubbing, frequent_cold, dry_cough, snoring, age):
    with open(r'C:\Users\jmars\PycharmProjects\4949_assignment2\model_pkl' , 'rb') as f:
        loadedModel = pickle.load(f)

    singleSampleDf = pd.DataFrame(
        columns=['Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards', 'Obesity', 'Passive Smoker',
                 'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Wheezing', 'Clubbing of Finger Nails', 'Frequent Cold',
                 'Dry Cough', 'Snoring', 'Age'])

    singleSampleDf = singleSampleDf.append({'Air Pollution': int(pollution), 'Alcohol use': int(alcohol), 'Dust Allergy': int(allergies),
                                                    'OccuPational Hazards': int(hazards), 'Obesity': int(obesity),
                                                    'Passive Smoker': int(passive_smoke), 'Chest Pain': int(chest_pain),
                                                    'Coughing of Blood': int(blood_cough), 'Fatigue': int(fatigue),
                                                    'Wheezing': int(wheezing), 'Clubbing of Finger Nails': int(clubbing),
                                                    'Frequent Cold': int(frequent_cold), 'Dry Cough': int(dry_cough),
                                                    'Snoring': int(snoring), 'Age': int(age)},
                                        ignore_index=True)

    singlePrediction = loadedModel.predict(singleSampleDf)

    return render(request, 'results.html', {'pollution': pollution, 'alcohol': alcohol, 'allergies': allergies,
                                                    'hazards': hazards, 'obesity': obesity,
                                                    'passive_smoke': passive_smoke, 'chest_pain': chest_pain,
                                                    'blood_cough': blood_cough, 'fatigue': fatigue,
                                                    'wheezing': wheezing, 'clubbing': clubbing,
                                                    'frequent_cold': frequent_cold, 'dry_cough': dry_cough,
                                                    'snoring': snoring, 'age': age, 'prediction': singlePrediction})
