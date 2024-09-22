from django.shortcuts import render
from .models import HeartParameter
from .serializers import HeartSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import numpy as np
import pickle
import pandas as pd


with open('svm_13_features.pkl', 'rb') as file:
    classifier = pickle.load(file)


class HeartApiView(APIView):
    def get(self, request, format=None):
        heart_parameter = HeartParameter.objects.all()
        serializer = HeartSerializer(heart_parameter, many=True)
        return Response(serializer.data)
    

    def post(self, request, format=None):
        serializer = HeartSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            features = np.array([[
                data['age'], data['sex'], data['chest_pain_type'], data['resting_blood_pressure'],
                data['serum_cholesterol'], data['fasting_blood_sugar_level'], data['resting_electrocardiographoc_results'],
                data['maximum_heart_rate'], data['exercise_induced_agina'], data['st_depression'],
                data['slope'], data['number_of_major_vessels'], data['thallium_stress_test_results']
            ]])

            column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
            features_df = pd.DataFrame(features, columns=column_names)
            
            prediction = classifier.predict(features_df)
            prediction = prediction[0].item()

            if prediction == 0:
                result = 'Your heart is fine, you do not have heart disease'
            else:
                result = 'Unfortunately, you have heart disease'
                causes = self.analyze_causes(data)
                result += '\n\nPossible contributing factors:\n ' + ', '.join(causes)

            serializer.save(prediction_result=result)
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def analyze_causes(self, data):
        causes = []
        if data['resting_blood_pressure'] >= 130:
            causes.append('High resting blood pressure')
        if data['serum_cholesterol'] > 370:
            causes.append('High serum cholesterol')
        if data['serum_cholesterol'] < 126:
            causes.append('Low serum cholesterol')     
        if data['fasting_blood_sugar_level'] == 1:
            causes.append('High fasting blood sugar level')
        if data['maximum_heart_rate'] < 120:
            causes.append('Minimum heart rate')    
        return causes


class HeartApiViewDetails(APIView):
    def get(self, request, pk, format=None):
        heart_parameter = HeartParameter.objects.get(pk=pk)
        serializer = HeartSerializer(heart_parameter, many=False)
        return Response(serializer.data)
    
    def delete(self, request, pk, format=None):
        heart_parameter = HeartParameter.objects.get(pk=pk)
        heart_parameter.delete()
        return Response(
             {
                "message": "Deleted successfully!"
             },
            status=status.HTTP_204_NO_CONTENT
            )
 