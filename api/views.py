from django.shortcuts import render
from .models import HeartParameter
from .serializers import HeartSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from django.contrib import messages
from rest_framework import status
import numpy as np
import pickle

with open('svc.pkl', 'rb') as file:
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
                data['serum_colesterol'], data['fasting_blood_sugar_level'], data['resting_electrocardiographoc_results'],
                data['maximum_heart_rate'], data['exercise_induced_agina'], data['st_depression'],
                data['slope'], data['number_of_major_vessels'], data['thallium_stress_test_results']
            ]])

            prediction = classifier.predict(features)
            prediction = prediction[0].item()

            result = 'Your heart is fine, you do not have heart disease' if prediction == 0 else 'Unfortunately, you have heart disease'
            serializer.save(prediction_result=result)

            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


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