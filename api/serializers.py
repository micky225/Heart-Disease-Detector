from rest_framework import serializers
from .models import HeartParameter

class HeartSerializer(serializers.ModelSerializer):
    class Meta:
        model = HeartParameter
        fields = '__all__'