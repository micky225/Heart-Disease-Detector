from django.urls import path
from .views import HeartApiView, HeartApiViewDetails

urlpatterns = [
    path('heart-api/', HeartApiView.as_view(), name='heart-api'),
    path('heart-api-details/<str:pk>/', HeartApiViewDetails.as_view(), name='heart-api-details')
]
