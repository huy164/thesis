from django.urls import path
from .views import predict_vn30

urlpatterns = [
    path('predict_vn30/', predict_vn30, name='predict_vn30'),
]

