from django.urls import path
from myapp.views import get_vn30_history
from myapp.views import predict_view

urlpatterns = [
    path('get-vn30-history/', get_vn30_history, name='get_vn30_history'),
    path('api/predictions/', predict_view, name='predictions'),

]
