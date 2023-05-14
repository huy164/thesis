from django.urls import path
from myapp.views import get_vn30_history

urlpatterns = [
    path('get-vn30-history/', get_vn30_history, name='get_vn30_history'),
]
