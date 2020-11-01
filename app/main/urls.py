from django.urls import path
from . import views
# from main.views import WineSearchView

app_name = 'main'
urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('predict2/', views.predict2, name='predict2'),
]