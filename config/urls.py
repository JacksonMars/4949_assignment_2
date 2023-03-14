from django.contrib import admin
from django.urls import path, include # new
from django.conf.urls import include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('pages.urls')), # new
]
