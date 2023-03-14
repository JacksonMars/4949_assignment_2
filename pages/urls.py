from django.urls import path
from .views import homePageView, aboutPageView, results, homePost

urlpatterns = [
    path('', homePageView, name='home'),
    path('about/', aboutPageView, name='about'),
    path('homePost/', homePost, name='homePost'),
    path('results/<int:pollution>/<int:alcohol>/<int:allergies>/<int:hazards>/<int:obesity>/<int:passive_smoke>/'
         '<int:chest_pain>/<int:blood_cough>/<int:fatigue>/<int:wheezing>/<int:clubbing>/<int:frequent_cold>/'
         '<int:dry_cough>/<int:snoring>/<int:age>/', results, name='results'),
]
