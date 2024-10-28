from django.db import models

# Create your models here.

class users(models.Model):
    uname=models.CharField(max_length=100)
    email=models.CharField(max_length=100)
    password=models.CharField(max_length=100)