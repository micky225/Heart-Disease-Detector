from django.db import models
import uuid

class HeartParameter(models.Model):
    SEX = [
        (0,'Male'),
        (1,'Female')
    ]

     #Categorized as above 120 mg/dl
    FASTING_BLOOD_SUGAR_LEVEL = [
        (0,'True'),
        (1,'False')
    ]

    id = models.UUIDField(editable=False, primary_key=True, default=uuid.uuid4, unique=True)
    age = models.IntegerField()
    sex = models.IntegerField(choices=SEX, default=0)
    resting_blood_pressure = models.IntegerField()
    serum_cholesterol = models.IntegerField()
    fasting_blood_sugar_level = models.IntegerField(
        choices=FASTING_BLOOD_SUGAR_LEVEL,
        default=0
          )
    maximum_heart_rate = models.IntegerField()
    prediction_result = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(verbose_name='date_added', null=True, blank=True, auto_now_add=True)

    def __str__(self):
        return f'{self.id} - {self.name}'
    

