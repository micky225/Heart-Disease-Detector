from django.db import models
import uuid

class HeartParameter(models.Model):
    SEX = [
        (0,'Male'),
        (1,'Female')
    ]

    CHEST_PAIN_TYPE = [
        (0,'Typical angina'),
        (1,'Atypical angina'),
        (2,'Non-anginal pain'),
        (3, 'Asymptomatic')
    ]

    #Categorized as above 120 mg/dl
    FASTING_BLOOD_SUGAR_LEVEL = [
        (0,'True'),
        (1,'False')
    ]

    RESTING_ELECTROCARDIOGRAPHOC_RESULTS = [
        (0,' Normal'),
        (1,'Having ST-T wave abnormality'),
        (2,'Showing probable or definite left ventricular hypertrophy')
    ]

    EXERCISE_INDUCED_AGINA = [
        (0,'Yes'),
        (1,'No')
    ]

    SLOPE = [
        (0,'Upsloping'),
        (1,'Flat'),
        (2,'Downsloping')
    ]

    THALLIUM_STRESS_TEST_RESULTS = [
        (0,'Normal'),
        (1,'Fixed defect'),
        (2,'Reversible defect'),
        (3,' Not described')
    ]

    id = models.UUIDField(editable=False, primary_key=True, default=uuid.uuid4, unique=True)
    name = models.CharField(max_length=100, null=True, blank=True)
    age = models.IntegerField()
    sex = models.IntegerField(choices=SEX, default=0)
    chest_pain_type = models.IntegerField(choices=CHEST_PAIN_TYPE, default=0)
    resting_blood_pressure = models.IntegerField()
    serum_colesterol = models.IntegerField()
    fasting_blood_sugar_level = models.IntegerField(
        choices=FASTING_BLOOD_SUGAR_LEVEL,
        default=0
          )
    resting_electrocardiographoc_results = models.IntegerField(
        choices=RESTING_ELECTROCARDIOGRAPHOC_RESULTS, 
        default=0
        )
    maximum_heart_rate = models.IntegerField()
    exercise_induced_agina = models.IntegerField(
        choices=EXERCISE_INDUCED_AGINA,
        default=0
        )
    st_depression = models.FloatField()
    slope = models.IntegerField(choices=SLOPE, default=0)
    number_of_major_vessels = models.IntegerField()
    thallium_stress_test_results = models.IntegerField(
        choices=THALLIUM_STRESS_TEST_RESULTS,
        default=0
        )
    created_at = models.DateTimeField(verbose_name='date_added', null=True, blank=True, auto_now_add=True)

    def __str__(self):
        return f'{self.id} - {self.name}'
    

