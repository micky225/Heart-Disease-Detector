# Generated by Django 5.0.7 on 2024-08-15 19:48

import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='HeartParameter',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False, unique=True)),
                ('age', models.IntegerField()),
                ('sex', models.IntegerField(choices=[(0, 'Male'), (1, 'Female')], default=0)),
                ('chest_pain_type', models.IntegerField(choices=[(0, 'Typical angina'), (1, 'Atypical angina'), (2, 'Non-anginal pain'), (3, 'Asymptomatic')], default=0)),
                ('resting_blood_pressure', models.IntegerField()),
                ('serum_cholesterol', models.IntegerField()),
                ('fasting_blood_sugar_level', models.IntegerField(choices=[(0, 'True'), (1, 'False')], default=0)),
                ('resting_electrocardiographoc_results', models.IntegerField(choices=[(0, ' Normal'), (1, 'Having ST-T wave abnormality'), (2, 'Showing probable or definite left ventricular hypertrophy')], default=0)),
                ('maximum_heart_rate', models.IntegerField()),
                ('exercise_induced_agina', models.IntegerField(choices=[(0, 'Yes'), (1, 'No')], default=0)),
                ('st_depression', models.FloatField()),
                ('slope', models.IntegerField(choices=[(0, 'Upsloping'), (1, 'Flat'), (2, 'Downsloping')], default=0)),
                ('number_of_major_vessels', models.IntegerField()),
                ('thallium_stress_test_results', models.IntegerField(choices=[(0, 'Normal'), (1, 'Fixed defect'), (2, 'Reversible defect'), (3, ' Not described')], default=0)),
                ('prediction_result', models.CharField(blank=True, max_length=255, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True, verbose_name='date_added')),
            ],
        ),
    ]
