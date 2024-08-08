# Generated by Django 5.0.7 on 2024-08-08 02:09

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0003_heartparameter_prediction_result'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='heartparameter',
            name='chest_pain_type',
        ),
        migrations.RemoveField(
            model_name='heartparameter',
            name='exercise_induced_agina',
        ),
        migrations.RemoveField(
            model_name='heartparameter',
            name='number_of_major_vessels',
        ),
        migrations.RemoveField(
            model_name='heartparameter',
            name='resting_electrocardiographoc_results',
        ),
        migrations.RemoveField(
            model_name='heartparameter',
            name='slope',
        ),
        migrations.RemoveField(
            model_name='heartparameter',
            name='st_depression',
        ),
        migrations.RemoveField(
            model_name='heartparameter',
            name='thallium_stress_test_results',
        ),
    ]
