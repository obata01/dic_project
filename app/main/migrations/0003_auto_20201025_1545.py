# Generated by Django 3.1.2 on 2020-10-25 15:45

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0002_auto_20201025_0937'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Item',
        ),
        migrations.AlterModelTable(
            name='recommend',
            table='recommend',
        ),
    ]