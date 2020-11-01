from django import forms

class PhotoForm(forms.Form):
    image = forms.ImageField(widget=forms.FileInput(attrs={'class':'custom-file-input'}))
    
    
class SearchwordForm(forms.Form):
     kword = forms.CharField(max_length=30)