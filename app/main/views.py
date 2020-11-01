from django.shortcuts import render
from django.shortcuts import redirect, get_object_or_404
from main.models import *
from django.http import HttpResponse
from django.template import loader
from .forms import PhotoForm, SearchwordForm
from .models import Photo
from utils.logging import Logger
logger = Logger(level='DEBUG')



def index(request):
    template = loader.get_template('main/index.html')
    context = {'form':PhotoForm()}
    return HttpResponse(template.render(context, request))


def predict(request):
    """画像アップロード後の処理"""
    # POSTか判定
    logger.debug('views.py - predict start...')
    if not request.method == 'POST':
        return redirect('main:index')
    
    # フォームの値が正常か判定
    form = PhotoForm(request.POST, request.FILES)
    if not form.is_valid():
        raise ValueError('Formが不正です')
    logger.debug('views.py - form. {}'.format(form))
    
    # アップロード画像を取得してpredict処理へまわす
    photo = Photo(image=form.cleaned_data['image'])
    predicted, percentage, info, recommend_items = photo.predict()
    percentage = str(int(percentage * 100)) + '％'
    
    # レスポンス準備
    template = loader.get_template('main/result.html')
    context = {
        'photo_data': photo.image_src(),
        'predicted': predicted,
        'percentage': percentage,
        'name': info[1],
        'name_jp': info[2],
        'url': info[3],
        'price': info[4],
        'made_in': info[5],
        'type1': info[6],
        'type2': info[7],
        'hinisyu': info[8],
        'desc0': info[9],
        'desc1': info[10],
        'recommend': recommend_items[:6],
    }
    logger.debug(info)
    return HttpResponse(template.render(context, request))


def predict2(request):
    if not request.method == 'GET':
        logger.warning('method is not post')
        return redirect('main:index')
    kword = request.GET.get('kw', None)
    logger.info(kword)
    
    # predict
    k = KeyWord()
    recommend_items = k.Search(kword)
    
    # レスポンス準備
    template = loader.get_template('main/result2.html')
    context = {
        'recommend': recommend_items[:6],
    }
    logger.debug(recommend_items)
    return HttpResponse(template.render(context, request))


#     """画像アップロード後の処理"""
#     # POSTか判定
#     logger.debug('views.py - predict start...')
#     if not request.method == 'POST':
#         return redirect('main:index')
    

    
#     # レスポンス準備
#     template = loader.get_template('main/result.html')
#     context = {
#         'photo_data': photo.image_src(),
#         'predicted': predicted,
#         'percentage': percentage,
#         'name': info[1],
#         'name_jp': info[2],
#         'url': info[3],
#         'price': info[4],
#         'made_in': info[5],
#         'type1': info[6],
#         'type2': info[7],
#         'hinisyu': info[8],
#         'desc0': info[9],
#         'desc1': info[10],
#         'recommend': recommend_items[:6],
#     }
#     logger.debug(info)
#     return HttpResponse(template.render(context, request))
