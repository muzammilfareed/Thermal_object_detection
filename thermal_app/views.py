from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
import os
import cv2
from image_object_detection import main


@csrf_exempt
def index(request):
    global encodeimg, polygon_points, res_img, ori_path
    folder = 'static/input_img/'
    if request.method == "POST":
        main_host = request.get_host()
        file = request.FILES.get('file')

        location = FileSystemStorage(location=folder)
        fn = location.save(file.name, file)
        path = os.path.join(folder, fn)
        img = main(path)
        output_path = f'static/result/{file.name}'
        input_path = f'{folder}{file.name}'
        cv2.imwrite(output_path,img)
        context = {
            "status":True,
            "Image_path":output_path,
            "orignal_path": input_path,
        }
        return JsonResponse(context)
    return render(request, 'index.html')

