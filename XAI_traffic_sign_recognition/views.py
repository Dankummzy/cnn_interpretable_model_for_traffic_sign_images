from django.shortcuts import render, redirect
from .forms import ImageUploadForm
import numpy as np
import tensorflow as tf
from .utils import get_img_array, make_gradcam_heatmap, save_and_display_gradcam
from .models import Prediction, XAIImage
import os
import tempfile
from django.conf import settings


def calculate_xai_values(heatmap):
    # Calculate or retrieve explanation method, features used, and explanation based on the XAI process used
    explanation_method = "Grad-CAM"
    features_used = "conv1, conv2"
    explanation = "The model focused on the edges and colors of the sign to make the prediction."
    return explanation_method, features_used, explanation

def index(request):
    xai_image = None  # Initialize xai_image here
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image to a temporary file
            uploaded_image = request.FILES['image']
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_image.read())
                temp_img_path = temp_file.name

            # Now load the image from the temporary file
            img_array = get_img_array(temp_img_path, size=(30, 30))
            model = tf.keras.models.load_model('C:/Users/Dell/Desktop/Software/matthew/TrafficSignRecognition/model/traffic_classifier.h5')
            last_conv_layer_name = 'conv2d_3'  # Update with the correct name of the last convolutional layer
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            # Save prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            prediction = Prediction.objects.create(image=uploaded_image, predicted_class=predicted_class, confidence=confidence)
            # Save XAI image
            xai_image_path = os.path.join(settings.MEDIA_ROOT, 'xai_images', f'{prediction.id}.jpg')
            save_and_display_gradcam(temp_img_path, heatmap, cam_path=xai_image_path)
            # Calculate or retrieve XAI values
            explanation_method, features_used, explanation = calculate_xai_values(heatmap)
            xai_image = XAIImage.objects.create(prediction=prediction, image=xai_image_path,
                                                 explanation_method=explanation_method,
                                                 features_used=features_used,
                                                 explanation=explanation)
            
            # Remove the temporary file
            os.remove(temp_img_path)            
            # Redirect to results page
            return redirect('results', prediction_id=prediction.id)
    else:
        form = ImageUploadForm()
    return render(request, 'XAI_traffic_sign_recognition/index.html', {'form': form, 'xai_image_url': xai_image.image.url if xai_image else None})


def results(request, prediction_id):
    prediction = Prediction.objects.get(id=prediction_id)
    xai_image = XAIImage.objects.get(prediction=prediction)
    return render(request, 'XAI_traffic_sign_recognition/results.html', {'prediction': prediction, 'xai_image': xai_image})
