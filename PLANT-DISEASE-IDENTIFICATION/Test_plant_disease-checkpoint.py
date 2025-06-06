from sqlalchemy import null


{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c2174fe8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf \n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "78f433dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "validation_set = tf.keras.utils.image_dataset_from_directory(\n",
    "    'Dataset1/valid',\n",
    "    labels=\"inferred\",\n",
    "    label_mode=\"categorical\",\n",
    "    class_names=None,\n",
    "    color_mode=\"rgb\",\n",
    "    batch_size=32,\n",
    "    image_size=(128, 128),\n",
    "    shuffle=True,\n",
    "    seed=None,\n",
    "    validation_split=None,\n",
    "    subset=None,\n",
    "    interpolation=\"bilinear\",\n",
    "    follow_links=False,\n",
    "    crop_to_aspect_ratio=False\n",
    ")\n",
    "class_name = validation_set.class_names\n",
    "print(class_name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c21edaab",
   "metadata": {},
   "outputs": [],
   "source": [
    "cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "48428614",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Test Image Visualization\n",
    "import cv2\n",
    "image_path = 'Dataset1/test/test/PotatoEarlyBlight5.JPG'\n",
    "# Reading an image in default mode\n",
    "img = cv2.imread(image_path)\n",
    "img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Converting BGR to RGB\n",
    "# Displaying the image \n",
    "plt.imshow(img)\n",
    "plt.title('Test Image')\n",
    "plt.xticks([])\n",
    "plt.yticks([])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "19ede945",
   "metadata": {},
   "outputs": [],
   "source": [
    "image = tf.keras.preprocessing.image.load_img(image_path,target_size=(128,128))\n",
    "input_arr = tf.keras.preprocessing.image.img_to_array(image)\n",
    "input_arr = np.array([input_arr])  # Convert single image to a batch.\n",
    "predictions = cnn.predict(input_arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "789ad12d",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c11b816b",
   "metadata": {},
   "outputs": [],
   "source": [
    "result_index = np.argmax(predictions) #Return index of max element\n",
    "print(result_index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c5bb22fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Displaying the disease prediction\n",
    "model_prediction = class_name[result_index]\n",
    "plt.imshow(img)\n",
    "plt.title(f\"Disease Name: {model_prediction}\")\n",
    "plt.xticks([])\n",
    "plt.yticks([])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2b5e07ec",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
