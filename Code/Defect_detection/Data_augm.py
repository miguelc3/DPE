from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import glob
import cv2

# Loading dataset
DATADIR = "..\..\imagens\\DATASETS_pg\\Falta_tinta\\OK\\*.png"

# Path to save images after data augmentation
path_save = "..\..\imagens\\DATASETS_pg\\3_FALTA_TINTA\\DATASET_1\\OK"

# creates a data generator object that transforms images
datagen = ImageDataGenerator(
                             rotation_range=5,
                             width_shift_range=1.5,
                             height_shift_range=1.5,
                             shear_range=0.2,
                             # zoom_range=[0.7, 1],
                             zoom_range=0.15,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest'
                             )

a = 0
for image_file in glob.iglob(DATADIR):

    print('image nr: ' + str(image_file))

    # Image to transform and label
    test_img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # convert image to numpy array
    img = image.img_to_array(test_img)

    # reshape image
    img = img.reshape((1,) + img.shape)
    plt.imshow(test_img)
    # plt.show()  # Just for testing

    i = 0
    # Apply data augmentation
    for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):
        # this loops runs forever until we break, saving images to current directory with specified prefix
        plt.figure(i)

        # Set params to save
        plt.axis('off')
        plt.set_cmap('hot')
        plt.set_cmap('gray')
        fig = plt.gcf()
        fig.set_size_inches(5.85, 5.85)

        # Display and save image
        plt.imshow(image.img_to_array(batch[0]))
        file_name = i+1+a

        fig.savefig(path_save + '\\' + str(file_name), bbox_inches='tight', pad_inches=0)

        i += 1
        if i > 9:  # create 10 images
            a += 10
            break


