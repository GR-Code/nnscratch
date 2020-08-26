import gzip
import numpy as np
# import matplotlib.pyplot as plt

def load_mnist_data(trainImage=r"data/train-images-idx3-ubyte.gz", trainLabel=r"data/train-labels-idx1-ubyte.gz", isTest=False):
    with gzip.open(trainImage,'rb') as tImages, gzip.open(trainLabel,'rb') as tLabels:
        magic_number = int.from_bytes(tImages.read(4),'big')    # Don't care about this
        n_images = int.from_bytes(tImages.read(4),'big')        
        rows = int.from_bytes(tImages.read(4),'big')
        cols = int.from_bytes(tImages.read(4),'big')
        # imageData = []
        # for i in range(n_images):        
        #     image = np.frombuffer(tImages.read(rows*cols),dtype=np.uint8).astype(np.float32).reshape(rows*cols,1)
        #     imageData.append(image/255)
        imageData = np.frombuffer(tImages.read(),dtype=np.uint8).astype(np.float32).reshape(n_images,rows*cols,1)/255

        magic_number = int.from_bytes(tLabels.read(4),'big')    # Don't care about this
        n_labels = int.from_bytes(tLabels.read(4),'big')        # Don't care about this
        buf = tLabels.read()
        if isTest:
            labelData = np.frombuffer(buf,dtype=np.uint8).astype(np.float32)
        else:
            labelData = [output_vector_from_label(label) for label in np.frombuffer(buf,dtype=np.uint8).astype(np.int)]

    return zip(imageData, labelData)
        

def output_vector_from_label(x):
    output = np.zeros((10,1))
    output[x] = 1.0
    return output


if __name__ == "__main__": 
    pass
    