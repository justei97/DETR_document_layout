# *****DETR: End-to-End Object Detection with Transformers*****

![Alt Text](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/05/Screenshot-from-2020-05-27-20-04-48.png)

Hello :)
This is the Repository for the Detection Transformer
To run the network you just need to execute [run_Detr.py](https://gropius.medien.uni-weimar.de/applied-deep-learning-for-computer-vision/detection-transformer/-/blob/main/run_Detr.py)
However, for a different dataset you need to change Paths in the [detr.py](https://gropius.medien.uni-weimar.de/applied-deep-learning-for-computer-vision/detection-transformer/-/blob/main/detr.py) file (**line 18-20**) ![Text](https://gropius.medien.uni-weimar.de/applied-deep-learning-for-computer-vision/detection-transformer/-/raw/main/change_path.JPG)  
for the dataloader and additionally the dataloader itself (depending on the desired dataset ofc.) Please note that the network will run on **GPU 0** as specified in line 15 ![Text](https://gropius.medien.uni-weimar.de/applied-deep-learning-for-computer-vision/detection-transformer/-/raw/main/GPU0.JPG). If that is not available, you need to change it!

Please find the commented version of the Code in the corresponding ipynb file (DETR.ipynb)

Hyperparameter Tuning can be also done with [run_Detr.py](https://gropius.medien.uni-weimar.de/applied-deep-learning-for-computer-vision/detection-transformer/-/blob/main/run_Detr.py). You only need to adjust the arguments and do multiple runs :) 




#### The trained model will be saved with the specified name in args in [run_Detr.py](https://gropius.medien.uni-weimar.de/applied-deep-learning-for-computer-vision/detection-transformer/-/blob/main/run_Detr.py).



#### For visualisation purposes:
Tensorboard logs including Loss and ap50 /ap75, as well as IOU per class will be automatically created and stored in runs/modelname



#### And for more explanations:
Please have a look into [DETR.ipynb](https://gropius.medien.uni-weimar.de/applied-deep-learning-for-computer-vision/detection-transformer/-/blob/main/DETR.ipynb)

####For inference purposes, the trained Model can be found at:
[DETR_Trained](https://www.dropbox.com/s/w07fatyge00q099/DETR_trained.pth?dl=0)



```python

```
