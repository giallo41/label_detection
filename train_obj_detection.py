from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()

data_dir = "./data/images/box/"
#model_dir = "./data/models/pretrained-yolov3.h5"
#model_dir = "./data/models/pretrained-yolov3.h5"
model_dir = f"./data/images/box/models/detection_model-ex-003--loss-0020.019.h5"

trainer.setDataDirectory(data_directory=data_dir)

trainer.setTrainConfig(object_names_array=['true'], 
                       batch_size=8, 
                       num_experiments=100, 
                       train_from_pretrained_model=model_dir)


trainer.trainModel()

