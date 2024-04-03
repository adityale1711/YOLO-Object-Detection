import yaml
import tkinter as tk

from ultralytics import YOLO
from roboflow import Roboflow
from multiprocessing import freeze_support
from tkinter.filedialog import askopenfilename, askdirectory

root = tk.Tk()
root.withdraw()

def downloadDatasets():
    rf = Roboflow(api_key="")
    project = rf.workspace("bootcamp-o49zr").project("football-analysis-eboqf")
    version = project.version(1)
    dataset = version.download("yolov8")
def main():
    # Load pretrained model for transfer weight
    weight_model = askopenfilename(title="Select pretrained weights model")
    dataset_config = askopenfilename(title="Select dataset config file")

    train_folder = askdirectory(title="Select training folder")
    val_folder = askdirectory(title="Select validation folder")
    test_folder = askdirectory(title="Select test folder")

    with open(dataset_config, 'r') as file:
        data = yaml.safe_load(file)

    data['test'] = test_folder
    data['train'] = train_folder
    data['val'] = val_folder

    with open(dataset_config, 'w') as file:
        yaml.dump(data, file)

    model = YOLO(weight_model)

    # Display model info
    model.info()

    train_results = model.train(data=dataset_config, epochs=100, imgsz=640, patience=25)

# downloadDatasets()
if __name__ == '__main__':
    freeze_support()
    # downloadDatasets()
    main()