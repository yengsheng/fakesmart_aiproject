import os

MODEL = 'model1024.pt'
MODEL_CHECKPOINT_FILE = os.path.join('.', 'models', MODEL)
NOISE_FOLDER = os.path.join('.', 'static', 'filter')
GENERATED_FOLDER = os.path.join('.', 'static', 'generated')

# attack options
attOptions = {
    'Untargetted': 0,
    'Targetted - Speed limit (20km/h)': 0, 
    'Targetted - Speed limit (30km/h)': 1, 
    'Targetted - Speed limit (50km/h)': 2, 
    'Targetted - Speed limit (60km/h)': 3, 
    'Targetted - Speed limit (70km/h)': 4, 
    'Targetted - Speed limit (80km/h)': 5, 
    'Targetted - End of speed limit (80km/h)': 6, 
    'Targetted - Speed limit (100km/h)': 7, 
    'Targetted - Speed limit (120km/h)': 8, 
    'Targetted - No passing': 9, 
    'Targetted - No passing veh over 3.5 tons': 10, 
    'Targetted - Right-of-way at intersection': 11, 
    'Targetted - Priority road': 12, 
    'Targetted - Yield': 13, 
    'Targetted - Stop': 14, 
    'Targetted - No vehicles': 15, 
    'Targetted - Veh > 3.5 tons prohibited': 16, 
    'Targetted - No entry': 17, 
    'Targetted - General caution': 18, 
    'Targetted - Dangerous curve left': 19, 
    'Targetted - Dangerous curve right': 20, 
    'Targetted - Double curve': 21, 
    'Targetted - Bumpy road': 22, 
    'Targetted - Slippery road': 23, 
    'Targetted - Road narrows on the right': 24, 
    'Targetted - Road work': 25, 
    'Targetted - Traffic signals': 26, 
    'Targetted - Pedestrians': 27, 
    'Targetted - Children crossing': 28, 
    'Targetted - Bicycles crossing': 29, 
    'Targetted - Beware of ice/snow': 30, 
    'Targetted - Wild animals crossing': 31, 
    'Targetted - End speed + passing limits': 32, 
    'Targetted - Turn right ahead': 33, 
    'Targetted - Turn left ahead': 34, 
    'Targetted - Ahead only': 35, 
    'Targetted - Go straight or right': 36, 
    'Targetted - Go straight or left': 37, 
    'Targetted - Keep right': 38, 
    'Targetted - Keep left': 39, 
    'Targetted - Roundabout mandatory': 40, 
    'Targetted - End of no passing': 41, 
    'Targetted - End no passing veh > 3.5 tons': 42,
    'Carlini Wagner Attack - Speed limit (20km/h)': 0, 
    'Carlini Wagner Attack - Speed limit (30km/h)': 1, 
    'Carlini Wagner Attack - Speed limit (50km/h)': 2, 
    'Carlini Wagner Attack - Speed limit (60km/h)': 3, 
    'Carlini Wagner Attack - Speed limit (70km/h)': 4, 
    'Carlini Wagner Attack - Speed limit (80km/h)': 5, 
    'Carlini Wagner Attack - End of speed limit (80km/h)': 6, 
    'Carlini Wagner Attack - Speed limit (100km/h)': 7, 
    'Carlini Wagner Attack - Speed limit (120km/h)': 8, 
    'Carlini Wagner Attack - No passing': 9, 
    'Carlini Wagner Attack - No passing veh over 3.5 tons': 10, 
    'Carlini Wagner Attack - Right-of-way at intersection': 11, 
    'Carlini Wagner Attack - Priority road': 12, 
    'Carlini Wagner Attack - Yield': 13, 
    'Carlini Wagner Attack - Stop': 14, 
    'Carlini Wagner Attack - No vehicles': 15, 
    'Carlini Wagner Attack - Veh > 3.5 tons prohibited': 16, 
    'Carlini Wagner Attack - No entry': 17, 
    'Carlini Wagner Attack - General caution': 18, 
    'Carlini Wagner Attack - Dangerous curve left': 19, 
    'Carlini Wagner Attack - Dangerous curve right': 20, 
    'Carlini Wagner Attack - Double curve': 21, 
    'Carlini Wagner Attack - Bumpy road': 22, 
    'Carlini Wagner Attack - Slippery road': 23, 
    'Carlini Wagner Attack - Road narrows on the right': 24, 
    'Carlini Wagner Attack - Road work': 25, 
    'Carlini Wagner Attack - Traffic signals': 26, 
    'Carlini Wagner Attack - Pedestrians': 27, 
    'Carlini Wagner Attack - Children crossing': 28, 
    'Carlini Wagner Attack - Bicycles crossing': 29, 
    'Carlini Wagner Attack - Beware of ice/snow': 30, 
    'Carlini Wagner Attack - Wild animals crossing': 31, 
    'Carlini Wagner Attack - End speed + passing limits': 32, 
    'Carlini Wagner Attack - Turn right ahead': 33, 
    'Carlini Wagner Attack - Turn left ahead': 34, 
    'Carlini Wagner Attack - Ahead only': 35, 
    'Carlini Wagner Attack - Go straight or right': 36, 
    'Carlini Wagner Attack - Go straight or left': 37, 
    'Carlini Wagner Attack - Keep right': 38, 
    'Carlini Wagner Attack - Keep left': 39, 
    'Carlini Wagner Attack - Roundabout mandatory': 40, 
    'Carlini Wagner Attack - End of no passing': 41, 
    'Carlini Wagner Attack - End no passing veh > 3.5 tons': 42
}

# episolon option
epOption = {
    '0.001': 0.001,
    '0.005': 0.005,
    '0.01': 0.01,
    '0.05': 0.05,
    '0.1': 0.1,
    '0.5': 0.5,
    '1.0': 1,
    '5.0': 5,
}
    

# Label Overview
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }