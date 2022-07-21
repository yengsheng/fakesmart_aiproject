import os

MODEL_CHECKPOINT_FILE = os.path.join('.', 'models', 'model_bs1024_100.pt')
NOISE_FOLDER = os.path.join('.', 'static', 'filter')
GENERATED_FOLDER = os.path.join('.', 'static', 'generated')

# attack options
attOptions = {
    'Untargetted - Epsilon 0.001': 0.001,
    'Untargetted - Epsilon 0.01': 0.01,
    'Untargetted - Epsilon 0.1': 0.1,
    'Untargetted - Epsilon 1.0': 1,
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
    'Targetted - End no passing veh > 3.5 tons': 42
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