import os
import torch

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    IMAGES_DIR = os.path.join(DATA_DIR, 'img_resized')
    IMG_TEXT_DIR = os.path.join(DATA_DIR, 'img_txt')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    CHECKPOINTS_DIR = os.path.join(MODELS_DIR, 'checkpoints')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    TEMP_DIR = os.path.join(BASE_DIR, 'temp')
    
    # Data files
    JSON_FILE = os.path.join(DATA_DIR, 'MMHS150K_GT.json')
    TRAIN_IDS = os.path.join(DATA_DIR, 'train_ids.txt')
    VAL_IDS = os.path.join(DATA_DIR, 'val_ids.txt')
    TEST_IDS = os.path.join(DATA_DIR, 'test_ids.txt')
    
    # Hate speech keywords
    HATE_KEYWORDS = [
        "asian drive", "feminazi", "sjw", "WomenAgainstFeminism", "blameonenotall",
        "islam terrorism", "notallmen", "victimcard", "victim card", "arab terror",
        "gamergate", "jsil", "racecard", "race card", "refugeesnotwelcome",
        "DeportallMuslims", "banislam", "banmuslims", "destroyislam", "norefugees",
        "nomuslims", "border jumper", "border nigger", "boojie", "surrender monkey",
        "chinaman", "hillbilly", "whigger", "white nigger", "wigger", "wigerette",
        "bitter clinger", "conspiracy theorist", "redneck", "rube", "trailer park trash",
        "trailer trash", "white trash", "yobbo", "retard", "retarded", "nigger",
        "coonass", "raghead", "house nigger", "camel fucker", "moon cricket",
        "wetback", "spic", "bint", "dyke", "twat", "bamboo coon", "limey",
        "plastic paddy", "sideways pussy", "zionazi", "muzzie", "soup taker",
        "faggot", "cunt", "nigga"
    ]
    
    HATE_HASHTAGS = [
        "#DontDateSJWs", "#Feminazi", "#FemiNazi", "#BuildTheWall", "#sorryladies",
        "#IWouldFuckYouBut", "#DeportThemALL", "#RefugeesNOTwelcome", "#BanSharia",
        "#BanIslam", "#nosexist"
    ]
    
    # Model hyperparameters
    BERT_MODEL = 'bert-base-uncased'
    CNN_MODEL = 'resnet50'
    TEXT_DIM = 768
    VISION_DIM = 2048
    HIDDEN_DIM = 512
    OUTPUT_DIM = 2
    ATTENTION_HEADS = 8
    MAX_TEXT_LENGTH = 128
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    EPOCHS = 20
    DROPOUT_RATE = 0.3
    
    # Soft label parameters
    NUM_ANNOTATORS = 3
    LABEL_SMOOTHING = 0.1
    
    # Training parameters
    EARLY_STOPPING_PATIENCE = 5
    GRADIENT_CLIP = 1.0
    WEIGHT_DECAY = 0.01
    
    # Fairness thresholds
    DEMOGRAPHIC_PARITY_THRESHOLD = 0.1
    EQUALIZED_ODDS_THRESHOLD = 0.1
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def create_dirs():
        """Create necessary directories"""
        dirs = [Config.DATA_DIR, Config.MODELS_DIR, Config.CHECKPOINTS_DIR, 
                Config.RESULTS_DIR, Config.TEMP_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
