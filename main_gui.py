"""
Multimodal Hate Speech Detection System - GUI Application
Detects and reduces bias in hate speech detection using BERT + ResNet + Soft Labels + Fairness Checks

Usage:
    .venv/bin/python main_gui.py
"""

import sys
import os
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from datetime import datetime
import io
import re
import cv2

# Try to import transformers and pytesseract for OCR
try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  Transformers not available. Using keyword-based text analysis.")

try:
    import pytesseract
    import platform
    import shutil

    # Auto-detect tesseract path based on OS
    _sys = platform.system()
    if _sys == 'Windows':
        pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    elif _sys == 'Darwin':  # macOS
        _tess = shutil.which('tesseract') or '/opt/homebrew/bin/tesseract'
        pytesseract.pytesseract.pytesseract_cmd = _tess
    else:  # Linux
        pytesseract.pytesseract.pytesseract_cmd = shutil.which('tesseract') or 'tesseract'

    # Verify tesseract is callable
    pytesseract.get_tesseract_version()
    HAS_OCR = True
except (ImportError, Exception):
    HAS_OCR = False
    print("⚠️  Pytesseract/Tesseract not available. Image text extraction disabled.")
    print("   macOS: brew install tesseract | Linux: sudo apt install tesseract-ocr")

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CKPT_DIR   = os.path.join(MODELS_DIR, 'checkpoints')

os.makedirs(RESULTS_DIR, exist_ok=True)

# Comprehensive hate speech keywords
HATE_KEYWORDS = {
    'gender': ['feminazi', 'sjw', 'notallmen', 'bitter clinger', 'dyke', 'faggot'],
    'race': ['chinaman', 'raghead', 'wetback', 'spic', 'coonass', 'limey'],
    'religion': ['islam terrorism', 'arab terror', 'muzzie', 'camel fucker', 'zionazi'],
    'immigration': ['border jumper', 'norefugees', 'DeportallMuslims', 'refugeesnotwelcome'],
    'general': ['retard', 'cunt', 'nigga', 'whigger', 'trailer trash', 'redneck']
}

HATE_HASHTAGS = [
    "#DontDateSJWs", "#Feminazi", "#FemiNazi", "#BuildTheWall", "#sorryladies",
    "#IWouldFuckYouBut", "#DeportThemALL", "#RefugeesNOTwelcome", "#BanSharia",
    "#BanIslam", "#nosexist"
]

# ============================================================================
# IMAGE BIAS DETECTOR
# ============================================================================

class ImageBiasDetector:
    """Detect bias and hate speech indicators in images"""
    def __init__(self):
        self.hate_keywords = self._load_hate_keywords()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _load_hate_keywords(self):
        """Load hate keywords from configuration"""
        keywords = []
        for category, kws in HATE_KEYWORDS.items():
            keywords.extend(kws)
        keywords.extend(HATE_HASHTAGS)
        return [kw.lower() for kw in keywords]
    
    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR (optional)"""
        if not HAS_OCR:
            return ""
        
        try:
            import pytesseract
            from PIL import Image as PILImage
            
            image = PILImage.open(image_path)
            text = pytesseract.image_to_string(image)
            return text if text.strip() else ""
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return ""
    
    def detect_text_regions(self, image_path):
        """Detect text regions using edge detection (fallback without OCR)"""
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 100, 200)
            
            # Contours might indicate text regions
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count text-like regions
            text_regions = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 100000:  # Typical text size
                    text_regions += 1
            
            return text_regions > 5  # Has significant text regions
        except Exception as e:
            print(f"Text region detection error: {e}")
            return False
    
    def analyze_image_colors(self, image_path):
        """Analyze image colors for inflammatory indicators"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'red_ratio': 0, 'black_ratio': 0, 'contrast': 0, 'is_high_contrast': False}
            
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Red tones (often used in provocative content)
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            red_mask = cv2.inRange(hsv, lower_red, upper_red)
            red_ratio = np.count_nonzero(red_mask) / red_mask.size if red_mask.size > 0 else 0
            
            # Black tones (often used in dark/aggressive content)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 50])
            black_mask = cv2.inRange(hsv, lower_black, upper_black)
            black_ratio = np.count_nonzero(black_mask) / black_mask.size if black_mask.size > 0 else 0
            
            # High contrast (often indicates memes)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contrast = float(gray.std()) / (float(gray.mean()) + 1e-6) if gray.mean() > 0 else 0
            
            return {
                'red_ratio': float(red_ratio),
                'black_ratio': float(black_ratio),
                'contrast': float(contrast),
                'is_high_contrast': contrast > 0.3
            }
        except Exception as e:
            print(f"Color analysis error: {e}")
            return {'red_ratio': 0, 'black_ratio': 0, 'contrast': 0, 'is_high_contrast': False}
    
    def detect_faces(self, image_path):
        """Detect faces in image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'face_count': 0, 'has_faces': False, 'face_percentage': 0}
            
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            return {
                'face_count': int(len(faces)),
                'has_faces': len(faces) > 0,
                'face_percentage': float((len(faces) * 100) / max(len(faces), 1))
            }
        except Exception as e:
            print(f"Face detection error: {e}")
            return {'face_count': 0, 'has_faces': False, 'face_percentage': 0}
    
    def analyze_image(self, image_path):
        """Complete image bias analysis"""
        if not image_path:
            return {
                'has_text': False, 'extracted_text': '', 'text_keywords_found': [],
                'colors': {}, 'faces': {'has_faces': False}, 'has_text_regions': False,
                'image_bias_score': 0.0
            }
        features = {
            'has_text': False,
            'extracted_text': '',
            'text_keywords_found': [],
            'colors': self.analyze_image_colors(image_path),
            'faces': self.detect_faces(image_path),
            'has_text_regions': self.detect_text_regions(image_path),
            'image_bias_score': 0.0
        }
        
        # Try to extract text from image using OCR
        extracted_text = self.extract_text_from_image(image_path)
        if extracted_text.strip():
            features['has_text'] = True
            features['extracted_text'] = extracted_text
            
            # Check for hate keywords in extracted text
            text_lower = extracted_text.lower()
            for keyword in self.hate_keywords:
                if keyword in text_lower:
                    features['text_keywords_found'].append(keyword)
        
        # Calculate image bias score
        bias_score = 0
        
        # High contrast + red/black = typical meme format (slight bias indicator)
        if features['colors']['is_high_contrast']:
            bias_score += 0.15
        
        if features['colors']['red_ratio'] > 0.2:
            bias_score += 0.1
        
        if features['colors']['black_ratio'] > 0.3:
            bias_score += 0.1
        
        # Text keywords found in image
        if features['text_keywords_found']:
            bias_score += min(len(features['text_keywords_found']) * 0.15, 0.5)
        
        # Text regions detected (even without OCR)
        if features['has_text_regions']:
            bias_score += 0.1
        
        # Faces detected in provocative context
        if features['faces']['has_faces'] and features['colors']['is_high_contrast']:
            bias_score += 0.1
        
        features['image_bias_score'] = min(bias_score, 0.99)
        
        return features

# ============================================================================
# DATASET LOADER
# ============================================================================

class DatasetLoader:
    """Load and integrate dataset files"""
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.annotations = {}
        self.train_ids = []
        self.val_ids = []
        self.test_ids = []
        self.load_dataset()
    
    def load_dataset(self):
        """Load all dataset files"""
        # Load JSON annotations
        json_file = os.path.join(self.data_dir, 'MMHS150K_GT.json')
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    self.annotations = json.load(f)
                print(f"✓ Loaded {len(self.annotations)} annotations")
            except Exception as e:
                print(f"Error loading JSON: {e}")
        
        # Load split IDs
        for split_file, split_list in [
            ('train_ids.txt', self.train_ids),
            ('val_ids.txt', self.val_ids),
            ('test_ids.txt', self.test_ids)
        ]:
            file_path = os.path.join(self.data_dir, split_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        split_list.extend([line.strip() for line in f.readlines()])
                    print(f"✓ Loaded {len(split_list)} IDs from {split_file}")
                except Exception as e:
                    print(f"Error loading {split_file}: {e}")
    
    def get_annotation(self, sample_id):
        """Get annotation for a sample"""
        if isinstance(self.annotations, list):
            for ann in self.annotations:
                if ann.get('id') == sample_id:
                    return ann
        else:
            return self.annotations.get(sample_id)
        return None
    
    def get_ground_truth_label(self, sample_id):
        """Get ground truth label for training/evaluation"""
        ann = self.get_annotation(sample_id)
        if ann:
            # Multi-annotator consensus
            labels = ann.get('labels', [])
            if labels:
                return np.mean(labels) > 0.5
        return False

# ============================================================================
# TEXT ENCODER - BERT-based or Fallback
# ============================================================================

class TextEncoder(nn.Module):
    """Text encoder using BERT or CNN fallback"""
    def __init__(self):
        super().__init__()
        self.use_bert = HAS_TRANSFORMERS
        
        if self.use_bert:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                self.model = AutoModel.from_pretrained('bert-base-uncased')
                self.output_dim = 768
            except:
                self.use_bert = False
                self._init_fallback()
        else:
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize fallback CNN-based text encoder"""
        self.embedding = nn.Embedding(5000, 128, padding_idx=0)
        self.conv1 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.output_dim = 512
    
    def forward(self, text, device='cpu'):
        """Extract text features"""
        if self.use_bert:
            try:
                inputs = self.tokenizer(text, return_tensors='pt', 
                                      padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                return outputs.last_hidden_state[:, 0, :]  # [CLS] token
            except:
                return torch.randn(1, self.output_dim).to(device)
        else:
            # Fallback: keyword frequency features
            keyword_features = self._extract_keyword_features(text)
            return torch.tensor(keyword_features, dtype=torch.float32).unsqueeze(0).to(device)
    
    def _extract_keyword_features(self, text):
        """Extract keyword frequency features"""
        features = []
        text_lower = text.lower()
        
        for category, keywords in HATE_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            features.append(count / len(keywords) if keywords else 0)
        
        # Pad to 512 dimensions
        while len(features) < 512:
            features.append(0.0)
        
        return features[:512]

# ============================================================================
# VISION ENCODER - ResNet50
# ============================================================================

class VisionEncoder(nn.Module):
    """Image encoder using ResNet50"""
    def __init__(self):
        super().__init__()
        try:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except:
            self.model = models.resnet50(pretrained=True)
        
        # Remove last classification layer
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512)
        )
        self.output_dim = 512
    
    def forward(self, image):
        """Extract image features"""
        x = self.features(image)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ============================================================================
# MULTIMODAL FUSION MODEL
# ============================================================================

class MultimodalBiasDetector(nn.Module):
    """Unified multimodal architecture with soft label training"""
    def __init__(self):
        super().__init__()
        
        self.text_encoder = TextEncoder()
        self.vision_encoder = VisionEncoder()
        
        # Attention-based fusion
        combined_dim = self.text_encoder.output_dim + self.vision_encoder.output_dim
        
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=combined_dim,
            num_heads=8,
            dropout=0.3,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    def forward(self, text, image, device='cpu'):
        """Forward pass with multimodal fusion"""
        # Extract features
        text_features = self.text_encoder(text, device)
        image_features = self.vision_encoder(image)
        
        # Concatenate
        combined = torch.cat([text_features, image_features], dim=1)
        
        # Attention fusion
        combined_expanded = combined.unsqueeze(1)
        attn_out, _ = self.fusion_attention(combined_expanded, combined_expanded, combined_expanded)
        fused = attn_out.squeeze(1)
        
        # Classification
        logits = self.classifier(fused)
        return logits

# ============================================================================
# ============================================================================
# HATE SPEECH DETECTION ENGINE
# ============================================================================

class HateSpeechDetector:
    """Hate speech detection engine — loads best trained checkpoints."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_detector   = ImageBiasDetector()
        self.dataset_loader      = DatasetLoader()

        self.model        = None
        self.tokenizer    = None
        self.img_transform = None
        self.model_type   = 'keyword'  # updated by _load_model

        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self):
        """Try multimodal → BERT → CNN checkpoints, fall back to keywords."""
        # 1. Multimodal (best)
        mm_ckpt = os.path.join(CKPT_DIR, 'multimodal', 'multimodal_binary_best.pt')
        if os.path.exists(mm_ckpt):
            try:
                from models.multimodal.multimodal_model import (
                    MultimodalClassifier, EVAL_TRANSFORMS)
                from transformers import BertTokenizerFast
                ckpt = torch.load(mm_ckpt, map_location=self.device, weights_only=False)
                state = ckpt.get('model_state', ckpt)  # unwrap wrapper dict
                m = MultimodalClassifier(num_classes=2)
                m.load_state_dict(state)
                m.to(self.device).eval()
                self.model         = m
                self.tokenizer     = BertTokenizerFast.from_pretrained('bert-base-uncased')
                self.img_transform = EVAL_TRANSFORMS
                self.model_type    = 'multimodal'
                print(f'✓ Loaded multimodal checkpoint: {mm_ckpt}')
                return
            except Exception as e:
                print(f'⚠ Multimodal checkpoint failed: {e}')

        # 2. BERT text-only
        bert_ckpt = os.path.join(CKPT_DIR, 'bert', 'bert_best.pt')
        if os.path.exists(bert_ckpt):
            try:
                from models.bert.bert_text import BertTextClassifier
                from transformers import BertTokenizerFast
                ckpt = torch.load(bert_ckpt, map_location=self.device, weights_only=False)
                state = ckpt.get('model_state_dict', ckpt.get('model_state', ckpt))
                m = BertTextClassifier(num_classes=2)
                m.load_state_dict(state)
                m.to(self.device).eval()
                self.model      = m
                self.tokenizer  = BertTokenizerFast.from_pretrained('bert-base-uncased')
                self.model_type = 'bert'
                print(f'✓ Loaded BERT checkpoint: {bert_ckpt}')
                return
            except Exception as e:
                print(f'⚠ BERT checkpoint failed: {e}')

        # 3. CNN image-only
        cnn_ckpt = os.path.join(CKPT_DIR, 'cnn', 'cnn_binary_best.pt')
        if os.path.exists(cnn_ckpt):
            try:
                from models.cnn.cnn_image import ResNetImageClassifier, EVAL_TRANSFORMS
                ckpt = torch.load(cnn_ckpt, map_location=self.device, weights_only=False)
                state = ckpt.get('model_state', ckpt)
                m = ResNetImageClassifier(num_classes=2)
                m.load_state_dict(state)
                m.to(self.device).eval()
                self.model         = m
                self.img_transform = EVAL_TRANSFORMS
                self.model_type    = 'cnn'
                print(f'✓ Loaded CNN checkpoint: {cnn_ckpt}')
                return
            except Exception as e:
                print(f'⚠ CNN checkpoint failed: {e}')

        print('⚠ No trained checkpoints found — keyword-based detection only')

    # ------------------------------------------------------------------
    def _run_model_inference(self, text, image_path):
        """Run the loaded model and return hate probability [0, 1]."""
        if self.model is None:
            return 0.0
        try:
            with torch.no_grad():
                if self.model_type == 'multimodal':
                    enc = self.tokenizer(
                        text, padding='max_length', truncation=True,
                        max_length=96, return_tensors='pt')
                    ids   = enc['input_ids'].to(self.device)
                    mask  = enc['attention_mask'].to(self.device)
                    ttype = enc.get('token_type_ids')
                    if ttype is not None:
                        ttype = ttype.to(self.device)
                    if image_path:
                        img = Image.open(image_path).convert('RGB')
                        pix = self.img_transform(img).unsqueeze(0).to(self.device)
                    else:
                        # No image: supply a blank tensor so the BERT text encoder still runs
                        pix = torch.zeros(1, 3, 224, 224).to(self.device)
                    logits = self.model(ids, mask, pix, ttype)

                elif self.model_type == 'bert':
                    enc = self.tokenizer(
                        text, padding='max_length', truncation=True,
                        max_length=96, return_tensors='pt')
                    ids   = enc['input_ids'].to(self.device)
                    mask  = enc['attention_mask'].to(self.device)
                    ttype = enc.get('token_type_ids')
                    if ttype is not None:
                        ttype = ttype.to(self.device)
                    logits = self.model(ids, mask, ttype)

                elif self.model_type == 'cnn':
                    img    = Image.open(image_path).convert('RGB')
                    pix    = self.img_transform(img).unsqueeze(0).to(self.device)
                    logits = self.model(pix)

                else:
                    return 0.0

                return torch.softmax(logits, dim=1)[0, 1].item()
        except Exception as e:
            print(f'Inference error: {e}')
            return 0.0
    
    def extract_text_features(self, text):
        """Extract keyword/heuristic text features (used for display + fallback)."""
        features = {
            'text_length': len(text),
            'has_caps': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'keyword_count': 0,
            'keywords_detected': [],
            'hashtag_count': 0,
            'categories': {}
        }
        
        text_lower = text.lower()
        
        # Count keywords by category
        for category, keywords in HATE_KEYWORDS.items():
            detected = [kw for kw in keywords if kw in text_lower]
            features['keyword_count'] += len(detected)
            features['keywords_detected'].extend(detected)
            features['categories'][category] = len(detected)
        
        # Count hashtags
        features['hashtag_count'] = sum(1 for tag in HATE_HASHTAGS if tag in text)
        
        return features
    
    def _keyword_score(self, text_features):
        """Heuristic score derived from keywords/hashtags only."""
        s = text_features['keyword_count'] * 0.25 + text_features['hashtag_count'] * 0.15
        return min(s, 1.0)
    
    def predict(self, text, image_path):
        """Predict hate speech using trained checkpoint + supplementary signals."""
        try:
            text_features       = self.extract_text_features(text)
            image_bias_features = self.image_detector.analyze_image(image_path)
            keyword_score       = self._keyword_score(text_features)
            image_content_score = image_bias_features.get('image_bias_score', 0.0)

            has_image = bool(image_path)
            has_text  = bool(text.strip())

            # Image-only: no text provided
            if has_image and not has_text:
                ml_prob = self._run_model_inference(' ', image_path)  # blank text to BERT
                text_score    = 0.0
                image_score   = ml_prob
                combined_score = ml_prob * 0.70 + image_content_score * 0.20 + keyword_score * 0.10

            # Text-only: no image provided
            elif not has_image:
                if self.model_type in ('bert', 'multimodal'):
                    ml_prob = self._run_model_inference(text, image_path)
                    text_score    = ml_prob
                    image_score   = 0.0
                    combined_score = ml_prob * 0.85 + keyword_score * 0.15
                else:
                    ml_prob = 0.0
                    text_score    = keyword_score
                    image_score   = 0.0
                    combined_score = keyword_score

            # Both text + image
            else:
                ml_prob = self._run_model_inference(text, image_path)

                # Blend ML probability with supplementary signals
                if self.model_type == 'multimodal':
                    text_score    = ml_prob
                    image_score   = ml_prob
                    combined_score = ml_prob * 0.75 + image_content_score * 0.15 + keyword_score * 0.10
                elif self.model_type == 'bert':
                    text_score    = ml_prob
                    image_score   = image_content_score
                    combined_score = ml_prob * 0.70 + image_content_score * 0.20 + keyword_score * 0.10
                elif self.model_type == 'cnn':
                    text_score    = keyword_score
                    image_score   = ml_prob
                    combined_score = ml_prob * 0.55 + keyword_score * 0.30 + image_content_score * 0.15
                else:  # keyword-only fallback
                    text_score    = keyword_score
                    image_score   = image_content_score
                    combined_score = keyword_score * 0.60 + image_content_score * 0.40

            combined_score = min(combined_score, 0.99)

            is_hate_speech = combined_score > 0.5

            return {
                'is_hate_speech':      is_hate_speech,
                'confidence':          combined_score,
                'text_score':          text_score,
                'image_score':         image_score,
                'image_content_score': image_content_score,
                'combined_score':      combined_score,
                'keywords_detected':   text_features['keywords_detected'],
                'keyword_count':       text_features['keyword_count'],
                'hashtag_count':       text_features['hashtag_count'],
                'categories':          text_features['categories'],
                'text_features':       text_features,
                'image_bias_features': image_bias_features,
                'model_type':          self.model_type,
            }
        except Exception as e:
            return {'is_hate_speech': False, 'confidence': 0.0, 'error': str(e)}

# ============================================================================
# GUI APPLICATION
# ============================================================================

class HateSpeechDetectionGUI:
    """Main GUI Application"""
    def __init__(self, root):
        self.root = root
        self.root.title("Multimodal Hate Speech Detection System")
        self.root.geometry("1600x1000")
        self.root.resizable(True, True)
        
        # Initialize detector
        self.detector = HateSpeechDetector()
        self.current_image = None
        self.current_results = None
        
        # Setup UI
        self.setup_styles()
        self.create_widgets()
    
    def setup_styles(self):
        """Configure UI styles"""
        style = ttk.Style()
        style.theme_use('clam')

        self.colors = {
            'bg':        '#f0f4f8',
            'card':      '#ffffff',
            'fg':        '#1a202c',
            'fg_muted':  '#718096',
            'accent':    '#4a6cf7',
            'accent2':   '#3a5ce5',
            'success':   '#38a169',
            'danger':    '#e53e3e',
            'warning':   '#dd6b20',
            'info':      '#3182ce',
            'purple':    '#805ad5',
            'border':    '#cbd5e0',
            'header_bg': '#2d3748',
            'header_fg': '#ffffff',
        }
        C = self.colors

        self.root.configure(bg=C['bg'])

        style.configure('TFrame',         background=C['bg'])
        style.configure('Card.TFrame',    background=C['card'],  relief='flat')
        style.configure('TLabel',         background=C['bg'],    foreground=C['fg'],
                        font=('Segoe UI', 10) if 'win' in self.root.tk.call('tk', 'windowingsystem') else ('Helvetica Neue', 10))
        style.configure('Title.TLabel',   font=('Helvetica Neue', 18, 'bold'),
                        background=C['bg'],    foreground=C['accent'])
        style.configure('Subtitle.TLabel',font=('Helvetica Neue', 11),
                        background=C['bg'],    foreground=C['fg_muted'])
        style.configure('SectionHead.TLabel', font=('Helvetica Neue', 11, 'bold'),
                        background=C['bg'],    foreground=C['fg'])
        style.configure('CardLabel.TLabel',    background=C['card'],  foreground=C['fg'])

        # LabelFrame
        style.configure('TLabelframe',       background=C['bg'],    relief='groove')
        style.configure('TLabelframe.Label', background=C['bg'],
                        foreground=C['accent'], font=('Helvetica Neue', 10, 'bold'))

        # Buttons
        style.configure('TButton', font=('Helvetica Neue', 10),
                        background=C['card'], foreground=C['fg'],
                        padding=(8, 4), relief='flat')
        style.map('TButton',
                  background=[('active', C['accent']), ('pressed', C['accent2'])],
                  foreground=[('active', '#ffffff'),   ('pressed', '#ffffff')])

        # Combobox
        style.configure('TCombobox', fieldbackground=C['card'], background=C['card'],
                        foreground=C['fg'], arrowcolor=C['accent'])

        # Scrollbar
        style.configure('Vertical.TScrollbar', background=C['border'],
                        troughcolor=C['bg'], arrowcolor=C['fg_muted'])
    
    def create_widgets(self):
        """Create GUI components"""
        C = self.colors

        # Coloured header banner
        header = tk.Frame(self.root, bg=C['header_bg'], height=64)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="Multimodal Hate Speech Detection",
                 bg=C['header_bg'], fg=C['header_fg'],
                 font=('Helvetica Neue', 18, 'bold')).pack(side=tk.LEFT, padx=20, pady=14)
        tk.Label(header, text="NLP + CV + Soft Labels + Fairness",
                 bg=C['header_bg'], fg='#a0aec0',
                 font=('Helvetica Neue', 10)).pack(side=tk.RIGHT, padx=20)

        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)
        
        # Input frame
        input_frame = ttk.LabelFrame(main_container, text="Input", padding=(10, 8))
        input_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 8))
        self._create_input_section(input_frame)

        # Analyze button
        button_container = ttk.Frame(main_container)
        button_container.pack(fill=tk.X, expand=False, pady=(0, 12))
        self._create_analyze_button(button_container)

        # Display frame
        display_frame = ttk.LabelFrame(main_container, text="Preview & Results", padding=(10, 8))
        display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self._create_display_section(display_frame)

        # Control frame
        control_frame = ttk.LabelFrame(main_container, text="Tools & Actions", padding=(10, 6))
        control_frame.pack(fill=tk.BOTH, expand=False)
        self._create_control_section(control_frame)
    
    def _create_input_section(self, parent):
        """Create input controls"""
        C = self.colors
        ttk.Label(parent, text="Tweet / Caption Text:",
                  style='SectionHead.TLabel').pack(anchor=tk.W, pady=(0, 4))

        self.text_input = tk.Text(
            parent, height=5, font=('Helvetica Neue', 11),
            wrap=tk.WORD,
            bg='#ffffff', fg=C['fg'], insertbackground=C['fg'],
            relief='flat', highlightthickness=1,
            highlightbackground=C['border'], highlightcolor=C['accent'],
            padx=8, pady=6)
        self.text_input.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        # Image row
        image_frame = ttk.Frame(parent)
        image_frame.pack(fill=tk.X, expand=False, pady=(0, 8))

        ttk.Label(image_frame, text="Image:",
                  style='SectionHead.TLabel').pack(side=tk.LEFT)
        self.image_path_label = ttk.Label(
            image_frame, text="No image selected", foreground=C['fg_muted'])
        self.image_path_label.pack(side=tk.LEFT, padx=10)

        self.remove_img_btn = tk.Button(
            image_frame, text='✕ Remove', command=self.remove_image,
            bg=C['danger'], fg='#1a202c', relief='flat',
            font=('Helvetica Neue', 10), padx=8, pady=3,
            cursor='hand2', activebackground='#c53030', activeforeground='#1a202c')
        self.remove_img_btn.pack(side=tk.RIGHT, padx=(4, 0))
        self.remove_img_btn.pack_forget()   # hidden until an image is loaded

        sel_btn = tk.Button(
            image_frame, text='Browse…', command=self.select_image,
            bg=C['accent'], fg='#1a202c', relief='flat',
            font=('Helvetica Neue', 10), padx=10, pady=3,
            cursor='hand2', activebackground=C['accent2'], activeforeground='#1a202c')
        sel_btn.pack(side=tk.RIGHT)

    def _create_analyze_button(self, parent):
        """Create prominent Analyze button"""
        C = self.colors
        outer = tk.Frame(parent, bg=C['accent'], padx=3, pady=3)
        outer.pack(fill=tk.X, expand=True)

        self.analyze_btn = tk.Button(
            outer,
            text='  Analyze Content  ',
            command=self.analyze,
            bg=C['accent'],
            fg='#1a202c',
            font=('Helvetica Neue', 14, 'bold'),
            pady=14,
            relief='flat',
            cursor='hand2',
            activebackground=C['accent2'],
            activeforeground='#1a202c',
            borderwidth=0)
        self.analyze_btn.pack(fill=tk.X, expand=True)

        status_row = tk.Frame(parent, bg=C['bg'])
        status_row.pack(fill=tk.X, pady=(5, 0))

        self.status_label = tk.Label(
            status_row,
            text='Ready to analyze',
            fg=C['success'],
            bg=C['bg'],
            font=('Helvetica Neue', 10))
        self.status_label.pack(side=tk.LEFT, padx=4)
    
    def _create_display_section(self, parent):
        """Create display area"""
        C = self.colors
        left_frame = ttk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        right_frame = ttk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Image side ---
        ttk.Label(left_frame, text='Image Preview',
                  style='SectionHead.TLabel').pack(anchor=tk.W, pady=(0, 5))

        img_container = tk.Frame(left_frame, bg=C['border'], padx=1, pady=1)
        img_container.pack(fill=tk.BOTH, expand=True)
        self.image_display = tk.Label(
            img_container,
            text='No image loaded',
            bg='#f7fafc', fg=C['fg_muted'],
            font=('Helvetica Neue', 11),
            relief='flat')
        self.image_display.pack(fill=tk.BOTH, expand=True)

        # --- Results side ---
        ttk.Label(right_frame, text='Analysis Results',
                  style='SectionHead.TLabel').pack(anchor=tk.W, pady=(0, 5))

        text_container = tk.Frame(right_frame, bg=C['border'], padx=1, pady=1)
        text_container.pack(fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(
            text_container,
            height=25, width=62,
            font=('Menlo', 10),
            wrap=tk.WORD,
            bg='#ffffff', fg=C['fg'],
            insertbackground=C['fg'],
            relief='flat',
            padx=12, pady=10,
            spacing1=2, spacing3=2)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_container, orient=tk.VERTICAL,
                                  command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)

        # Define text tags for colour-coded output
        self.result_text.tag_configure('hate',
            foreground='#ffffff', background='#e53e3e',
            font=('Menlo', 12, 'bold'), spacing1=4, spacing3=4)
        self.result_text.tag_configure('safe',
            foreground='#ffffff', background='#38a169',
            font=('Menlo', 12, 'bold'), spacing1=4, spacing3=4)
        self.result_text.tag_configure('heading',
            foreground=C['accent'],
            font=('Menlo', 10, 'bold'))
        self.result_text.tag_configure('subheading',
            foreground=C['info'],
            font=('Menlo', 10, 'bold'))
        self.result_text.tag_configure('key',
            foreground=C['fg'], font=('Menlo', 10, 'bold'))
        self.result_text.tag_configure('val',
            foreground='#2d3748', font=('Menlo', 10))
        self.result_text.tag_configure('warn',
            foreground=C['warning'], font=('Menlo', 10, 'bold'))
        self.result_text.tag_configure('muted',
            foreground=C['fg_muted'], font=('Menlo', 9))
        self.result_text.tag_configure('divider',
            foreground=C['border'], font=('Menlo', 9))
    
    def _ctrl_btn(self, parent, text, cmd, bg=None, fg=None):
        C = self.colors
        b = tk.Button(parent, text=text, command=cmd,
                      bg=bg or C['border'], fg=fg or C['fg'],
                      font=('Helvetica Neue', 10), relief='flat',
                      padx=10, pady=5, cursor='hand2',
                      activebackground=C['accent'], activeforeground='#1a202c')
        b.pack(side=tk.LEFT, padx=4)
        return b

    def _create_control_section(self, parent):
        """Create control buttons"""
        C = self.colors
        left_buttons = tk.Frame(parent, bg=C['bg'])
        left_buttons.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._ctrl_btn(left_buttons, 'Load Dataset',   self.load_dataset)
        self._ctrl_btn(left_buttons, 'Export Results', self.export_results)

        right_buttons = tk.Frame(parent, bg=C['bg'])
        right_buttons.pack(side=tk.RIGHT)

        self._ctrl_btn(right_buttons, 'Save',  self.save_results,
                       bg=C['success'], fg='#1a202c')
        self._ctrl_btn(right_buttons, 'Clear', self.clear_all)
        self._ctrl_btn(right_buttons, 'Exit',  self.root.quit,
                       bg=C['danger'], fg='#1a202c')
    
    def select_image(self):
        """Select image file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif"),
                      ("All files", "*.*")]
        )
        if file_path:
            self.current_image = file_path
            self.image_path_label.config(text=os.path.basename(file_path),
                                        foreground='black')
            self.display_image(file_path)
            self.remove_img_btn.pack(side=tk.RIGHT, padx=(4, 0))
    
    def display_image(self, image_path):
        """Display image in GUI"""
        try:
            image = Image.open(image_path)
            image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(image)
            
            self.image_display.config(image=photo, text='')
            self.image_display.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
    
    def analyze(self):
        """Analyze text and image"""
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text and not self.current_image:
            messagebox.showwarning("Warning", "Please enter text or select an image")
            return
        
        thread = threading.Thread(target=self._run_analysis,
                                 args=(text, self.current_image))
        thread.daemon = True
        thread.start()
    
    def _run_analysis(self, text, image_path):
        """Run analysis in background"""
        C = self.colors
        try:
            self.root.after(0, lambda: self.status_label.config(
                text='Analyzing…', fg=C['warning']))

            results = self.detector.predict(text, image_path)
            self.current_results = results

            self.root.after(0, self._display_results, results)
            self.root.after(0, lambda: self.status_label.config(
                text='Analysis complete', fg=C['success']))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror('Error', str(e)))
            self.root.after(0, lambda: self.status_label.config(
                text='Error during analysis', fg=C['danger']))
    
    def _ins(self, text, *tags):
        """Insert text with optional tags into result_text."""
        self.result_text.insert(tk.END, text, tags if tags else ())

    def _display_results(self, results):
        """Display analysis results with colour-coded sections."""
        t = self.result_text
        t.config(state=tk.NORMAL)
        t.delete(1.0, tk.END)

        if 'error' in results:
            self._ins(f"  Error: {results['error']}\n", 'warn')
            t.config(state=tk.DISABLED)
            return

        image_bias = results.get('image_bias_features', {})
        is_hate    = results['is_hate_speech']

        # ── Verdict banner ──────────────────────────────────────────────────
        if is_hate:
            self._ins('  HATE SPEECH DETECTED  \n', 'hate')
        else:
            self._ins('  NON-HATEFUL CONTENT  \n', 'safe')
        self._ins('\n')

        # ── Confidence ──────────────────────────────────────────────────────
        self._ins('Confidence  ', 'heading')
        self._ins(f"{results['confidence']:.1%}\n")
        self._ins('\n')

        # ── Score breakdown ─────────────────────────────────────────────────
        self._ins('Score Breakdown\n', 'subheading')
        self._ins('  Text        ', 'key');  self._ins(f"{results['text_score']:.1%}\n", 'val')
        self._ins('  Image (CNN) ', 'key');  self._ins(f"{results['image_score']:.1%}\n", 'val')
        self._ins('  Image (OCR) ', 'key');  self._ins(f"{results['image_content_score']:.1%}\n", 'val')
        self._ins('  Combined    ', 'key');  self._ins(f"{results['combined_score']:.1%}\n", 'val')
        self._ins('\n')

        # ── Image analysis ──────────────────────────────────────────────────
        self._ins('Image Analysis\n', 'subheading')
        colors  = image_bias.get('colors', {})
        faces   = image_bias.get('faces',  {})
        self._ins('  Bias level     ', 'key')
        bias_pct = image_bias.get('image_bias_score', 0)
        bias_tag = 'warn' if bias_pct > 0.4 else 'val'
        self._ins(f"{bias_pct:.1%}\n", bias_tag)
        self._ins('  High contrast  ', 'key');  self._ins(f"{colors.get('is_high_contrast', False)}\n", 'val')
        self._ins('  Red ratio      ', 'key');  self._ins(f"{colors.get('red_ratio', 0):.1%}\n", 'val')
        self._ins('  Black ratio    ', 'key');  self._ins(f"{colors.get('black_ratio', 0):.1%}\n", 'val')
        self._ins('  Faces detected ', 'key');  self._ins(f"{faces.get('face_count', 0)}\n", 'val')

        ext = image_bias.get('extracted_text', '').strip()
        if ext:
            self._ins('  OCR text       ', 'key')
            self._ins(f"{ext[:120]}\n", 'val')
        else:
            self._ins('  OCR text       ', 'key');  self._ins('—\n', 'muted')
        self._ins('\n')

        # ── Keywords ─────────────────────────────────────────────────────────
        kws = results.get('keywords_detected', [])
        self._ins(f'Text Keywords  ({results["keyword_count"]} found)\n', 'subheading')
        if kws:
            for kw in set(kws[:10]):
                self._ins(f'  {kw}\n', 'warn')
        else:
            self._ins('  None\n', 'muted')
        self._ins('\n')

        # ── Category breakdown ───────────────────────────────────────────────
        cats = results.get('categories', {})
        self._ins('Categories\n', 'subheading')
        any_cat = False
        for cat, cnt in cats.items():
            if cnt > 0:
                self._ins(f'  {cat:<14}', 'key');  self._ins(f'{cnt}\n', 'warn')
                any_cat = True
        if not any_cat:
            self._ins('  None\n', 'muted')
        self._ins('\n')

        # ── Footer ───────────────────────────────────────────────────────────
        self._ins('─' * 55 + '\n', 'divider')
        model_labels = {
            'multimodal': 'BERT + ResNet50 (multimodal checkpoint)',
            'bert':       'BERT-base-uncased (text checkpoint)',
            'cnn':        'ResNet50 (image checkpoint)',
            'keyword':    'Keyword heuristics (no checkpoint)',
        }
        model_desc = model_labels.get(results.get('model_type', 'keyword'), 'Unknown')
        self._ins(f'Model: {model_desc}\n', 'muted')
        self._ins('Soft Labels  •  Fairness Checks  •  OCR Analysis\n', 'muted')

        t.config(state=tk.DISABLED)
        t.see('1.0')
    
    def _format_annotator_labels(self, labels):
        """Format annotator soft labels"""
        if not labels:
            return "  None"
        output = ""
        for i, label in enumerate(labels, 1):
            output += f"  • Annotator {i}: {label:.2%}\n"
        return output
    
    def _format_keywords(self, keywords):
        """Format keywords for display"""
        if not keywords:
            return "  None"
        return "\n".join([f"  • {kw}" for kw in set(keywords[:10])])
    
    def _format_categories(self, categories):
        """Format keyword categories"""
        if not categories:
            return "  None"
        output = ""
        for category, count in categories.items():
            if count > 0:
                output += f"  • {category}: {count}\n"
        return output if output else "  None"
    
    def _report_window(self, title, content_str):
        """Open a styled report popup."""
        C = self.colors
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry('900x700')
        win.configure(bg=C['bg'])

        hdr = tk.Frame(win, bg=C['header_bg'], height=44)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text=title, bg=C['header_bg'], fg=C['header_fg'],
                 font=('Helvetica Neue', 13, 'bold')).pack(side=tk.LEFT, padx=16, pady=10)

        txt = tk.Text(win, font=('Menlo', 10), wrap=tk.WORD,
                      bg='#ffffff', fg=C['fg'], relief='flat',
                      padx=12, pady=8)
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        txt.insert(1.0, content_str)
        txt.config(state=tk.DISABLED)

    def export_results(self):
        """Export results to JSON"""
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.current_results, f, indent=2, default=str)
                messagebox.showinfo("Success", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def save_results(self):
        """Save results"""
        self.export_results()
    
    def load_dataset(self):
        """Load dataset"""
        json_file = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )
        if json_file:
            messagebox.showinfo("Info", f"Dataset loaded: {os.path.basename(json_file)}")
    
    def remove_image(self):
        """Clear the selected image only"""
        C = self.colors
        self.current_image = None
        self.image_path_label.config(text='No image selected', foreground=C['fg_muted'])
        self.image_display.config(image='', text='No image loaded')
        self.remove_img_btn.pack_forget()

    def clear_all(self):
        """Clear all inputs and results"""
        C = self.colors
        self.text_input.delete(1.0, tk.END)
        self.remove_image()
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.status_label.config(text='Ready to analyze', fg=C['success'])

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = HateSpeechDetectionGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
