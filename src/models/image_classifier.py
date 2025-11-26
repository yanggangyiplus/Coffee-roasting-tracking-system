"""
이미지 기반 원두 상태 분류 모델 (CNN)
원두 이미지를 입력받아 배전도 상태를 분류합니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import json

from src.utils.constants import RoastLevel


class BeanImageDataset(Dataset):
    """원두 이미지 데이터셋"""
    
    def __init__(self, data_dir: str, transform=None, mode: str = "train"):
        """
        Args:
            data_dir: 데이터 디렉토리 경로
            transform: 이미지 변환
            mode: train 또는 test
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.mode = mode
        
        # 클래스 매핑
        self.class_to_idx = {
            "Green": 0,
            "Light": 1,
            "Medium": 2,
            "Dark": 3,
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 데이터 로드
        self.images = []
        self.labels = []
        
        mode_dir = self.data_dir / mode
        if mode_dir.exists():
            for class_name in self.class_to_idx.keys():
                class_dir = mode_dir / class_name
                if class_dir.exists():
                    for img_path in class_dir.glob("*.png"):
                        self.images.append(str(img_path))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 이미지 로드
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class BeanImageClassifier(nn.Module):
    """원두 이미지 분류 모델 (ResNet18 기반)"""
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        """
        Args:
            num_classes: 분류 클래스 수 (Green, Light, Medium, Dark)
            pretrained: 사전 학습된 가중치 사용 여부
        """
        super(BeanImageClassifier, self).__init__()
        
        # ResNet18 백본 사용
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # 마지막 레이어 교체
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class ImageClassifierTrainer:
    """이미지 분류 모델 학습 클래스"""
    
    def __init__(self, model_dir: str = "models"):
        """
        Args:
            model_dir: 모델 저장 디렉토리
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_to_idx = None
    
    def train(
        self,
        train_data_dir: str,
        val_data_dir: Optional[str] = None,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        save_best: bool = True
    ) -> Dict:
        """
        모델 학습
        
        Args:
            train_data_dir: 학습 데이터 디렉토리
            val_data_dir: 검증 데이터 디렉토리 (없으면 train에서 분할)
            epochs: 학습 에포크 수
            batch_size: 배치 크기
            learning_rate: 학습률
            save_best: 최고 성능 모델 저장 여부
            
        Returns:
            학습 히스토리
        """
        # 데이터 변환 정의
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 데이터셋 로드
        train_dataset = BeanImageDataset(train_data_dir, transform=train_transform, mode="train")
        
        if val_data_dir:
            val_dataset = BeanImageDataset(val_data_dir, transform=val_transform, mode="test")
        else:
            # train에서 검증셋 분할
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            val_dataset.dataset.transform = val_transform
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 클래스 매핑 저장
        self.class_to_idx = train_dataset.class_to_idx
        
        # 모델 초기화
        self.model = BeanImageClassifier(num_classes=4, pretrained=True)
        self.model = self.model.to(self.device)
        
        # 손실 함수 및 옵티마이저
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # 학습 히스토리
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        best_val_acc = 0.0
        
        print(f"학습 시작 (디바이스: {self.device})")
        print(f"학습 데이터: {len(train_dataset)}개")
        print(f"검증 데이터: {len(val_dataset)}개")
        
        for epoch in range(epochs):
            # 학습
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # 검증
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # 히스토리 저장
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 최고 성능 모델 저장
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("best_model.pth")
                print(f"  ✓ 최고 성능 모델 저장 (Val Acc: {val_acc:.2f}%)")
            
            scheduler.step()
        
        # 최종 모델 저장
        self.save_model("final_model.pth")
        
        return history
    
    def save_model(self, filename: str):
        """모델 저장"""
        model_path = self.model_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "class_to_idx": self.class_to_idx,
        }, model_path)
        print(f"모델 저장: {model_path}")
    
    def load_model(self, filename: str = "best_model.pth"):
        """모델 로드"""
        model_path = self.model_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_to_idx = checkpoint["class_to_idx"]
        
        self.model = BeanImageClassifier(num_classes=4, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"모델 로드 완료: {model_path}")


class ImageClassifierPredictor:
    """이미지 분류 예측 클래스"""
    
    def __init__(self, model_path: str = "models/best_model.pth"):
        """
        Args:
            model_path: 모델 파일 경로
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None
        
        # 모델 로드
        self.load_model(model_path)
        
        # 이미지 변환
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path: str):
        """모델 로드"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_to_idx = checkpoint["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        self.model = BeanImageClassifier(num_classes=4, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image_path: str) -> Dict:
        """
        이미지 예측
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            예측 결과 딕셔너리
        """
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 예측
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        predicted_class_idx = predicted.item()
        predicted_class = self.idx_to_class[predicted_class_idx]
        confidence = probabilities[0][predicted_class_idx].item()
        
        # 모든 클래스의 확률
        all_probs = {
            self.idx_to_class[i]: probabilities[0][i].item()
            for i in range(len(self.idx_to_class))
        }
        
        # RoastLevel 매핑
        roast_level_mapping = {
            "Green": RoastLevel.GREEN,
            "Light": RoastLevel.LIGHT,
            "Medium": RoastLevel.MEDIUM,
            "Dark": RoastLevel.DARK,
        }
        
        return {
            "predicted_class": predicted_class,
            "roast_level": roast_level_mapping.get(predicted_class, RoastLevel.GREEN),
            "confidence": confidence,
            "all_probabilities": all_probs,
        }

