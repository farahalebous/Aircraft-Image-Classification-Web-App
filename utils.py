import torch
from torchvision import transforms
from PIL import Image
import os


IMAGE_SIZE_64 = 64
IMAGE_SIZE_224 = 224
INPUT_SIZE_64 = 64 * 64 * 3
INPUT_SIZE_224 = 224 * 224 * 3
HIDDEN_SIZE1 = 512
HIDDEN_SIZE2 = 256

# Transformation for 64x64 images (CNN/MLP)
transform_64 = transforms.Compose([
    transforms.Resize((IMAGE_SIZE_64, IMAGE_SIZE_64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])



# Transformation for 224x224 images (AlexNet/ResNet)
transform_224 = transforms.Compose([
    transforms.Resize((IMAGE_SIZE_224, IMAGE_SIZE_224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])



def preprocess_image(image_path, image_size=64):

    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    elif isinstance(image_path, Image.Image):
        image = image_path.convert('RGB')
    else:
        raise ValueError("image_path must be a string path or PIL Image")
    
    if image_size == 224:
         t = transform_224
    else:  
        t = transform_64
    
    image_tensor = t(image).unsqueeze(0)  
    return image_tensor


def load_model(model_path, model_type='cnn', num_classes=None, device='cpu'):

    from models import ImprovedCNN, MLP, resnet50, alexnet
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_type_lower = model_type.lower()
    
    if model_type_lower == 'cnn':
        model = ImprovedCNN(num_classes=num_classes)
    elif model_type_lower == 'mlp':
        model = MLP(
            input_size=INPUT_SIZE_64,
            hidden_size=HIDDEN_SIZE1,
            hidden_size2=HIDDEN_SIZE2,
            output_size=num_classes
        )
    elif model_type_lower == 'resnet':
        model = resnet50(num_classes, device)
    elif model_type_lower == 'alexnet':
        model = resnet50(num_classes, device)    
    # Load state dict
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load checkpoint
    if model_type_lower in ['resnet', 'alexnet']:
        model.load_state_dict(checkpoint, strict=False)
    else:
        # For CNN and MLP, load full state dict
        model.load_state_dict(checkpoint, strict=True)
    
    model.to(device)
    model.eval()
    
    return model


def predict_image(model, image_tensor, class_names=None, device='cpu', model_type='cnn'):
    
    image_tensor = image_tensor.to(device)
    

    if model_type.lower() == 'mlp':
        # Flatten the image: (batch, channels, height, width) 
        batch_size = image_tensor.size(0)
        image_tensor = image_tensor.view(batch_size, -1)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
    
    result = {
        'predicted_class_idx': predicted_idx,
        'confidence': confidence,
        'all_probabilities': probabilities[0].cpu().numpy().tolist()
    }
    
    if class_names:
        result['predicted_class_name'] = class_names[predicted_idx]
        result['all_predictions'] = [
            {'class': class_names[i], 'probability': prob}
            for i, prob in enumerate(result['all_probabilities'])
        ]
    
    return result

