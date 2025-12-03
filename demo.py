import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import create_model
from flexish import Flexish
import torch.nn.functional as F


def demo_flexish_function():
    """Demo the Flexish activation function"""
    print("=== FLEXISH Activation Function Demo ===")
    
    # Create Flexish instance
    flexish = Flexish(beta_init=0.5)
    print(f"Initial beta: {flexish.get_beta():.4f}")
    
    # Test with sample input
    x = torch.linspace(-3, 3, 100)
    y = flexish(x)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), y.detach().numpy(), 'b-', linewidth=2, label='Flexish')
    plt.plot(x.numpy(), F.relu(x).numpy(), 'r--', linewidth=2, label='ReLU')
    plt.plot(x.numpy(), x * torch.sigmoid(x).numpy(), 'g:', linewidth=2, label='Swish')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Flexish Activation Function')
    plt.legend()
    plt.grid(True)
    plt.savefig('examples/flexish_plot.png')
    plt.show()
    
    print("Plot saved to examples/flexish_plot.png")
    return flexish


def load_and_predict(image_path, model_path=None):
    """Load model and make prediction on an image"""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Create model (example with 2 classes)
    model = create_model('resnet18', num_classes=2, activation='flexish')
    
    # Load weights if provided
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, 1)
        predicted_class = torch.argmax(probabilities, 1).item()
    
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probabilities.numpy()}")
    
    # Display image with prediction
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f'Prediction: Class {predicted_class}')
    plt.axis('off')
    plt.savefig('examples/output_prediction.png')
    plt.show()
    
    return predicted_class, probabilities


if __name__ == "__main__":
    import os
    
    # Create examples directory if it doesn't exist
    os.makedirs('examples', exist_ok=True)
    
    print("FLEXISH Demonstration")
    print("====================")
    
    # Demo 1: Show activation function
    flexish = demo_flexish_function()
    
    # Demo 2: Load and predict (if example image exists)
    example_image = 'examples/input_image.jpg'
    if os.path.exists(example_image):
        print("\n=== Image Prediction Demo ===")
        predicted_class, probs = load_and_predict(example_image)
    else:
        print(f"\nExample image not found at {example_image}")
        print("Please place an image at examples/input_image.jpg for prediction demo")
