import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF


# Create CNN Model
class CNN(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units*7*7,
                out_features=output_shape
            )
        )

    def forward(self, x):
        x = self.cnn_block_1(x)
        x = self.cnn_block_2(x)
        x = self.classifier(x)
        return x

def cnn_predict(image_path):
    device = "cpu"
    save_path = 'models/CNNv2.pth'
    model = torch.load(save_path, map_location=torch.device('cpu')).to(device)

    # Define the transformation to be applied to the input image
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize the image
        transforms.ToTensor()  # Convert the image to a tensor
    ])

    # Load and preprocess the PNG image
    # image_path = 'static/5.png'
    image = Image.open(image_path).convert('RGB')  # Convert the image to RGB if needed
    image = transform(image)

    #convert from rgb to grayscale
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).unsqueeze(1).unsqueeze(2)
    grayscale_tensor = torch.sum(image * weights, dim=0, keepdim=True)
    image = grayscale_tensor.unsqueeze(0)

    # Determine if the background is white based on the mean pixel value
    mean_value = torch.mean(image)
    is_white_background = mean_value > 0.5

    if is_white_background:
        print("background is white")
        image = 1 - image

    # Convert the tensor to a PIL Image
    final_image_path  = 'static/finalimage.png'
    finalimage_pil = TF.to_pil_image(image.squeeze(0))
    finalimage_pil.save(final_image_path)

    with torch.no_grad():
        pred_logit = model(image)
        pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
        # print(f"raw pred: {pred_prob}")

    #calculate confidence
    confidence, index = torch.max(pred_prob, dim=0)
    conf = f"{confidence*100:.1f}%"
    idx = index.item()

    return conf, idx, final_image_path