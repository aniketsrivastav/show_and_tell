import torch
import torchvision.transforms as transforms
from get_dataloader import get_loader
from model import CNNtoRNN
from PIL import Image
## We can also use TorchGeo, it is a PyTorch domain library, similar to torchvision, that provides datasets, transforms, samplers, and pre-trained models specific to geospatial data.

def predict(model, device, dataset, img):
    transform = transforms.Compose(
        [transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.50, 0.50, 0.50), (0.50, 0.50, 0.50)),]
                                    )

    model.eval()
    test_img = transform(Image.open(img).convert("RGB")).unsqueeze(0)
    print("OUTPUT:"+" ".join(model.caption_image(test_img.to(device), dataset.vocab)))


transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

loader, dataset = get_loader(
    'flickr8k/images/', 'flickr8k/captions.txt', transform=transform
    )
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = CNNtoRNN(embed_size=256, hidden_size=256, vocab_size=len(dataset.vocab), num_layers=1).to(device)
predict(model=model, device=device, dataset=dataset, img='img1.jpg')
