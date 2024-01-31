import torchvision.transforms as transforms

def get_resize(image_size):
        for i in range(20):
            if 2**i >= image_size:
                return 2**i
        return image_size
    

def img_transform(img,image_size):
    img_transform = transforms.Compose([
        transforms.Resize(get_resize(image_size)),  # Resize to the maximum size
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return img_transform(img)

