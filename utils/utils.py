
def train_valid_split_from_csv(csv_file, root_dir):
    '''Helper function to split training and validation sets
        Parameters:
        - (str) csv_file: Path to csv file w/ image filenames
        - (str) root_dir: Path to image data
    '''
    df = pd.read_csv(os.path.join(root_dir, csv_file))
    train, valid = df.loc[df.valid == 0], df.loc[df.valid == 1]
    train.to_csv(os.path.join(root_dir, 'train_sample.csv'),
                 sep=',', header=True, index=False)
    valid.to_csv(os.path.join(root_dir, 'valid_sample.csv'),
                 sep=',', header=True, index=False)


def remove_bad_images(csv_file, root_dir):
    '''Helper function to drop bad images from dataset
    Parameters:
        - (str) csv_file: Path to csv file w/ image filenames
        - (str) root_dir: Path to image data
    '''
    df = pd.read_csv(os.path.join(root_dir, csv_file))
    for idx, img in enumerate(df.image):
        try:
            Image.open(os.path.join(root_dir, img))
        except IOError as e:
            df.drop(index=idx, inplace=True)
    
    df.to_csv(os.path.join(root_dir, 'sample_clean.csv'),
              sep=',', header=True, index=False)

# Helper function to show samples in batch
def show_img_batch(sample_batch):
    '''Show images w/ label from a batch of the sample'''
    imgs, labels = sample_batch['image'], sample_batch['label']
    batch_size = len(sample_batch)
    img_size = imgs.size(1)
    grid = torchvision.utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.show()
    
# Helper function to plot an image in a dataset
def show_img(sample):
    '''
    Plot image samples from dataset
    Parameters:
    - (dict) sample: sample from dataset 
    '''
    ax = plt.subplot()
    plt.tight_layout()
    ax.set_title('{}'.format(sample['label']))
    ax.axis('off')
    plt.imshow(sample['image'])
    plt.show()
    
# Create custom transforms to apply to dataset
def imagenet_transforms(size=(224, 224)):
    '''
    Helper function to apply imagenet transforms to dataset
    Parameters:
    - (int, tuple) size: 
    '''
    imagenet = [transforms.Resize(size=size),  # Resize image
                transforms.ToTensor(),  # Convert numpy image to torch image
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Imagenet normalization
                                     std=[0.229, 0.224, 0.225])]

    return torchvision.transforms.Compose(imagenet)
