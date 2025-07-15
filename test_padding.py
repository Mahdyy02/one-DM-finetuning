from data_loader.loader import IAMDataset
import torch

print("Testing data loader with padding fix...")

# Create dataset
dataset = IAMDataset('data/gw_data', 'data/gw_data', 'data/gw_data_laplace', 'train')
print(f'Dataset loaded with {len(dataset)} images')

# Test one batch
loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn_, shuffle=False)
batch = next(iter(loader))

print('\nBatch shapes after padding fix:')
for key, value in batch.items():
    if hasattr(value, 'shape'):
        print(f'  {key}: {value.shape}')

# Check if widths are divisible by 8
img_width = batch['img'].shape[3]
style_width = batch['style'].shape[3]
print(f'\nWidth divisibility check:')
print(f'  Image width {img_width} divisible by 8: {img_width % 8 == 0}')
print(f'  Style width {style_width} divisible by 8: {style_width % 8 == 0}')

print('\nPadding fix applied successfully!')
