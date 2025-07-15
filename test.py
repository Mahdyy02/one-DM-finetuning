import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import Random_StyleIAMDataset, ContentData, generate_type
from models.unet import UNetModel
from tqdm import tqdm
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
import torchvision
import torch.distributed as dist
from utils.util import fix_seed
from torch.cuda.amp import autocast
import gc

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)

    # Set memory optimization environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    """ set multi-gpu if distributed is enabled """
    if opt.distributed:
        dist.init_process_group(backend='nccl')
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        totol_process = dist.get_world_size()
    else:
        local_rank = 0
        totol_process = 1
        if torch.cuda.is_available():
            # Parse opt.device to get device index
            if isinstance(opt.device, str) and opt.device.startswith("cuda"):
                if ":" in opt.device:
                    device_index = int(opt.device.split(":")[1])
                else:
                    device_index = 0
            else:
                device_index = int(opt.device)
            torch.cuda.set_device(device_index)

    # Force CPU mode if GPU memory is insufficient
    if opt.force_cpu or not torch.cuda.is_available():
        opt.device = 'cpu'
        print("Running on CPU due to memory constraints")

    load_content = ContentData()

    text_corpus = generate_type[opt.generate_type][1]
    with open(text_corpus, 'r') as _f:
        texts = _f.read().split()
    each_process = len(texts)//totol_process

    """split the data for each GPU process"""
    if  len(texts)%totol_process == 0:
        temp_texts = texts[local_rank*each_process:(local_rank+1)*each_process]
    else:
        each_process += 1
        temp_texts = texts[local_rank*each_process:(local_rank+1)*each_process]

    
    """setup data_loader instances"""
    style_dataset = Random_StyleIAMDataset(os.path.join(cfg.DATA_LOADER.STYLE_PATH,generate_type[opt.generate_type][0]), 
                                           os.path.join(cfg.DATA_LOADER.LAPLACE_PATH, generate_type[opt.generate_type][0]), len(temp_texts))
    
    
    print('this process handle characters: ', len(style_dataset))
    
    # Reduce batch size and num_workers for memory efficiency
    batch_size = 1 if opt.device.startswith('cuda') else 1
    num_workers = min(2, cfg.DATA_LOADER.NUM_THREADS) if opt.device.startswith('cuda') else 0
    
    style_loader = torch.utils.data.DataLoader(style_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=False,
                                                num_workers=num_workers,
                                                pin_memory=False if opt.device == 'cpu' else True
                                                )

    target_dir = os.path.join(opt.save_dir, opt.generate_type)

    diffusion = Diffusion(device=opt.device)

    """build model architecture with memory optimization"""
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM)
    
    """load pretrained one_dm model"""
    if len(opt.one_dm) > 0: 
        # Load with map_location to handle CPU/GPU differences
        checkpoint = torch.load(f'{opt.one_dm}', map_location='cpu')
        unet.load_state_dict(checkpoint)
        del checkpoint  # Free memory immediately
        print('load pretrained one_dm model from {}'.format(opt.one_dm))
    else:
        raise IOError('input the correct checkpoint path')
    
    # Move to device after loading
    unet = unet.to(opt.device)
    unet.eval()

    # Load VAE with memory optimization
    if opt.device == 'cpu':
        vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae", torch_dtype=torch.float32)
    else:
        vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae", torch_dtype=torch.float16)
    
    vae = vae.to(opt.device)
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    
    # Enable memory efficient attention if available
    if hasattr(vae, 'enable_xformers_memory_efficient_attention'):
        try:
            vae.enable_xformers_memory_efficient_attention()
        except:
            pass
    
    if hasattr(unet, 'enable_xformers_memory_efficient_attention'):
        try:
            unet.enable_xformers_memory_efficient_attention()
        except:
            pass

    # Clear memory before processing
    clear_memory()

    loader_iter = iter(style_loader)
    with torch.no_grad():
        for idx, x_text in enumerate(tqdm(temp_texts, position=0, desc='batch_number')):
            try:
                data = next(loader_iter)
                data_val, laplace, wid = data['style'][0], data['laplace'][0], data['wid']
                data_loader = []
                
                # Reduce chunk size for memory efficiency
                max_chunk_size = 112 if opt.device.startswith('cuda') else 224  # Smaller chunks for GPU
                
                # split the data into smaller chunks when the length of data is too large
                if len(data_val) > max_chunk_size:
                    for i in range(0, len(data_val), max_chunk_size):
                        end_idx = min(i + max_chunk_size, len(data_val))
                        data_loader.append((data_val[i:end_idx], laplace[i:end_idx], wid[i:end_idx]))
                else:
                    data_loader.append((data_val, laplace, wid))
                
                for chunk_idx, (data_val, laplace, wid) in enumerate(data_loader):
                    style_input = data_val.to(opt.device)
                    laplace = laplace.to(opt.device)
                    text_ref = load_content.get_content(x_text)
                    text_ref = text_ref.to(opt.device).repeat(style_input.shape[0], 1, 1, 1)
                    
                    # Reduce latent dimensions for memory efficiency
                    latent_height = style_input.shape[2]//8
                    latent_width = (text_ref.shape[1]*32)//8
                    
                    # Create smaller batches if needed
                    batch_size = min(style_input.shape[0], 2 if opt.device.startswith('cuda') else 4)
                    
                    for batch_start in range(0, style_input.shape[0], batch_size):
                        batch_end = min(batch_start + batch_size, style_input.shape[0])
                        
                        style_batch = style_input[batch_start:batch_end]
                        laplace_batch = laplace[batch_start:batch_end]
                        text_batch = text_ref[batch_start:batch_end]
                        wid_batch = wid[batch_start:batch_end]
                        
                        x = torch.randn((text_batch.shape[0], 4, latent_height, latent_width)).to(opt.device)
                        
                        # Use mixed precision only on GPU
                        if opt.device.startswith('cuda'):
                            with autocast():
                                if opt.sample_method == 'ddim':
                                    ema_sampled_images = diffusion.ddim_sample(unet, vae, text_batch.shape[0], 
                                                                            x, style_batch, laplace_batch, text_batch,
                                                                            opt.sampling_timesteps, opt.eta)
                                elif opt.sample_method == 'ddpm':
                                    ema_sampled_images = diffusion.ddpm_sample(unet, vae, text_batch.shape[0], 
                                                                            x, style_batch, laplace_batch, text_batch)
                                else:
                                    raise ValueError('sample method is not supported')
                        else:
                            # CPU inference without autocast
                            if opt.sample_method == 'ddim':
                                ema_sampled_images = diffusion.ddim_sample(unet, vae, text_batch.shape[0], 
                                                                        x, style_batch, laplace_batch, text_batch,
                                                                        opt.sampling_timesteps, opt.eta)
                            elif opt.sample_method == 'ddpm':
                                ema_sampled_images = diffusion.ddpm_sample(unet, vae, text_batch.shape[0], 
                                                                        x, style_batch, laplace_batch, text_batch)
                            else:
                                raise ValueError('sample method is not supported')
                        
                        # Process and save images
                        for index in range(len(ema_sampled_images)):
                            im = torchvision.transforms.ToPILImage()(ema_sampled_images[index])
                            image = im.convert("L")
                            out_path = os.path.join(target_dir, wid_batch[index][0])
                            os.makedirs(out_path, exist_ok=True)
                            image.save(os.path.join(out_path, x_text + ".png"))
                        
                        # Clear memory after each batch
                        del x, ema_sampled_images, style_batch, laplace_batch, text_batch, wid_batch
                        clear_memory()
                    
                    # Clear memory after each chunk
                    del style_input, laplace, text_ref
                    clear_memory()
                    
            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA OOM error at text {idx}: {e}")
                print("Clearing memory and continuing...")
                clear_memory()
                continue
            except Exception as e:
                print(f"Error processing text {idx}: {e}")
                continue

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='Generated', help='target dir for storing the generated characters')
    parser.add_argument('--one_dm', dest='one_dm', default='', required=True, help='pre-train model for generating')
    parser.add_argument('--generate_type', dest='generate_type', required=True, help='four generation settings:iv_s, iv_u, oov_s, oov_u')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for test')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--sample_method', type=str, default='ddim', help='choose the method for sampling')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed mode')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU mode for memory constrained systems')
    opt = parser.parse_args()
    main(opt)