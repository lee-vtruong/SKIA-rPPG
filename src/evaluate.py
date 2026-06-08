import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal
from src.model import PhysNetSmall

class UBFCrPPG_Eval(Dataset):
    def __init__(self, root_dir, clip_length=64, bvp_row=0):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.bvp_row = bvp_row
        self.subjects = sorted([
            s for s in os.listdir(root_dir)
            if os.path.exists(os.path.join(root_dir, s, 'frames.npy'))
            and os.path.exists(os.path.join(root_dir, s, 'mask.npy'))
        ])
        print(f"Đánh giá trên {len(self.subjects)} subjects.")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj_dir = os.path.join(self.root_dir, self.subjects[idx])
        frames_path = os.path.join(subj_dir, 'frames.npy')
        mask_path = os.path.join(subj_dir, 'mask.npy')
        gt_path = os.path.join(subj_dir, 'ground_truth.txt')

        all_frames = np.load(frames_path)      # (T, H, W, C)
        all_masks  = np.load(mask_path)        # (T, H, W)

        T = min(len(all_frames), len(all_masks))

        gt_data = np.loadtxt(gt_path)
        if gt_data.ndim == 1:
            bvp_signal = gt_data
        else:
            bvp_signal = gt_data[self.bvp_row, :]
        # Resample nếu cần (thường không cần vì đã khớp)
        if len(bvp_signal) != T:
            bvp_signal = signal.resample(bvp_signal, T)

        frames = torch.from_numpy(all_frames).permute(3, 0, 1, 2).float()
        masks  = torch.from_numpy(all_masks).unsqueeze(0).float()
        bvp    = torch.from_numpy(bvp_signal).float()
        return frames, masks, bvp

def evaluate_model(model_path, data_root, device, clip_length=64):
    model = PhysNetSmall(clip_length=clip_length).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = UBFCrPPG_Eval(data_root, clip_length=clip_length)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(len(dataset)):
            frames, masks, bvp = dataset[i]
            frames = frames.unsqueeze(0).to(device)  # (1, C, T, H, W)
            masks = masks.unsqueeze(0).to(device)    # (1, 1, T, H, W)
            T = frames.size(2)

            pred_bvp = []
            for start in range(0, T, clip_length):
                end = min(start + clip_length, T)
                chunk_frames = frames[:, :, start:end, :, :]
                chunk_masks  = masks[:, :, start:end, :, :]
                current_len = end - start
                # Nếu đoạn cuối ngắn hơn clip_length, pad thêm
                if current_len < clip_length:
                    pad_len = clip_length - current_len
                    chunk_frames = torch.nn.functional.pad(chunk_frames, (0,0,0,0,0,pad_len))
                    chunk_masks  = torch.nn.functional.pad(chunk_masks, (0,0,0,0,0,pad_len))

                pred_chunk = model(chunk_frames, chunk_masks)  # (1, clip_length)
                pred_chunk = pred_chunk.squeeze(0)             # (clip_length,)
                # Chỉ giữ lại phần không phải padding
                pred_chunk = pred_chunk[:current_len]
                pred_bvp.append(pred_chunk.cpu())

            pred_bvp = torch.cat(pred_bvp, dim=0)  # (T,)
            all_preds.append(pred_bvp)
            all_targets.append(bvp)
            print(f'{dataset.subjects[i]}: done, len={len(pred_bvp)}')

    # Nối tất cả subject
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # Kiểm tra độ dài
    print(f'Total preds: {len(preds)}, targets: {len(targets)}')

    # Tính MAE, RMSE, Pearson trên tín hiệu BVP
    mae_bvp = np.mean(np.abs(preds - targets))
    rmse_bvp = np.sqrt(np.mean((preds - targets)**2))
    pearson = np.corrcoef(preds, targets)[0, 1]

    print(f'\n===== Kết quả trên {len(dataset.subjects)} subjects =====')
    print(f'  MAE (BVP):      {mae_bvp:.4f}')
    print(f'  RMSE (BVP):     {rmse_bvp:.4f}')
    print(f'  Pearson (r):    {pearson:.4f}')

    # Lưu kết quả để phân tích sau
    np.savez('evaluation_results.npz', preds=preds, targets=targets)
    print('Đã lưu preds và targets vào evaluation_results.npz')

if __name__ == '__main__':
    model_path = 'best_model.pth'
    data_root = '/raid/hvtham/whale/SIKA/data/UBFC-rPPG/1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate_model(model_path, data_root, device, clip_length=64)