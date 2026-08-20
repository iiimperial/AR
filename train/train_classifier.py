import argparse
import json
import numpy as np
import torch
from pathlib import Path
from models.unet_spm import UNetSPM
from models.cnn_lstm import ResNetLSTM
from data.preprocess import load_dicom_video, mask_scan_sector, normalize_intensity, sample_uniform_frames, resize_frame
from inference.postprocess import mask_postprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_ckpt", type=str, required=True)
    parser.add_argument("--cls_ckpt", type=str, required=True)
    parser.add_argument("--input_dicom", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seg_model = UNetSPM(in_channels=1,num_classes=1).to(device)
    seg_model.load_state_dict(torch.load(args.seg_ckpt, map_location=device))
    seg_model.eval()

    cls_model = ResNetLSTM(num_classes=2,lstm_hidden=256).to(device)
    cls_model.load_state_dict(torch.load(args.cls_ckpt, map_location=device))
    cls_model.eval()

    raw_frames = load_dicom_video(args.input_dicom)
    sel_idx = sample_uniform_frames(raw_frames.shape[0],n_sample=16)
    video_input = []
    for idx in sel_idx:
        f = raw_frames[idx]
        f = mask_scan_sector(f)
        f = normalize_intensity(f)
        f = resize_frame(f,(256,256))
        video_input.append(f)

    video_np = np.stack(video_input,axis=0)
    masked_frames = []
    with torch.no_grad():
        for f in video_np:
            inp = torch.from_numpy(f).unsqueeze(0).unsqueeze(0).float().to(device)
            logit = seg_model(inp)
            prob = torch.sigmoid(logit).cpu().numpy()[0,0]
            mask = mask_postprocess(prob)
            masked_img = f * mask
            masked_frames.append(masked_img)
    input_seq = np.stack(masked_frames,axis=0)
    input_seq = torch.from_numpy(input_seq).unsqueeze(0).unsqueeze(2).float().to(device)
    with torch.no_grad():
        logits = cls_model(input_seq)
        prob = torch.softmax(logits,dim=1).cpu().numpy()[0]
        pred_label = int(np.argmax(prob))
    result = {
        "apical_rocking_pred": pred_label,
        "prob_no_ar": float(prob[0]),
        "prob_ar": float(prob[1])
    }
    with open(args.output_json,"w",encoding="utf‑8") as fw:
        json.dump(result,fw,indent=2)

if __name__ == "__main__":
    main()