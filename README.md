# STLM

The code of paper: A SAM-guided Two-stream Lightweight Model for Anomaly Detection. 

Install Mobile Segment Anything to 'mobile_sam' folder. We employ the image encoder and modify the mask decoder, using two-layer features and removing some unneeded codes.

Visualization results

<p float="center">
  <img src="images/visual.png?raw=true" width="99.1%" />
</p>

<p float="center">
  <img src="images/visA.png?raw=true" width="99.1%" />
</p>

## Core Code

```python
twostream.train()
            segmentation_net.train()
            tlm_optimizer.zero_grad()
            seg_optimizer.zero_grad()
            img_origin = data["img_origin"].to(device)
            img_pseudo = data["img_aug"].to(device)
            mask = data["mask"].to(device)

            pfeature1, pfeature2 = fix_teacher(img_pseudo)
            dfeature1, dfeature2 = fix_teacher(img_origin)
            Tpfeature = [pfeature1, pfeature2]
            Tdfeature = [dfeature1, dfeature2]
            Pfeature, Dfeature = twostream(img_pseudo)

            outputs_Tplain = [
                l2_norm(output_p.detach()) for output_p in Tpfeature
            ]
            outputs_Tdenoising = [
                l2_norm(output_d.detach()) for output_d in Tdfeature
            ]

            outputs_Splain = [
                l2_norm(output_p) for output_p in Pfeature
            ]
            outputs_Sdenoising = [
                l2_norm(output_d) for output_d in Dfeature
            ]

            output_pain_list = []
            for output_t, output_s in zip(outputs_Tplain, outputs_Splain):
                a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
                output_pain_list.append(a_map)

            output_denoising_list = []
            for output_t, output_s in zip(outputs_Tdenoising, outputs_Sdenoising):
                a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
                output_denoising_list.append(a_map)

            output = torch.cat(
                [
                    F.interpolate(
                        -output_p * output_d,
                        size=outputs_Sdenoising[0].size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for output_p, output_d in zip(outputs_Splain, outputs_Sdenoising)
                ],
                dim=1,
            )

            output_segmentation = segmentation_net(output)

```

## Requirements

##### Our code is based on [DeSTSeg](https://github.com/apple/ml-destseg) and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM).

Thanks for their contributions.



# Citation

```tex
@article{STLM,
author = {Li, Chenghao and Qi, Lei and Geng, Xin},
title = {A SAM-guided Two-stream Lightweight Model for Anomaly Detection},
year = {2024},
journal = {ACM Transactions on Multimedia Computing Communications and Applications},
}


```
