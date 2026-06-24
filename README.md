# Our paper is accepted by Biomedical Signal Processing and Control.
# Echocardiogram video temporal-spatial enhancement via improved dynamic time warping and diffusion model for remote cardiac ultrasound examinations
## Abstract
The mortality rates of cardiovascular diseases (CVDs) in rural areas have exceeded those in urban areas and have remained higher. Remote robotic-assisted cardiac ultrasound examination has emerged as a crucial solution to address the uneven distribution of medical resources. However, video quality obtained from remote cardiac ultrasound examinations is frequently compromised by two primary technical limitations: insufficient frame rate and low signal-to-noise ratio. This paper proposes an innovative video quality enhancement method to improve ultrasound video quality. First, we developed a reliable interpolation algorithm based on dynamic time warping (DTW) integrated with left ventricle and mitral valve semantic segmentation to address insufficient frame rate. This approach leverages the periodic characteristics of cardiac motion by moving frames from historical cycles to designated cycles, making this method more clinically trustworthy for physicians compared to conventional generative approaches. After the frame rate increment, key frames can be extracted from the temporally enhanced video. Second, we develop a robust and efficient denoising model based on a diffusion model combined with wavelet frequency enhancement for keyframe denoising, which demonstrates exceptional performance in high-noise scenarios.  External validation demonstrates that our temporal-spatial enhancement method shows significant potential in clinical echocardiogram video enhancement and further improves cardiac function assessment in remote robotic-assisted cardiac examinations, particularly for underserved remote areas.

## Video Interpolation via ValveDTW

![Frame Sources](P5LELS82_256x256_enhanced_cycle3_frame_sources.png)
*Visualization of frame sources during ValveDTW interpolation process. This image shows how frames from different cardiac cycles are mapped and interpolated to enhance the temporal resolution.*

### 🎬 Demo Videos

#### 1. Original Video
[🎥 Watch Video](https://drive.google.com/file/d/1STasmR8C_WdvBUsJVxdun5fYJlW3lpbV/view?usp=drive_link)  


#### 2. Segmentation Mask
[🎥 Watch Video](https://drive.google.com/file/d/1c9gp0JsVk9k6nzWEbQPCIAYQOKp3ikhW/view?usp=drive_link) 


#### 3. The Third Cycle of the Original Cardiac Cycle
[🎥 Watch Video](https://drive.google.com/file/d/1EOeH9y9GzvkV602ULjK5ghwzerPh-P7Z/view?usp=drive_link)  


#### 4. The Third Cycle after Video Interpolation
[🎥 Watch Video](https://drive.google.com/file/d/1Zg4Ed4LKDFLLLrlSMbSmsVwt0CP_GWzK/view?usp=drive_link)  



## Code Release
🔄 **Status**: Coming soon

We are committed to reproducible research. The complete source code, including:
- ValveDTW
- DiffUIR-WFAM
- Evaluation code
- The private dataset (PLAGH) can not be public, but you can ask for the information. The dataset was collected from two ultrasound machines: one is the Philips EPIQ Elite, with an original frame rate of 39-47 Hz and 800×600 resolution; the other is the GE E90, with an original frame rate of 40-50 Hz and 1016×708 resolution. But we have another electrocardiogram dataset, which is being collected from Guilin People's Hospital. One of the co-authors is doing this job, which can be made publicly available if our paper is accepted.


## Contact
For questions about this work, please contact: yc47499@um.edu.mo

## Reference
@article{qin2026echoenhance,
  title={Echocardiogram video temporal--spatial enhancement via improved dynamic time warping and diffusion model for remote cardiac ultrasound examinations},
  author={Qin, Ye-Ying and Wong, Pak Kin and Zhao, Baoliang and Yao, Liang and Chan, In Neng and Zhang, Yun and Luo, Qiong and Wong, Chi Hong and Yan, Tao and Lv, Faqin and Hu, Ying},
  journal={Biomedical Signal Processing and Control},
  volume={125},
  pages={110795},
  year={2026},
  publisher={Elsevier}
}
