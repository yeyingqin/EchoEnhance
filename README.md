# EchoEnhance
# Echocardiogram video temporal-spatial enhancement via improved dynamic time warping and diffusion model for remote cardiac ultrasound examinations
## Abstract
The mortality rates of cardiovascular diseases (CVDs) in rural areas have exceeded those in urban areas and have remained higher. Remote robotic-assisted cardiac ultrasound examination has emerged as a crucial solution to address the uneven distribution of medical resources. However, video quality obtained from remote cardiac ultrasound examinations is frequently compromised by two primary technical limitations: insufficient frame rate and low signal-to-noise ratio. This paper proposes an innovative video quality enhancement method to improve ultrasound video quality. First, we developed a reliable interpolation algorithm based on dynamic time warping (DTW) integrated with left ventricle and mitral valve semantic segmentation to address insufficient frame rate. This approach leverages the periodic characteristics of cardiac motion by moving frames from historical cycles to designated cycles, making this method more clinically trustworthy for physicians compared to conventional generative approaches. After the frame rate increment, key frames can be extracted from the temporally enhanced video. Second, we develop a robust and efficient denoising model based on a diffusion model combined with wavelet frequency enhancement for keyframe denoising, which demonstrates exceptional performance in high-noise scenarios.  External validation demonstrates that our temporal-spatial enhancement method shows significant potential in clinical echocardiogram video enhancement and further improves cardiac function assessment in remote robotic-assisted cardiac examinations, particularly for underserved remote areas.

## Code Release
🔄 **Status**: Code will be released after paper acceptance

We are committed to reproducible research. The complete source code, including:
- ValveDTW
- DiffUIR-WFAM
- Evaluation code
- The dataset can not be public, but you can ask for the information. The dataset was collected from two ultrasound machines: one is the Philips EPIQ Elite, with an original frame rate of 39-47 Hz and 800×600 resolution; the other is the GE E90, with an original frame rate of 40-50 Hz and 1016×708 resolution. Each Echocardiogram video contains at least two complete cardiac cycles.

Will be made publicly available following the acceptance of our paper.

## Updates
- [ ] Paper submitted (Date)
- [ ] Paper under review
- [ ] Paper accepted
- [ ] Code released

## Contact
For questions about this work, please contact: yc47499@um.edu.mo
