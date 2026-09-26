import sys
sys.path.insert(0, r'C:\Users\Monika\FRED\ferd-sprint')
from demo_live import LandmarkEmotionDetector
import numpy as np

detector = LandmarkEmotionDetector()

def make_base():
    lm = np.zeros((478, 2))
    lm[1] = [320, 280]; lm[152] = [320, 400]
    lm[33] = [260, 250]; lm[133] = [300, 250]
    lm[362] = [340, 250]; lm[263] = [380, 250]
    lm[70] = [290, 220]; lm[336] = [350, 220]
    return lm

detector = LandmarkEmotionDetector()

# 1. NEUTRAL
lm = make_base()
lm[61] = [260, 325]; lm[291] = [380, 325]
lm[13] = [320, 310]; lm[14] = [320, 340]
lm[159] = [280, 245]; lm[145] = [280, 255]
lm[386] = [360, 245]; lm[374] = [360, 255]
lm[33] = [260, 250]; lm[133] = [300, 250]
lm[362] = [340, 250]; lm[263] = [380, 250]
lm[66] = [280, 225]; lm[296] = [360, 225]
lm[70] = [290, 220]; lm[336] = [350, 220]
lm[1] = [320, 280]; lm[152] = [320, 400]
emotions = detector.detect(lm)
print('Neutral:', {k: f'{v:.2f}' for k,v in detector.detect(lm).items()})

# HAPPY
lm = make_base()
lm[61] = [260, 300]; lm[291] = [380, 300]
lm[13] = [320, 305]; lm[14] = [320, 340]
lm[159] = [280, 245]; lm[145] = [280, 255]
lm[386] = [360, 245]; lm[374] = [360, 255]
lm[33] = [260, 250]; lm[133] = [300, 250]
lm[362] = [340, 250]; lm[263] = [380, 250]
lm[66] = [280, 225]; lm[296] = [360, 225]
lm[70] = [290, 225]; lm[336] = [350, 225]
lm[1] = [320, 280]; lm[152] = [320, 400]
emotions = detector.detect(lm)
print('Happy:', {k: f'{v:.2f}' for k,v in detector.detect(lm).items()})

# SAD
lm = np.zeros((478, 2))
lm[1] = [320, 280]; lm[152] = [320, 400]
lm[33] = [260, 250]; lm[133] = [300, 250]
lm[362] = [340, 250]; lm[263] = [380, 250]
lm[70] = [290, 220]; lm[336] = [350, 220]
lm[61] = [260, 335]; lm[291] = [380, 335]  # corners DOWN
lm[13] = [320, 310]; lm[14] = [320, 340]
lm[159] = [280, 248]; lm[145] = [280, 258]  # droopy eyes
lm[386] = [360, 248]; lm[374] = [360, 258]
lm[33] = [260, 250]; lm[133] = [300, 250]
lm[362] = [340, 250]; lm[263] = [380, 250]
lm[66] = [280, 220]; lm[296] = [360, 220]
lm[70] = [290, 220]; lm[336] = [350, 220]
lm[1] = [320, 280]; lm[152] = [320, 400]
emotions = detector.detect(lm)
print('Sad:', {k: f'{v:.2f}' for k,v in detector.detect(lm).items()})

# SURPRISE
lm = np.zeros((478, 2))
lm[1] = [320, 280]; lm[152] = [320, 400]
lm[33] = [260, 250]; lm[133] = [300, 250]
lm[362] = [340, 250]; lm[263] = [380, 250]
lm[70] = [290, 220]; lm[336] = [350, 220]
lm[61] = [260, 320]; lm[291] = [380, 320]
lm[13] = [320, 300]; lm[14] = [320, 355]  # mouth open
lm[159] = [280, 220]; lm[145] = [280, 280]  # wide eyes
lm[386] = [360, 220]; lm[374] = [360, 280]
lm[33] = [260, 250]; lm[133] = [300, 250]
lm[362] = [340, 250]; lm[263] = [380, 250]
lm[66] = [280, 185]; lm[296] = [360, 185]  # HIGH brows
lm[70] = [290, 220]; lm[336] = [350, 220]
lm[1] = [320, 280]; lm[152] = [320, 400]
lm[159] = [280, 220]; lm[145] = [280, 280]
lm[386] = [360, 220]; lm[374] = [360, 280]
emotions = detector.detect(lm)
print('Surprise:', {k: f'{v:.2f}' for k,v in detector.detect(lm).items()})

# ANGER
lm = np.zeros((478, 2))
lm[1] = [320, 280]; lm[152] = [320, 400]
lm[33] = [260, 250]; lm[133] = [300, 250]
lm[362] = [340, 250]; lm[263] = [380, 250]
lm[70] = [290, 220]; lm[336] = [350, 220]
lm[61] = [260, 330]; lm[291] = [380, 330]  # frown
lm[13] = [320, 310]; lm[14] = [320, 340]
lm[159] = [280, 248]; lm[145] = [280, 252]  # tight eyes
lm[386] = [360, 248]; lm[374] = [360, 252]
lm[70] = [295, 225]; lm[336] = [345, 225]  # inner brows CLOSE
lm[66] = [280, 230]; lm[296] = [360, 230]  # brows down
lm[1] = [320, 280]; lm[152] = [320, 400]
emotions = detector.detect(lm)
print('Anger:', {k: f'{v:.2f}' for k,v in detector.detect(lm).items()})