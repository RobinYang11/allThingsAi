

import speech_recognition as sr

# 初始化语音识别器
recognizer = sr.Recognizer()

# 使用麦克风作为音频源
with sr.Microphone() as source:
    print("请说话...")
    # 调整麦克风噪声
    recognizer.adjust_for_ambient_noise(source)
    # 录制音频
    audio = recognizer.listen(source)

    
     