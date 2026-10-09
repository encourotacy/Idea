import numpy as np
import matplotlib.pyplot as plt

# 1 秒内采样 1000 个点
t = np.linspace(0, 1, 1000)
# 频率 5 Hz，振幅 1
y = np.sin(2 * np.pi * 5 * t)

plt.figure(figsize=(8, 4))
plt.plot(t, y)
plt.title("Sine wave, 5 Hz")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.savefig("sine_wave.png", dpi=150)
plt.show()
