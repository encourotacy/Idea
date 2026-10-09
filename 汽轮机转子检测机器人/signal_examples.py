"""学生 C 在探头到货前可以先跑通的计算草稿。

用仿真数据串起四件事：闸门幅值和信噪比、速度-采样密度、
双线阵按编码器位置融合、耦合剂流量随速度的设定值。
真实仪器数据接进来时，替换 make_ascan / make_bscan，后面的计算不用改。

运行：python3 signal_examples.py
"""

import numpy as np
from scipy import signal

PROBE_HZ = 5e6
SAMPLE_HZ = 50e6
# 爬行器编码器侧的脉冲重复频率，用来估算每毫米有多少个 A 扫
PRF_HZ = 2000


def samples_per_mm(speed_mm_s, prf_hz=PRF_HZ):
    """同样的脉冲重复频率下，走得越快，单位长度上的采样点越少。"""
    return prf_hz / speed_mm_s


def make_ascan(sound_path_us=8.0, echo_amp=0.8, noise=0.02, interference_hz=80e3):
    """一帧 A 扫：5 MHz 回波 + 底噪 + 一条电机干扰谱线。"""
    duration_us = 20.0
    t = np.arange(0, duration_us * 1e-6, 1 / SAMPLE_HZ)
    envelope = np.exp(-0.5 * ((t - sound_path_us * 1e-6) / 0.4e-6) ** 2)
    echo = echo_amp * envelope * np.sin(2 * np.pi * PROBE_HZ * t)
    interference = 0.15 * np.sin(2 * np.pi * interference_hz * t)
    rng = np.random.default_rng(0)
    noisy = echo + interference + rng.normal(0, noise, t.size)
    return t, noisy, echo


def bandpass(x, low_hz=3e6, high_hz=7e6):
    """保留探头中心频率附近的频带，压低电机那条低频干扰。"""
    b, a = signal.butter(4, [low_hz, high_hz], btype="band", fs=SAMPLE_HZ)
    return signal.filtfilt(b, a, x)


def gate_peak(t, x, start_us, end_us):
    """闸门内的绝对峰值，以及它对应的时刻。"""
    gate = (t >= start_us * 1e-6) & (t <= end_us * 1e-6)
    segment = x[gate]
    index = np.argmax(np.abs(segment))
    return float(segment[index]), float(t[gate][index])


def snr(t, x, gate_us, noise_us):
    """闸门绝对峰值相对噪声窗均方根的倍数。"""
    peak, _ = gate_peak(t, x, *gate_us)
    noise_gate = (t >= noise_us[0] * 1e-6) & (t <= noise_us[1] * 1e-6)
    rms = np.sqrt(np.mean(x[noise_gate] ** 2))
    return abs(peak) / rms, peak


def spectrum_lines(x):
    """返回幅值最大的频率，用来核对干扰线有没有被滤掉。"""
    spec = np.abs(np.fft.rfft(x))
    freq = np.fft.rfftfreq(x.size, d=1 / SAMPLE_HZ)
    # 去掉直流
    spec[0] = 0
    return float(freq[np.argmax(spec)])


def depth_mm(sound_path_mm, refracted_deg):
    """一次波深度。声程沿声束方向，深度是它在壁厚方向的投影。"""
    return sound_path_mm * np.cos(np.deg2rad(refracted_deg))


def make_bscan(positions_mm, defect_at_mm=40.0, offset_mm=1.5):
    """两路扫查：正向看缺陷，反向有安装零点差，并带一个单路毛刺。"""
    forward = np.exp(-0.5 * ((positions_mm - defect_at_mm) / 3.0) ** 2)
    backward = np.exp(-0.5 * ((positions_mm - offset_mm - defect_at_mm) / 3.0) ** 2)
    forward = forward + 0.02
    backward = backward + 0.02
    # 只有正向出现的毛刺，融合时不应把它当成双向缺陷
    forward[positions_mm == 10] = 0.9
    return forward, backward


def fuse_dual_array(positions_mm, forward, backward, offset_mm, threshold):
    """按编码器位置对齐后融合。两路都超阈才算双向指示，单路超阈单独标记。"""
    back_at_forward = np.interp(positions_mm, positions_mm - offset_mm, backward, left=0, right=0)
    both = (forward >= threshold) & (back_at_forward >= threshold)
    forward_only = (forward >= threshold) & ~both
    return both, forward_only, back_at_forward


def cluster_indications(positions_mm, mask, amplitude):
    """把沿扫查方向连续超阈的点收成一个指示。"""
    indications = []
    start = None
    for i, flagged in enumerate(mask):
        if flagged and start is None:
            start = i
        if start is not None and (not flagged or i == len(mask) - 1):
            end = i if flagged and i == len(mask) - 1 else i - 1
            segment = slice(start, end + 1)
            peak_i = start + np.argmax(amplitude[segment])
            indications.append({
                "start_mm": float(positions_mm[start]),
                "end_mm": float(positions_mm[end]),
                "peak_mm": float(positions_mm[peak_i]),
                "peak_amp": float(amplitude[peak_i]),
            })
            start = None
    return indications


def target_flow_mL_min(speed_mm_s, width_mm=10.0, film_mm=0.2):
    """维持同样水膜厚度时，体积流量正比于扫查速度。"""
    mm3_per_s = width_mm * film_mm * speed_mm_s
    return mm3_per_s * 60 / 1000


def couplant_command(speed_mm_s, flow_meas, pressure_mpa, pressure_limit=1.0):
    """交给学生 B 的接口：设定值、实测值和报警。"""
    flow_set = target_flow_mL_min(speed_mm_s)
    if pressure_mpa > pressure_limit:
        alarm = "overpressure"
    elif flow_meas < 0.3 * flow_set:
        alarm = "underflow"
    else:
        alarm = "ok"
    return {
        "speed_mm_s": speed_mm_s,
        "flow_set_mL_min": round(flow_set, 3),
        "flow_meas_mL_min": flow_meas,
        "pressure_meas_MPa": pressure_mpa,
        "alarm": alarm,
    }


def main():
    t, raw, _ = make_ascan()
    filtered = bandpass(raw)
    gate = (7.0, 9.0)
    noise_window = (1.0, 3.0)
    raw_snr, raw_peak = snr(t, raw, gate, noise_window)
    filt_snr, filt_peak = snr(t, filtered, gate, noise_window)
    print("C-1 闸门峰值 / 信噪比")
    print(f"  滤波前 峰值 {raw_peak:.3f}  信噪比 {raw_snr:.1f}  主频 {spectrum_lines(raw)/1e6:.2f} MHz")
    print(f"  滤波后 峰值 {filt_peak:.3f}  信噪比 {filt_snr:.1f}  主频 {spectrum_lines(filtered)/1e6:.2f} MHz")
    print("  采样密度")
    for speed in (10, 50, 120):
        print(f"  {speed:3d} mm/s -> {samples_per_mm(speed):5.1f} 点/mm")

    positions = np.arange(0, 80, 1.0)
    forward, backward = make_bscan(positions)
    both, forward_only, aligned = fuse_dual_array(positions, forward, backward, offset_mm=1.5, threshold=0.4)
    print("C-2 双向指示")
    for item in cluster_indications(positions, both, np.maximum(forward, aligned)):
        item["depth_mm"] = round(depth_mm(sound_path_mm=12.0, refracted_deg=60), 2)
        print(" ", item)
    print("  单路毛刺位置 mm", positions[forward_only].tolist())

    print("C-4 耦合剂设定")
    print(" ", couplant_command(speed_mm_s=50, flow_meas=6.0, pressure_mpa=0.3))
    print(" ", couplant_command(speed_mm_s=50, flow_meas=0.2, pressure_mpa=0.3))
    print(" ", couplant_command(speed_mm_s=50, flow_meas=6.0, pressure_mpa=1.2))


if __name__ == "__main__":
    main()
