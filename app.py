from flask import Flask, request, redirect, send_from_directory, url_for, send_file, render_template
import os
import numpy as np
from scipy.io import wavfile
from scipy.fft import ifft
import scipy.signal as signal
from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# 오디오 파일 읽기 함수
def read_audio_file(file_path):
    audio = AudioSegment.from_file(file_path)
    fs, data = wavfile.read(file_path)
    return audio, fs, data

# 스테레오 데이터를 각 채널로 분리하는 함수
def separate_stereo_channels(data):
    left_channel = data[:, 0]
    right_channel = data[:, 1]
    return left_channel, right_channel

# 시간 축 생성 함수
def create_time_axis(data, fs):
    return np.arange(0, len(data)) / fs

# 노치 필터의 옥타브 폭 설정 함수
def get_octave_width(notch_freq):
    octave_ratio = 2 ** (1/12)  # 반음 간격의 비율
    options = {
        1: 12,
        2: 6,
        3: 3
    }
    choice = int(request.form['notch_width'])
    if choice in options:
        return notch_freq * (octave_ratio ** options[choice] - octave_ratio ** (-options[choice]))

# 노치 필터 적용 함수
def apply_notch_filter(data, notch_freq, notch_width, fs):
    quality_factor = notch_freq / notch_width
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    return signal.filtfilt(b, a, data)

# 구간별 평균 볼륨 계산 함수
def calculate_segment_volume(audio, segment_duration_ms):
    segments = []
    for i in range(0, len(audio), segment_duration_ms):
        segment = audio[i:i + segment_duration_ms]
        segments.append(segment.dBFS)
    return segments

# 핑크 노이즈 생성 함수
def generate_pink_noise(duration_ms, fs):
    samples = int(duration_ms * fs / 1000.0)
    uneven = samples % 2
    X = np.random.randn(samples // 2 + 1 + uneven) + 1j * np.random.randn(samples // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)  # Power spectrum density
    y = (ifft(X / S)).real
    if uneven:
        y = y[:-1]
    return np.int16(y * 32767 / np.max(np.abs(y)))

# 밴드패스 필터 적용 함수
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

# 필터링된 데이터를 저장하는 함수
def save_filtered_data(filtered_data, file_path, fs):
    wavfile.write(file_path, fs, filtered_data)

# 구간별 평균 볼륨 계산 함수
def calculate_segment_volume(audio, segment_duration_ms):
    segments = []
    for i in range(0, len(audio), segment_duration_ms):
        segment = audio[i:i+segment_duration_ms]
        segments.append(segment.dBFS)
    return segments

@app.route('/static/<path:filename>')
def serve_static_from_templates(filename):
    return send_from_directory('static', filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and (file.filename.endswith('.wav') or file.filename.endswith('.mp3')):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 파일이 저장되었는지 확인
            if not os.path.exists(file_path):
                print(f'File {file_path} not found after saving.')
                return redirect(request.url)

            # MP3 파일을 WAV 파일로 변환
            if file.filename.endswith('.mp3'):
                mp3_audio = AudioSegment.from_mp3(file_path)
                wav_filename = os.path.splitext(filename)[0] + '.wav'
                wav_file_path = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)
                mp3_audio.export(wav_file_path, format='wav')
                os.remove(file_path)  # 원본 MP3 파일 삭제
                return redirect(url_for('process_file', filename=wav_filename))
            else:
                return redirect(url_for('process_file', filename=filename))
    return render_template('upload.html')

@app.route('/process/<filename>', methods=['GET', 'POST'])
def process_file(filename):
    if request.method == 'POST':
        notch_freq = int(request.form['notch_freq'])
        notch_width = get_octave_width(notch_freq) #float(request.form['notch_width'])
        ear_choice = int(request.form['ear_choice'])
        sound_choice = int(request.form['sound_choice'])
        if sound_choice == 1:
            wave_type = int(request.form['wave_type'])
            wave_freq_diff = {1: 10, 2: 20, 3: 40}.get(wave_type, 10)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio, fs, data = read_audio_file(file_path)
        
        left_channel, right_channel = separate_stereo_channels(data)

        # 채널마다 필터 적용
        filtered_data_left = apply_notch_filter(left_channel, notch_freq, notch_width, fs)
        filtered_data_right = apply_notch_filter(right_channel, notch_freq, notch_width, fs)
        
        # 데이터를 int16 형식으로 변환
        filtered_data_left = np.int16(filtered_data_left / np.max(np.abs(filtered_data_left)) * 32767)
        filtered_data_right = np.int16(filtered_data_right / np.max(np.abs(filtered_data_right)) * 32767)
        
        filtered_data = np.column_stack((filtered_data_left, filtered_data_right))
        
        filtered_file_path = os.path.join(app.config['PROCESSED_FOLDER'], 'filtered.wav')
        save_filtered_data(filtered_data, filtered_file_path, fs)
        
        # 필터링된 오디오 불러오기
        filtered_audio = AudioSegment.from_file(filtered_file_path)
        
        # 구간별 평균 볼륨 계산
        segment_duration_ms = 5000  # 5초 단위로 구간 나누기
        segment_volumes = calculate_segment_volume(filtered_audio, segment_duration_ms)
        
        # 기존 오디오의 평균 볼륨 측정
        average_dbfs = filtered_audio.dBFS
        
        # 밴드패스 필터 설정
        lowcut = notch_freq - notch_width / 2
        highcut = notch_freq + notch_width / 2
        
        combined_audio = AudioSegment.silent(duration=len(filtered_audio))
        
        if sound_choice == 1:
            # 이명 주파수에 따른 바이노럴 비트 주파수 계산
            frequency_left, frequency_right = (notch_freq, notch_freq + wave_freq_diff) if ear_choice == 1 else (notch_freq + wave_freq_diff, notch_freq)
            # 각 구간의 평균 볼륨을 기반으로 사인파 합성
            for i, segment_volume in enumerate(segment_volumes):
                # 현재 구간의 길이를 계산 (구간의 길이가 남은 오디오 길이를 초과하지 않도록 함)
                duration_ms = min(segment_duration_ms, len(filtered_audio) - i * segment_duration_ms)
                # 현재 구간의 시작 시간(ms)
                start_ms = i * segment_duration_ms

                # 채널 별 사인파 생성 및 볼륨 조절 (구간 볼륨 - 5dB)
                sine_right = Sine(frequency_right).to_audio_segment(duration=duration_ms).apply_gain(segment_volume - 5)
                sine_left = Sine(frequency_left).to_audio_segment(duration=duration_ms).apply_gain(segment_volume - 5)
                # 왼쪽과 오른쪽 사인파를 결합하여 스테레오 사인파 생성
                stereo_sine = AudioSegment.from_mono_audiosegments(sine_left, sine_right)

                # 기존 오디오의 현재 구간과 사인파를 합성
                combined_audio = combined_audio.overlay(filtered_audio[start_ms:start_ms + duration_ms], position=start_ms)
                # 스테레오 사인파를 기존 오디오에 합성
                combined_audio = combined_audio.overlay(stereo_sine, position=start_ms)
        elif sound_choice == 2:
            white_noise = WhiteNoise().to_audio_segment(duration=len(filtered_audio)).apply_gain(average_dbfs - 10)
            white_noise_data = np.array(white_noise.get_array_of_samples(), dtype=np.float32)
            bandpass_white_noise = bandpass_filter(white_noise_data, lowcut, highcut, fs)
            bandpass_white_noise_audio = AudioSegment(bandpass_white_noise.astype(np.int16).tobytes(), frame_rate=fs, sample_width=white_noise.sample_width, channels=1)
            combined_audio = filtered_audio.overlay(bandpass_white_noise_audio)
        elif sound_choice == 3:
            duration_ms = len(filtered_audio)
            pink_noise_data = generate_pink_noise(duration_ms, fs).astype(np.float32)
            bandpass_pink_noise = bandpass_filter(pink_noise_data, lowcut, highcut, fs)
            pink_noise = AudioSegment(bandpass_pink_noise.astype(np.int16).tobytes(), frame_rate=fs, sample_width=2, channels=1)
            pink_noise = pink_noise.apply_gain(average_dbfs - 10)
            combined_audio = filtered_audio.overlay(pink_noise)
        else:
            return "유효하지 않은 선택입니다."
        
        output_filename = f'processed_{os.path.splitext(filename)[0]}.mp3'
        output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        combined_audio.export(output_filepath, format="mp3")
        return redirect(url_for('download_file', filename=output_filename))

    
    return render_template('process.html', filename=filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), as_attachment=True, mimetype='audio/mp3')


if __name__ == '__main__':
    app.run(debug=True,)